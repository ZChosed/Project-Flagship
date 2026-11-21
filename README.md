# Multi-layer thermal control for wire-fed DED

## Goal

Use reinforcement learning (RL) on a reduced-order model (ROM) of heat transfer in a multi-layer part that grows during wire-fed DED. The RL policy generates the torch trajectory directly for each layer, as feedforward control. The aim is to make the temperature field more uniform, which gives better material properties, less warping, and possibly faster prints.

### Reference: Sideris et al.

- Sideris et al., *J. Mater. Process. Technol.* 327 (2024) 118364: path planning for uniform temperature fields in WAAM. Uses an FVM thermal model, a ROM made with balanced truncation, a steady-state response computed with the Laplace transform, and MCTS to select the order, start point, and direction of each feature.
- Sideris, Petrik, Bambach, *Manufacturing Letters* 41 (2024) 879–890 ("Too hot to print, too slow to handle"): an RL agent finds the fastest path, the path is split into segments, then MCTS optimizes the segments for thermal response.

### How this project is different

- **The RL policy generates the trajectory directly.** Sideris selects the order and direction of preset features. This project does not use preset features.
- **Speed is a smaller concern.** Training the RL policy pays the computation cost offline. At print time, one pass through the policy replaces the costly online search through the thermal model. The model must still be fast enough to give the number of rollouts that training needs.
- **Model reduction uses Component Mode Synthesis (CMS), not balanced truncation.** CMS is built for geometry that grows during deposition and stays accurate for arbitrary deposition patterns [8], [13].
- **Possible extension (not part of step 1):** keep the ROM fast enough to run online. Give its prediction to the policy as an input feature, and correct the ROM with measurements during the print. The ROM then acts as a digital twin of the substrate. This is probably too complex for step 1.

## Reference cases

| Case | Material | Purpose | Source of data |
|---|---|---|---|
| Primary | Aluminum: an aerospace Al-Mg-Sc alloy for large-format WAAM | Target application | TODO: find a public proxy (see note) |
| Comparison | ER70S-6 steel | Validation against published data | Sideris et al. (2024), Tables 1–2: 4.25 kW, absorption 0.85, 1.2 mm wire, 5 m/min wire feed, 400 mm/min travel, 2.5 mm layer height, 2 min dwell |

Note on the aluminum alloy: aerospace Al-Mg-Sc alloys for additive manufacturing are often proprietary, and large-format printers may use laser or plasma arc heat sources rather than a standard GMAW/CMT arc. TODO: select a proxy alloy with public thermal property data (for example, an Al-Mg-Sc alloy, or ER5356 as a common Al-Mg WAAM wire) and record the process parameters to use.

## Physics

> **TODO (Zachary):** write this section. Include the governing equation, every heat term (arc input, element birth, conduction, convection, radiation, contact to the table, latent heat), the boundary conditions, a parameter table with a source for each value, and the terms left out with a reason for each.

## Working design (thermal model)

A linear 3D part-scale thermal model of wire-arc DED for any part geometry, written in JAX as the environment for control and RL. Code lives in `thermal_sim/`.

| # | Component | Choice | Rationale | Source |
|---|---|---|---|---|
| 1 | Physics | Transient conduction + volumetric heat source, constant properties | Linear in temperature and arc power, so one model can simulate, be reduced and be controlled. | [1] |
| 2 | Geometry | General 3D (no thin-wall assumption) | Must work for any toolpath and part, not only single walls. | — |
| 3 | Discretization | Structured 3D voxel finite-volume grid, 7-point stencil | Voxels match layer-by-layer deposition and make activation a simple index operation. | [1], [2] |
| 4 | Growing geometry | Fixed bounding-box grid + boolean `active` mask (inactive-element method) | Activation flips mask bits only, so array shapes stay fixed for `jit`/`vmap`, and inactive cells add no ill-conditioning. | [3] |
| 5 | Boundary loss | Convection on every active face next to an inactive cell, derived from the mask | Exposed surfaces stay correct for any shape with no extra geometry code. | [3] |
| 6 | Heat source | Goldak double-ellipsoid, arc power *u* as input (efficiency η stated) | Power enters linearly through B, so power and dwell become control inputs. | [4] |
| 7 | Time integration | Implicit (backward) Euler, variable step (fine while depositing, coarse in dwell) | Unconditionally stable, so the step follows the torch, not a stability limit. The discrete energy balance is exact up to the CG tolerance. | [5] |
| 8 | Linear solver | Matrix-free preconditioned CG (Jacobi first, warm start from last step) | (C/Δt + K) is symmetric positive-definite, and matrix-free avoids 3D factorization fill-in. | [6], [7], [9] |
| 9 | Stack | JAX: `jit`, `vmap`, `grad`, `jax.scipy.sparse.linalg.cg` | GPU speed, batched RL environments and gradients for MPC from one code base. | [9]–[12] |
| 10 | Reduction (later) | Component Mode Synthesis | Built for evolving DED geometry; accurate under arbitrary deposition patterns. | [8], [13] |

**Known limits:** no temperature-dependent properties, latent heat or radiation (each breaks linearity); staircase edges on curved surfaces.

**Open decisions**
- Material and properties (k, ρ, c_p, h, melt temperature, η); see Reference cases.
- Whether step 8 (latent heat, temperature-dependent properties) is in scope, given design decision 1 (linear model).
- RL observation features (candidates: interpass temperature at next segment start, layer mean/max, a few sensor points, surface field).
- Preconditioner upgrade (AMG or multigrid) if CG needs more than ~50 iterations.

## Plan

**Energy-balance tolerance:** with implicit Euler, input = stored + losses holds exactly up to the CG tolerance (~1e-8 relative). An error near 1% means a bug.

### A. Define the problem
1. State what the model is for: the output metrics, the accuracy target, and the speed target.
2. Select the reference cases (done: aluminum primary, steel comparison).
3. Write the physics (section above).
4. Select dx and Δt from time scales: dx²/α, dx/v, layer time, cooling time.

### B. Build in layers, with a test for each
5. **Milestone 1** (can start now; needs no source or deposition, so does not wait on steps 1–4): grid + `active` mask + matrix-free masked 7-point operator, then an implicit Euler step with CG on a 50 × 50 × 20 block. Tests, in order:
   1. Operator is symmetric and positive (relative 1e-12).
   2. JAX matches a NumPy reference on a 20 × 20 × 10 grid with a random mask (1e-12).
   3. Adiabatic box (h = 0): total energy constant (~1e-10), T goes to the volume mean.
   4. Hot cube with convection: max T falls monotonically to T∞; per-step energy balance to ~1e-8.
   5. Time convergence: error falls ~2× from Δt to Δt/2 (first order).
6. Moving source on a thick plate vs the Rosenthal solution, far from the source [14], with grid convergence at 2, 1 and 0.5 mm voxels.
7. Element birth, with an energy balance check (to CG tolerance) in every run.
8. Latent heat and temperature-dependent properties (scope open; see Open decisions).
9. One layer, then several layers.

### C. Make it credible
10. Sensitivity study on the uncertain parameters.
11. Calibrate the parameters with the largest effect to published data (thin-wall interpass temperatures), and validate.

### D. Use it
12. Dwell study: 0, 30, and 60 s.
13. Make a CMS ROM, and confirm that it ranks strategies the same way as the full model.
14. Gymnasium environment and RL.

## Files

- `thermal_sim/`: JAX thermal model.
- `Papers/`: reference papers.

## References
1. C. He, N. Wood, N. B. Bugdayci, C. Okwudire, "Generalized SmartScan: An Intelligent LPBF Scan Sequence Optimization Approach for 3D Part Geometries," preprint, doi:10.20944/preprints202311.0153 (peer-reviewed version exists).
2. S. V. Patankar, *Numerical Heat Transfer and Fluid Flow*, Hemisphere, 1980.
3. P. Michaleris, "Modeling metal deposition in heat transfer analyses of additive manufacturing processes," *Finite Elements in Analysis and Design*, 86, 51–60, 2014, doi:10.1016/j.finel.2014.04.003.
4. J. Goldak, A. Chakravarti, M. Bibby, "A new finite element model for welding heat sources," *Metallurgical Transactions B*, 15(2), 299–305, 1984, doi:10.1007/BF02667333.
5. R. J. LeVeque, *Finite Difference Methods for Ordinary and Partial Differential Equations*, SIAM, 2007.
6. M. R. Hestenes, E. Stiefel, "Methods of conjugate gradients for solving linear systems," *J. Res. Natl. Bur. Stand.*, 49(6), 409–436, 1952, doi:10.6028/jres.049.044.
7. Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
8. R. Du, J. Liu, Y. Zheng, R. Ahmad, "A deposition-agnostic reduced-order thermal modeling strategy for evolving geometries in directed energy deposition," *Additive Manufacturing*, 121, 105148, 2026, doi:10.1016/j.addma.2026.105148.
9. JAX docs, `jax.scipy.sparse.linalg.cg`: https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.cg.html
10. JAX docs, `jax.jit`: https://docs.jax.dev/en/latest/_autosummary/jax.jit.html
11. JAX docs, `jax.vmap`: https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html
12. JAX docs, `jax.grad`: https://docs.jax.dev/en/latest/_autosummary/jax.grad.html
13. R. R. Craig, M. C. C. Bampton, "Coupling of substructures for dynamic analyses," *AIAA Journal*, 6(7), 1313–1319, 1968, doi:10.2514/3.4741.
14. D. Rosenthal, "The theory of moving sources of heat and its application to metal treatments," *Trans. ASME*, 68, 849–866, 1946.
