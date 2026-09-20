# Plan: active-set-aware pressure multigrid, first in 2D

The full-depth bounds and pressure acceptance/recovery safeguard are prerequisites,
not substitutes for this work. Keep them enabled throughout development. The
immediate objective is to make coarse corrections compatible with the fine-grid
pressure inequalities, reducing rejected cycles and recovery cost without changing
the converged physical pressure problem.

The pressure equations are a complementarity problem. With A p = b and lower bound
l, feasible p satisfies p >= l. Free rows require A p - b = 0; a row at l permits
A p - b >= 0. Its ordinary equation residual can therefore be negative without an
error in the constrained solution. Restricting that residual into an unconstrained
coarse correction was the Figure 9 failure. Carrying bounds throughout prevents
that specific mismatch, but max-restriction of bounds combined with geometric
prolongation can overconstrain a coarse level, particularly around embedded bodies.
The live buoyancy scene already exercises the bounded recovery path repeatedly.

Implement the following in order. Each step produces measured evidence before the
next step changes the algorithm.

1. **Build a frozen-system corpus and a reliable fine-grid reference.** Capture
   the pre-projection matrix coefficients, RHS, bounds, phi/topology, dt/rho and
   dimensions. Include Figure 9 frame 104 from the original failing revision,
   a quiet pool, separating ceiling/side walls, an embedded solid, the submerged
   moving-body fixture, disconnected pools, near-vanishing surface rows, and
   odd/semi-coarsened dimensions. Add a tightly converged projected fine-grid
   reference for small systems. Measure projected residual, feasibility,
   complementarity gap, pressure-gradient/velocity error, active-set changes,
   and runtime separately. Pressure comparisons must account for any genuine
   component nullspace; velocity gradients are the physical comparison.
2. **Prototype a truncated active-set V-cycle in Rust 2D.** After pre-smoothing,
   classify a row as bound-active only when p is at its lower bound within a
   scale-aware roundoff threshold and its equation residual points below that
   bound. Distinguish these rows from air, which is not a pressure unknown.
   Freeze the classification for one correction cycle, constrain correction
   to zero on active rows, and form residual/restriction on the remaining free
   rows. Permit the next smoothing/classification step to release a row; do not
   permanently freeze contact. First implement a V-cycle; defer Full-Cycle
   integration until the correction operator is validated.
3. **Make transfer and the coarse operator agree.** Construct truncated
   prolongation P on free unknowns and its corresponding volume-weighted
   restriction R. Compare a Galerkin A_c = R A_f P reference with the current
   rediscretized coarse operator. Handle disconnected components, disappearing
   free rows and halos explicitly. A coarse correction must preserve fine
   feasibility under prolongation: a maximum over geometric children alone is
   not a proof of that property for overlapping interpolation stencils. Use a
   projected/damped correction and a feasible step bound; keep the existing
   finest residual acceptance gate as the final authority. Do not merely zero
   negative residuals while leaving the old transfer and operator unchanged.
4. **Integrate Full-Cycles and bounded recovery.** Apply the same active-set
   rules to nested initialization and shifted bounds, not just ordinary
   V-cycles. Retain the dedicated accepted-pressure storage, rejection counts,
   initial/accepted residuals, and honest recovery-exhaustion reporting.
   Measure whether safeguarding needs damping before rejection, using the
   same constrained merit function and a bounded number of trials.
5. **Port the accepted variant to GPU 2D, then ordinary 3D.** Start with a dense
   implementation and one physical Z cell; verify each frozen-system stage
   against Rust. Then cover actual 3D boundaries and embedded bodies, and only
   afterwards optimize active windows, tile scheduling and memory reuse.
   GPU reductions must reject NaN/Inf and preserve the last accepted field
   without CPU readback or synchronization in the simulation loop.

Acceptance requires all of the following:

- Converged reference pressure gradients and complementarity, including active
  rows becoming free again; preserving a static active set is insufficient.
- No published non-finite pressure and no accepted increase over the initial
  constrained residual. Inject bad finite, NaN and infinite cycle results and
  verify restoration, recovery, counter reset and exhaustion behavior.
- Figure 9 through at least six seconds in native/scalar/SIMD 2D and 128×128×64
  GPU 3D, plus hydrostatics, both side walls, separating ceilings and live bodies.
  Preserve the existing mass, symmetry and one-step parity bounds.
- Fewer rejected cycles/recovery sweeps on the corpus, including moving bodies.
  Report full-solve wall time and GPU dispatch/memory cost; a lower iteration
  count alone does not establish an improvement. Compare at matched residuals.
- Track phi/V mismatch and energy independently. Pressure robustness must not
  be credited with solving level-set transport or surface-volume disagreement.

The likely first decision is whether truncated transfer with rediscretization is
adequate. If it fails the frozen systems, retain the Galerkin/reference route to
identify the mismatch before optimizing storage. Keep the current safeguarded
solver as the production fallback until the complete scene and parity gates pass.
