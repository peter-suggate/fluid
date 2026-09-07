# Simulation variation points

Each method owns its composition. Choice metadata in `features/algorithms/definition.ts`
is shared by its method parameter schema and the composition validator. The adapter in
`../core/method-parameter-variants.ts` accepts only explicit choice schemas; it does not infer variants
from numeric settings or scan solver implementation flags.

| Method | Supported dimensions |
| --- | --- |
| Uniform CM12 | Dense surface publication; fixed CM11a LCP multigrid pressure; dense/work-box dispatch; gamma diffusion; interface sharpening; local mass return; partial-solid excess correction; rigid coupling; semi-Lagrangian/MacCormack velocity transport; liquid-only/unrestricted momentum sampling; paper/scene timestep; render reconstruction policy |
| Adaptive-mass | Sparse surface publication; fixed sparse Jacobi-PCG pressure; surface/activity/coarse-first adaptivity; gamma diffusion; conservative sharpening; paper/scene timestep |
| Losasso | Octree or 4×/8× fine-band surface; fixed V-cycle MGPCG pressure; fixed-Jacobi/causal-front air velocity extension |
| Power Liquids | Octree or 4×/8× fine-band surface; fixed Power 2017 hybrid-preconditioned pressure |

Pressure implementations remain coupled to their methods. A numerical budget or
smoothing count is not a preconditioner choice. Surface ladder/refinement ratios,
leaf size and topology cadence are resolution/work controls rather than separate
algorithm identities. Pressure-journal reservation is instrumentation capacity.

Turning off uniform interface sharpening makes the selected mass-return setting
inactive; it does not make that stored combination invalid. Re-enabling sharpening
restores the remembered setting through the existing solver adapter. `scene` render
reconstruction is a policy selection, not a new numerical solver.

The current product exposes no further selectable pressure, preconditioner or
transport implementations in the octree and sparse CM12 methods. Hidden QA seams
(frozen topology, legacy face transport, forced surface rungs, alternative packet
or owner arithmetic and capacity-repair experiments) remain diagnostic constructor
inputs, outside the product composition. Algorithm implementation extraction beyond
the adaptivity package remains domain-owned in the existing solver modules; the
composition describes actual choices without claiming those monoliths are decomposed.
