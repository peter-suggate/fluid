# Adaptive-volume compiled topology implementation

This records implementation and validation of the [performance plan](adaptive-volume-performance-plan.md). The working tree includes earlier staged changes; the comparison baseline preserves those changes rather than using bare HEAD. No simulation tolerance, timestep, iteration ceiling, topology policy, or regression timing ceiling is changed by this work.

## Implemented architecture

CNX1 compiles a complete accepted connectivity image when the accepted topology generation or slot changes. Cells and rows have accepted ordinals, ordered row terms, and cell incidence ranges. Terms retain stable cell IDs and source ordering so consumers access existing field planes directly without changing numerical reduction order. Physical subfaces and their signed cell adjacency are compiled with the same generation and reuse the existing geometric-volume storage. A generation is published only after its required views validate.

The compiler runs at initial publication, after in-frame topology admission, and before later presentation consumers. A separate indirect-argument producer disables build dispatches on unchanged generations. Compiler entry points have their own shader dependency slice to avoid expanding the initial presentation module. The argument producer has a stable header binding and does not expose its writable argument buffer to indirect consumers.

Pressure operator, projection, collocation, generic velocity extension, and interface reconstruction use the shared connectivity. Hot loops check the generation once and then use direct accessors. The geometric transport step consumes the sealed physical graph instead of rebuilding its topology every frame.

The static-solid limiter reads one complete factor bank and writes the other. Its first update uses implicit all-one inputs, absorbing initialization; its convergence receipt retains the audited bank. Moving-solid FISTA retains its previous schedule. Continuous source geometry is computed once per active cell and reused through component gathering and rate publication. Disabled-source readers return zero. Unused interface-history buffers and copies are removed.

The pressure seed computes the initial operator image, residual, Jacobi direction, and reduction partials together. It retains the initial convergence decision and the final true-residual check. Face prediction and body forces remain separate because prediction reads neighboring source velocities; writing that source in a fused dispatch would introduce a cross-workgroup race.

## Storage limits and remaining replacement work

This is a compatibility-stage representation: accepted ordinals are dense, but state planes and copied term/incidence address ranges still reserve physical capacity. Old topology construction and publication authorities remain. Consequently this implementation does **not** yet deliver the plan's accepted-sized allocation or complete retirement of the all-rung architecture.

| Initial scene | Accepted cells | Accepted rows | Terms/incidences | Added CNX storage | Complete topology arena |
| --- | ---: | ---: | ---: | ---: | ---: |
| mini32 | 4,887 | 15,006 | 29,322 | 47,619,008 B | 109,255,424 B |
| mini64 | 10,266 | 31,209 | 61,635 | 80,789,584 B | 204,674,192 B |

Both topology arenas fit the 256 MiB storage-binding limit. Removing the two unused interface histories saves 32 bytes per physical-capacity cell (7,817,216 B in initial mini32), but the new connectivity image increases total memory overall. The removed history storage must not be presented as an overall memory reduction.

The current integration builds CNX immediately after the existing topology selector commits. A compiler failure halts consumers; it does not roll back that prior selector. Moving full compilation and its certificates inside candidate pre-admission is still required for the plan's complete retain-old-on-refusal transaction.

Remaining architecture work includes accepted-sized field/storage allocation, removal of competing topology journals and replay authorities, local static-solid support certificates with explicit geometry invalidation, compiled sampling and presentation support, and replacement of capacity-shaped activity/presentation scheduling. These are full-generation replacements; CNX has no incremental repair path.

## Validation status

Validation is in progress. Targeted CPU tests cover copied connectivity, mixed-seam and boundary ordering, duplicate incidence semantics, limiter factor generations, source disable/resume behavior, velocity-extension traversal, and exact pressure-seed reduction order.

An earlier candidate gate was interrupted after Chrome was discovered running. Its timing results are excluded from acceptance. Chrome was quit normally with the user's authorization, its process absence was verified, and an isolated preserved-baseline gate was started. Final clean comparison receipts and the unchanged candidate gate will be recorded here before acceptance.
