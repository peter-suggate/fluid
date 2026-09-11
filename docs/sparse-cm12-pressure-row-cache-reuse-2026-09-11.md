# Pressure-row cache reuse: simplified implementation

This extends the existing [compact pressure-row publication](sparse-cm12-compact-tile-execution-proposal-2026-09-11.md) by reusing data already present. It adds no persistent allocation or dispatch and preserves the existing membership and theta representations.

## Skip redundant publication lookups

Keep the compact list of 64-row tile IDs and the established per-row classification predicate. If topology, gravity or solids already require classification, skip scalar-change-mask reads. Read the previous membership bit only when reusing it. There is no new workgroup cache, barrier, allocation or dispatch.

The existing row-repair scratch remains **102,656 bytes** at the Figure 7 reset capacity: **816,276 reserved row slots**, rounded up to **12,755 tiles of 64 rows**, with one stamp and one list word per tile plus padded control storage. These are allocation capacities, not liquid-row counts. The simplification adds **zero bytes** of persistent or workgroup storage.

## Reuse the existing packed row word across stages

A row's existing packed term word contains a 23-bit offset and a 9-bit count. Callers that need both used separate accessors, each loading that same word. `rowTermRange` loads it once and returns the first and exclusive-end indices. No format change or new buffer is required.

| Stage | Converted traversal sites |
|---|---:|
| Transport face support/preparation | 2 |
| Gamma diffusion | 1 |
| Sharpening statistics | 1 |
| Density capacity repair: area, scatter, gather | 3 |
| Pressure cell-submersion classification | 1 |
| Pressure row classification, dirty-tile discovery and publication | 3 |
| Pressure row gradient used by the operator | 1 |
| Topology face-velocity remapping | 1 |
| Aggregate pressure compatibility | 1 |
| Closed-world contact and exterior-row replacement | 2 |
| Velocity-extension seam traversal | 1 |
| Topology semantic validation | 1 |
| Pressure diagonal construction (direct incidence term) | 1 |
| **Total** | **19** |

The range accessor removes a duplicate packed-word load at 18 sites. Velocity extension additionally keeps the own-term coefficient through its neighbour loop. Pressure diagonal construction reads the incidence term directly, removing the offset-to-ordinal-to-offset round trip. Runtime rows avoid repeating the lookup behind the accessor. The offset-only accessor remains for inflow; the unused count accessor, old resident hot-term/ordinal adapters and separate ISA offset/count adapters were removed. This does not claim a tile/halo conversion of every simulation field.

Velocity projection also checks zero theta before the separating-wall lookup. The unused production full-domain publication pipeline and unused cache-generator options were removed; the reference shader entry remains available to tests.

## Why keep the representation simple?

Accepted-row discovery does not enumerate every eligible template/seam row inside a selected tile. Replacing the tile's established predicate with discovery's dirty bits changes membership. Supporting that requires additional coverage masks and fallback logic. The implementation therefore keeps whole-tile evaluation and removes that experimental machinery. The experimental theta/membership packing was also removed. There are no new per-row tags, dirty/coverage records, buffer clears or GPU stages.

## Verification and measurement

Focused GPU tests cover compact versus full publication, reuse, gravity, solids, retirement, recycling, partial words, active zero response, dense masks, eligible rows outside the discovery stream, face support, remapping and mixed-seam behavior. Existing packed-row accessor tests also run.

The Figure 7 A/B uses radius 0.1 m, scene timestep, production defaults, pressure tolerance 0.194, 8 warmup frames and 24 measured frames. The control is the compact-tile implementation before these reuse changes. Tests run serially with the browser simulation unloaded.

Terminal field hashes now request the complete accepted world, including runtime pages, in spatial coordinates. Previous authored-only hashes were insufficient. Raw row-slot hashes can vary across unchanged runs because allocation IDs vary and are not a spatial equivalence test. The optional authored coefficient hash is explicitly scoped and bounds its immutable catalogue traversal to authored cells; this is QA-only work after measurement.

### Measurement status

The explicit 16-byte workgroup metadata cache was rejected: two opposite-order A/B pairs measured control/reuse advance medians of **13.3693 / 13.8281 ms** and **13.1072 / 13.3038 ms**. Row publication stayed **0.1966 ms** in all four. That version passed all 17 canonical lanes, but did not justify its extra barrier. It has been removed from the final implementation.

The final version retains only the packed-range accessor, conditional lookup elimination and unused-code removal. The final Figure 7 capture measures **13.2383 ms advance**, **4.0632 ms pressure** and **8.9784 ms non-pressure**, versus the nearest unchanged control’s **13.1072 / 3.9322 / 8.9129 ms**. Publication remains **0.1966 ms**. This is not a demonstrated whole-step speedup. Full-world density and gamma hashes match, and no WebGPU validation errors occurred. Final canonical gate status is recorded below. No timing ceilings or behavior baselines were changed.

Full-world density and gamma hashes matched across all four captures. Velocity hashes vary even between unchanged controls, so no bitwise velocity-equivalence claim is made. Timestamp granularity is about 0.065536 ms.

The pressure-row, gravity/cache, surface-pressure affine, face-support, face-remap, candidate-face-world, topology-face-transfer, physical-subface and packed-row tests pass. The old sharpening affine fixture had stale incidence stubs; after updating those, it exposes the same 0.03333334 versus 0 error in both the unchanged control and this implementation. Its tolerance is unchanged; no simulation workaround was added. The physical-subface fixture also needed its existing recovery-strength parameter supplied. Repository TypeScript checking reports 52 errors outside the changed files.

## Final receipts

- `artifacts/cm12-figure-7-radius-01/pressure-row-load-final.json`
- `artifacts/cm12-figure-7-radius-01/pressure-row-range-control-repeat.json`
- `artifacts/cm12-figure-7-radius-01/pressure-row-load-final.patch` (against the isolated compact control; both sides contain the QA catalogue-bounds fix)

The migration boundary is the row-data reuse described above, across every applicable simulation consumer. No broader topology representation change is included. Work pauses here after final regression validation and restoring the saved browser scene.

Final validation: **all 17 canonical Dawn lanes passed** in 277.5 seconds. Mini32: **28.7048 ms**; mini64: **82.9686 ms**. Thresholds are unchanged. Receipt: `artifacts/cm12-figure-7-radius-01/pressure-row-load-final-regression.json`.
