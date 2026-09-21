# Native pressure execution: implementation and measurements

Implemented the pressure-layout part of `resident-execution-design.md`.
Production pressure fields are now native rectangular textures for every root
page count. Root fluid fields and their all-resident allocation are unchanged;
this is not completion of velocity-driven domain allocation.

## Numerical implementation

- Multigrid uses its original native shader fragment, with native texture
  loads/stores and ceil(logical extent / 4) dispatches at each level.
- Pipelines, bind groups and the complete cycle plan are built at initialization.
  The existing residual-evidence policy chooses a prebuilt prefix during advance.
- Projection binds the solved native field directly. The former atlas-to-dense
  pressure publication allocation/pass is absent.
- Tolerance, sweeps, hierarchy dimensions, pressure classification and safety/
  recovery operators are unchanged.
- `pressureStorageForQA` retains the tiled atlas and a logical-dispatch atlas
  strictly as controlled oracles. Production uses native storage. Both direct
  and indirect cycle gating remain available to the safety tests.

## Measurements

Same Metal adapter as the preceding comparisons, 44 advances, four warmups,
1/30 s timestep, rendering excluded, hardware timestamps enabled, GPU runs
exclusive and sequential. Controls fix one Full-Cycle, zero V-Cycles, full-domain
work, 6/6 sweeps, tolerance 10 and direct dispatch in every current-branch arm.

| Pressure layout control | Cycles | Entire advance |
|---|---:|---:|
| Tiled atlas, direct | 25.494 ms | 77.274 ms |
| Logical dispatch, same atlas | 23.134 ms | 72.678 ms |
| Native workspace, direct | 5.931 ms | 56.223 ms |
| Main, preceding matched control | 6.029 ms | 39.062 ms |

The three current-branch controls ended with the identical fine residual
2.7183847427368164. Separate every-step GPU comparisons establish bit equality
of fields; matching final residual alone is not used as proof of equivalence.
Removing padding explains a smaller part of the saving than replacing the atlas
representation. Cache misses were not measured; no separate cache-miss fraction
is claimed. Original production indirect atlas timing (29.753 ms) must not be
used as the isolated padding baseline because dispatch mode also changed.

With production defaults, native pressure measures 5.865 ms in cycles and
56.412 ms for the full advance, versus the preceding 29.753 / 82.275 ms capture.
Main's production cycles measured 8.061 ms (normally two encoded cycles).
The pressure-stage target is met locally; whole-frame parity is **not** met,
because the root fluid stages still use the generic page adapters.

Pressure workspace allocation in the long-dam equivalence fixture drops from
153,290,880 bytes to 71,220,480 bytes. This removes atlas padding and the dense
publication adapter; it does not represent a reduced logical resident domain.
The partial-page fixture drops from 7,809,492 to 1,900,308 bytes.

MiniDam32's production repeat is 16.627 ms per advance over 120 measured samples
(after four warmups), retaining its preceding ~16.4 ms performance and remaining
below the previously measured main median of 21.218 ms.

## Verification

- Native/tiled/logical-atlas layouts agree bit-for-bit in pressure, V, velocity
  and vertex phi after **every** step of both 12-step fixtures: partial pages
  and the actual long dam. Both insert liquid on frame 7. Identical preceding
  fields supply identical inputs for subsequent solves; this does not compare
  independently diverged main/branch trajectories.
- Minimal pressure budget and evidence-driven escalation still pass.
- Expanded corrupt-cycle tests pass all 13 reported checks across reference,
  native-page indirect and former paged QA paths (finite, NaN, infinity and
  rejection after an accepted cycle).
- Five focused CPU checks pass, including the logical-dispatch shader oracle.
- Type checking reports the same 15 existing errors outside the changed files.
- Canonical Sparse CM12 gate **fails: 4/17 lanes pass in 400.2 s**.
  See `native-pressure-sparse-regression.json`. Previously failing sparse
  expansion, topology, far-wall/collapse, performance and timeout areas recur.
  No thresholds were edited. This separate adaptive-volume gate is not clean.

Raw timings/source hashes/adapter information/cycle evidence and physical/logical
work censuses are in `native-pressure-measurements.json`. The MiniDam32 repeat is
included there as a separate capture. For the attribution controls run the
existing benchmark with `ONE_CYCLE=1 FULL_DOMAIN=1 ASYNC_DEMAND=1 FRAMES=44` and
`PRESSURE_STORAGE=paged`, `PRESSURE_STORAGE=paged-logical`, or omit
`PRESSURE_STORAGE` for native production storage.
