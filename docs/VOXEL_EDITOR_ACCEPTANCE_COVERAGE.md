# Representative editor acceptance coverage

The editor acceptance suite complements `npm run test:dawn:sparse-cm12`; it does
not replace that simulation regression gate. Native editor tests should use the
production authoring and rendering contracts, and CPU tests cover document/UI
rules that do not require a GPU. Neither category alone proves browser pointer
routing, focus, or appearance.

## Production path

| Stage | Production owner | What an acceptance test must retain |
| --- | --- | --- |
| Tool selection and controls | `VoxelToolShelf`, registered plugin `ui`, `toolValues` | All eight solid plugins and three fluid plugins, normalized controls, scene defaults and unavailable reasons |
| Pointer preview and lifecycle | `useVoxelToolGesture`, `beginToolTransaction` | Begin/update/release/cancel ordering; latest preview; document ownership; exactly one completed history item |
| Solid acceptance | `WebGPUViewport` callback, worker client `validateLiveSolidEdit`, renderer `acceptLiveSolidEdit` | Await GPU acceptance before publishing the document; reject wet insertion without changing either authority |
| Fluid action | Transaction executor, worker `edit-fluid`, renderer `editFluid` | Execute the emitted descriptor once on release; no authored scene or history mutation |
| Worker boundary | Worker scene-cache adoption and renderer request handling | Structured clone and the production cache-transfer helper; identity assumptions must not survive a simulated boundary accidentally |
| GPU publication | Sparse CM12 adapter/resident and SVO renderer | Real device, retained topology, occupancy, fluid density and visible-surface publication; same world and advancing clock |
| Undo/Redo | Simulation controller registered live-acceptance callback | Real callback acceptance before stack/document change, including rejected Redo and cancelled pending work |
| Save/import/Continue | Simulation controller, scene library/autosave, `parseScene` | Authored scene roundtrip, compact pool compatibility, actual failure outcome and preservation of existing data |

## Run and latest result

```bash
npm run test:dawn:voxel-editor
```

The sequential, lease-aware suite passed **6/6** native tests in **34.40 s** on
Metal. It includes production editor acceptance, all fluid shapes, empty fluid
surface publication, live terrain, and new-room insertion. The final strengthened
production-path test also passed independently in 2.35 s process time.
Logs: `/tmp/voxel-editor-representative-suite.log` and
`/tmp/voxel-editor-production-final-dawn.log`.

`tests/voxel-editor-production-dawn.test.ts` drives registered plugin gestures and
transactions through the shared production worker acceptance coordinator using
structured-cloned requests, a real renderer acceptance method, real solver/SVO,
and the controller's registered Undo/Redo callback. It checks every solid plugin
in empty-running and evolved moving-water scenes, exact GPU solid values and
untouched cells, history restoration, pending cancellation, rejected wet Redo,
fluid add/remove descriptor parity and baseline mass, and physics advancement
while acceptance is pending. Registry assertions fail if a new tool lacks coverage.

The harness is a headless acceptance host: it supplies the renderer's real GPU
consumers but does not initialize a canvas or full water render pipeline. It
executes the worker coordinator rather than an actual browser Worker message
loop. Dedicated mesh classification/scan tests cover stale surface removal;
worker lifecycle tests and targeted browser checks cover the remaining crossings.
The final browser Wall → Undo → Redo → Carve → Undo sequence passed with moving
water and a continuing clock through 21.2 s.

The broader required Sparse CM12 gate remains red at its unchanged 180-second
budget: 5 passed, 6 timed out, 6 unrun. These editor results do not supersede that
limitation or claim all simulation regressions passed.

## Matrix and required evidence

| Scenario | Native evidence | CPU companion evidence |
| --- | --- | --- |
| Build, Carve, Box, Cut, Sphere, Drill, Wall, Channel | Registered plugin gesture and transaction reach the real renderer; expected affected and untouched GPU voxels; Undo restores occupancy | All signed faces, mirror, retraction, continuous interpolation, negative coordinates, shell targeting and bounded-work rejection in `voxel-tool-plugins.test.ts` |
| Water ball/cube/torus, add and remove | Real plugin action reaches renderer; add creates density and a surface; removal clears density **and surface vertices** without time advance or replacement | Preview/descriptor parity, torus hole and dimensions, scene-aware defaults, explicit height preservation in `fluid-tool-plugins.test.ts` |
| Cancel and pending controls | Cancel an accepted solid transaction through its rollback callback; no leftover geometry/history | Pending Save/Undo/Redo and import guards; cancellation upgrades, failed rollback and lost document ownership in `voxel-tool-lifecycle.test.ts` and `fluid-tool-transaction.test.ts` |
| Wet solid rejection during motion | GPU admission rejects without mass, occupancy or document change; ordinary frames continue while receipt maps | Controller keeps both history stacks unchanged on rejected acceptance in `voxel-history-acceptance.test.ts` |
| Terrain and cold worker input | Real refined terrain accepts fill/carve/Undo; worker clone retains its compiled terrain and adopted catalogue | Cold catalogue/picker and scene-cache tests; exact terrain picking and immutable source tests |
| Save, reopen, import | GPU is unnecessary for the persistence contract; reinitialization from an imported document is a separate native scene-load check | Real scene-library/controller functions, large valid terrain, quota failure preserving previous bytes, pool rename/delete/Continue and malformed import guards |
| Runtime failure | Strict health and uncaptured GPU errors fail the native test; accepted receipts cannot conceal faults | Worker shutdown/crash rejects outstanding requests; UI pending state settles |

Each native sequence should check strict simulation health and uncaptured GPU
errors, and record the resident identity and submitted time. Do not assert a
running simulation's exact time equality; assert monotonic progress and no reset.
For a paused drop/removal sequence, unchanged time is the expected evidence.

## Limits of substitutes

- `FakeWorker` in `voxel-worker-lifecycle.test.ts` proves request lifetime and
  rejection routing. Its emitted success messages do not prove worker execution,
  GPU admission, or rendering.
- A manually resolved acceptance callback in controller history tests proves
  stack ordering. It does not prove wet-overlap rejection or GPU rollback.
- A native solver test that directly calls `setSolidWorld` or `editFluid` bypasses
  plugin controls, gestures, transaction ownership and the renderer. It remains
  useful backend evidence, but must not be labelled full editor coverage.
- Calling the renderer with a production worker-cache helper and a structured
  clone tests that boundary's data semantics. It does not execute an actual
  browser Worker or verify its message dispatch; retain the worker lifecycle
  tests and describe this distinction in runner results.
- Fluid mass alone is insufficient: the observed removal bug had zero density
  while stale mesh vertices remained. Check production mesh classification and
  draw counts or an actual rendered image as well.
- A fake GPU device or patched no-op publication method cannot establish GPU
  acceptance. A headless renderer with a placeholder canvas can test its real
  acceptance methods when backed by real solver/SVO resources, but must be
  labelled as such. It does not exercise device initialization, canvas output,
  water-pipeline invalidation, or worker message dispatch. Do not replace a
  readiness method with an unconditional success stub.
- A mode named “moving water” must actually submit simulation advances while
  acceptance is pending. Advancing only between edits proves editing an evolved
  fluid state, not uninterrupted simulation during receipt mapping.
- Any changed occupancy array is weaker than correct editing. Compare expected
  affected voxel values and untouched sentinels; a wrong location or inverted
  operation must fail. For add/remove pairs, confirm the same emitted descriptor
  geometry and return to the baseline within the existing numeric tolerance.
- CPU projection/highlight tests do not establish hit testing, outside-menu
  dismissal, Escape propagation, keyboard focus, or CSS contrast in a browser.
  The browser QA record remains the evidence for those interactions.

The existing CPU suites already cover the declarative tool constraints and
persistence lifecycle above. Avoid adding implementation-mirroring tests merely
to inflate the representative suite's count; prioritize missing crossings
between real owners.
