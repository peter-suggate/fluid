# NB-FLIP data-transformation experiments — 20 October 2026

This continues the [formal dataflow analysis](nbflip-formal-dataflow-2026-10-20.md). The experiments change the work or its data representation. Dispatch sizes and occupancy are not the design variables. Unit tests were deliberately not run, as requested; verification uses GPU replay, whole-frame benchmarks and type checks.

## Product result

Four changes are integrated: skip pressure return values that projection overwrites; skip band-owned root divergence with no solver reader; publish the surface through canonical metric owners using native capacity masks; and avoid a velocity norm that coupled particle prediction does not use. None allocates another GPU buffer.

The final controlled combined run measures **49.941 → 49.562 ms GPU pass sum**, saving **0.379 ms/frame (0.76%)** over 61 synchronized post-impact pour frames. Wall time averages 54.877 → 54.600 ms; paired median saving is **0.285 ms**. This is a modest measured frame improvement, not a large algorithmic speedup. The larger individual stage percentages below must not be added together. [Final combined benchmark](../../artifacts/nbflip-stage-analysis-2026-10-20/data-transform-product-complete-state.json).

In that same run:

| Affected stage | Original ms | Product ms |
|---|---:|---:|
| Fine/coarse pressure return | 0.252 | 0.178 |
| Root pressure RHS | 0.251 | 0.144 |
| Complete final surface metric/publication chain | 3.120 | 3.016 |

All final reported physical quantities match, including pressure residual. Particle, fine-tile and band-tile counts match on every measured frame. Intermediate residual reports differ by at most 0.0000992, within the corrected unchanged-code control's observed 0.000134 variation. That control shows no apparent speedup (mean GPU saving −0.071 ms), versus +0.379 ms for the product. Timing-order effects remain, so the result supports a small gain, not a precise universal percentage. Exact restored-state stage replays remain the stronger evidence for each transformation.

## Measurement contract

The reference snapshot is `4acecd97`. Each replay captures one actual GPU state, runs the original complete chain as an oracle, then restores the same inputs for every variant and repeat. Variant order rotates and reverses. GPU timestamps cover the complete replacement chain, including added setup and cleanup. Snapshot copies, output comparisons and workload censuses are outside timing. Original outputs are restored before the simulation continues. GPU jobs run serially under the shared GPU lock; investigation and implementation run in parallel.

Native pour frames 80 and 120 are at 3.33 and 5.00 seconds. The additional 1/60-second runs at frames 200–212 are at 3.33–3.53 seconds. These are after impact and during filling: the authored source stops at 5.25 seconds. Separate runs can have different particle counts because allocation and simulation history are not deterministic; only within-run, restored-state comparisons establish a speedup.

## Adopted: eliminate pressure return values with no reader

The pressure return reconstructs fine tile velocities from the coarse root. Immediately afterward, band projection overwrites every velocity texel in each band tile using its retained forced velocity and solved pressure. The intervening copy and preserved-component reads explicitly exclude band texels.

Let B be current band tiles, R the return reconstruction, C the preservation copy, and P band projection. The rewrite is:

`P(C(R(u)), pressure) = P(C(R outside B(u)), pressure)`.

The new implementation reads the existing band tile map directly and skips those fine return jobs. Coarse return, frame-failure checks, frozen regions, pressure, boundary rules and projection remain unchanged. No new buffer or texture is allocated. Generic transfer callers retain their original behavior unless a subsequent full band projection is explicitly attached.

Complete-chain replay, milliseconds (medians; control is an unchanged replay):

| Scene/state | Baseline | Control | Skip overwritten return | Comparison |
|---|---:|---:|---:|---|
| Pour, native frame 80 | 0.613 | 0.612 | 0.545 | Every output word exact |
| Pour, native frame 120 | 0.735 | 0.729 | 0.648 | Every output word exact |
| Pour, 1/60 frame 200 | 0.617 | 0.617 | 0.558 | Every output word exact |
| Pour, 1/60 frame 206 | 0.623 | 0.626 | 0.553 | Every output word exact |
| Pour, 1/60 frame 212 | 0.646 | 0.646 | 0.578 | Every output word exact |
| Breaking dam, frame 20 | 0.204 | 0.207 | 0.190 | Every output word exact |
| Breaking dam, frame 40 | 0.234 | 0.234 | 0.219 | Every output word exact |

Each pour replay checks approximately 488 MB of physical buffers/textures, including boundary values, pressure arena state and presentation storage. The product-form change was then benchmarked on 61 synchronized whole frames, 180–240: the targeted return pass averaged **0.240 → 0.176 ms (26.6% less)**. Whole-frame results are inconclusive: mean wall time saved 0.088 ms, paired median saved −0.011 ms, trimmed paired mean saved −0.033 ms. Do not advertise a demonstrated FPS increase from this small change.

Local evidence: [native replay](../../artifacts/nbflip-stage-analysis-2026-10-20/pressure-return-chain-v3.json), [late replay](../../artifacts/nbflip-stage-analysis-2026-10-20/pressure-return-chain-late.json), [dam replay](../../artifacts/nbflip-stage-analysis-2026-10-20/pressure-return-chain-dam.json), [product whole-frame run](../../artifacts/nbflip-stage-analysis-2026-10-20/pressure-return-product.json).

## Particle prediction: preserve the algebra, then fix the access pattern

The current filed-particle deposit and refinement passes traverse the same tile runs. Seeds and spill records are a separate population; they must retain their tail processing. Newly appended seeds do not participate in this step's filed-run momentum deposit.

### Iteration 1: combine deposit and prediction

The candidate reuses the loaded Particle and newly computed support mode to perform prediction inside deposit. Its OR/max outputs are disjoint from fixed-point deposit sums. Every output word matched, but the complete pair slowed from **2.078 → 2.878 ms** at frame 80 and **1.953 → 2.520 ms** at frame 120 (control versus candidate).

The removed work was record loading and traversal. All per-particle crossing queries, box construction, travel math and reduction operations remained. The experiment establishes that saving those reads is insufficient to pay for the changed combined execution. It does not establish a particular register or occupancy explanation.

### Iteration 2: exact per-cell sufficient statistics

Within each already-sorted cell, reduce six travel maxima and five independent join-mask slices, then update the tile once. OR is kept as OR; a union is never replaced by its enclosing rectangle. Exceptional boxes keep their original path.

At native frame 80, 838,470 filed particles occupied 104,173 cells (8.05 per occupied cell; maximum 249). Travel atomic invocations fell from **2,524,245 to 353,559**; half-mask atomics from 838,470 to 104,173. Nevertheless, the pair slowed from **2.023 → 2.480 ms**. Frame 120 likewise slowed from **1.945 → 2.220 ms**. Outputs remained exact.

Assigning a lane to a cell gives it a variable-length serial run and changes contiguous particle access into multiple independent runs. The density census exposes the long-tail work distribution. The measured result rejects this implementation despite the algebraic reduction in atomic operations.

### Iteration 3: reduce contiguous chunks

Keep the original contiguous 64-record chunks. Tile-wide half/join masks can be OR-reduced across a subgroup without a cell key. Travel maxima additionally admit an exact segmented reduction by support category. Partial chunks use neutral values.

Both variants were exact and improved on cell serialization, but still lost to the original:

| State | Control | Per-cell | Subgroup OR | OR + segmented travel |
|---|---:|---:|---:|---:|
| Native frame 80 | 2.009 | 2.470 | 2.133 | 2.257 |
| Native frame 120 | 1.905 | 2.075 | 2.002 | 2.120 |

At frame 80, half atomics fell from 809,394 to 28,009, and segmented travel atomics from 2,437,860 to 265,637. The reduction arithmetic and coordination still cost more than the eliminated atomics. The next hypothesis targets repeated nearest-crossing lookup/decoding through a shared per-cell query view, retaining contiguous traversal and the baseline direct reductions.

Local evidence: [fusion](../../artifacts/nbflip-stage-analysis-2026-10-20/particle-chain-native.json), [cell reduction and workload census](../../artifacts/nbflip-stage-analysis-2026-10-20/particle-chain-cells-v3.json), [contiguous subgroup reductions and census](../../artifacts/nbflip-stage-analysis-2026-10-20/particle-chain-subgroups-native.json).

### Iteration 4: cache the crossing query representation

A 64-cell tile view decodes crossing coordinates once and keeps the original contiguous particle traversal and direct atomics. Outputs were exact, but the improvement was marginal: baseline/control/candidate were 2.112/2.118/2.085 ms at frame 80 and 1.832/1.835/1.827 ms at frame 120.

The diagnostic census explains why this needs another adaptation. Frame 80 actually demanded 139,987 crossing decodes, while the full tile cache performed 244,798; frame 120 demanded 88,520 versus 215,750 cached decodes. Reuse was outweighed by eagerly representing cells that had no qualifying query. The next candidate must restrict the query view to demand without introducing another expensive particle traversal. [Measurements and demand census](../../artifacts/nbflip-stage-analysis-2026-10-20/particle-chain-crossing-view.json).

### Iteration 5: demand emitted by the existing producer

Deposit already knows which particles will request a nearest crossing. It emits a 64-bit cell-demand mask; refinement decodes only those cells. No extra particle traversal is introduced, and all setup/publication is included in timing.

This achieved the intended operation reduction but lost overall. Frame 80 crossing decodes fell 117,847 → 27,296, while control/candidate time rose 2.079 → 2.323 ms. Frame 120 decodes fell 88,531 → 21,201 and time rose 1.822 → 2.121 ms. Every output word matched. Constructing and consuming this intermediate view costs more than the work it removes. [Demanded-view replay](../../artifacts/nbflip-stage-analysis-2026-10-20/particle-chain-demanded-view.json).

The next useful direction is to change the information produced, not add another cache: publish a directly usable nearest-crossing representation, or move monotone speed conversion after the existing maximum reduction. Both can eliminate repeated arithmetic without an additional demand representation.

### Iterations 6–7: move monotone arithmetic after the maximum

Mathematically, positive scaling and square root commute with maximum. A candidate reduced squared ballistic speed per support half, then took square root and scaled only the final maxima. It removed repeated arithmetic and was close to baseline timing, but failed exactness: only residency-travel words differed. A second candidate retained the original per-particle `length(v)` while deferring scaling; it also failed.

The diagnostic comparisons establish that every difference is **one floating-point ULP**, confined to the six directional travel channels. They do not establish the exact compiler lowering. Storing the intermediate before scaling changes the observed floating-point evaluation boundary even when the source uses the same `length` function. Neither candidate is eligible for an exact representation rewrite. [Squared-speed diagnostic](../../artifacts/nbflip-stage-analysis-2026-10-20/particle-chain-raw-travel-diagnostic.json), [original-length ablation](../../artifacts/nbflip-stage-analysis-2026-10-20/particle-chain-raw-length.json).

The adapted candidate keeps the original complete ballistic expression and reductions, while using an explicit coupled-particle branch to avoid its unnecessary norm calculation. This tests dead arithmetic removal without moving its rounding boundary or constructing another representation. It passes exact comparison: control/candidate 2.105/2.063 ms at native frame 80 and 1.817/1.793 ms at frame 120. [Conditional-travel replay](../../artifacts/nbflip-stage-analysis-2026-10-20/particle-chain-conditional-travel.json). The independent dam check also passed exactness: baseline/control/candidate were 0.114539/0.114246/0.113704 ms at frame 20 and 0.161747/0.160665/0.159872 ms at frame 40. Those sub-microsecond differences are effectively neutral; they establish no dam speedup. [Dam check](../../artifacts/nbflip-stage-analysis-2026-10-20/particle-chain-conditional-dam.json). The six-line helper and single sorted-refinement call change are integrated; seed/spill handling remains as measured.

## Adopted: surface ownership and native capacity masks

### Iteration 1: metric and final publication together

The metric and interface store have different vertex owners. The initial fusion accommodated both by computing additional boundary distances. It also performed publication-capacity setup on coarse-only bricks which the original Store skips. Interface textures matched, but the complete chain slowed from **2.800 → 5.164 ms** at native frame 80.

### Iteration 2: one computation owner

Compute distance only on the original metric-owned vertices and route publication metadata to the original writer brick. A boundary fallback handles vertices without a canonical arena owner; a small commit preserves revision effects. This improved the candidate to **4.672 ms**, still slower than the **2.878 ms** control. Interface textures remained exact.

The next variants preserve fine-writer support before doing publication setup, and replace a derived representation:

`tile open-cell masks → 1,000 scalar capacity values → eight-cell OR`

becomes

`tile open-cell masks → direct eight-bit query`.

The experiment stages 64 two-word masks (512 bytes) instead of constructing 1,000 scalar booleans (4,000 bytes), preserving the distinct fine-cell and coarse-tile threshold rules. It tests this independently in the original standalone Store and in the canonical-owner chain, so a win from representation cannot be confused with a win from fusion.

### Iteration 3: direct mask queries

| Native state | Baseline | Control | Fine-gated canonical | Masked canonical | Standalone masked Store |
|---|---:|---:|---:|---:|---:|
| Frame 80 | 2.892 | 2.949 | 4.668 | **2.747** | 2.863 |
| Frame 120 | 3.015 | 2.987 | 4.819 | **2.868** | 2.975 |

All physical fields and metrics were exact; other control state passed the observable comparison below. Fine-only support gating by itself did not recover the loss. Eliminating the scalar-capacity representation removed roughly 1.9 ms from the canonical candidate and produced a small complete-chain win at both states. Standalone mask publication was much closer to measurement variation. The larger post-impact states confirm the chain improvement:

| Pour at 1/60 second | Baseline | Control | Masked canonical |
|---|---:|---:|---:|
| Frame 200 | 2.708 | 2.706 | **2.561** |
| Frame 206 | 2.816 | 2.778 | **2.656** |
| Frame 212 | 2.920 | 2.894 | **2.754** |

These nine-repeat runs passed the same metric/field and revision-contract comparison. [Late pour replay](../../artifacts/nbflip-stage-analysis-2026-10-20/surface-chain-masks-late.json). Independent dam replays passed the same correctness contract but exposed a small regression: 0.214 → 0.226 ms at frame 20 and 0.214 → 0.220 ms at frame 40. [Dam replay](../../artifacts/nbflip-stage-analysis-2026-10-20/surface-chain-masks-dam.json). The next adaptation folds scratch clearing into existing gather and revision commit into boundary publication. This removes two traversals rather than hiding the regression behind a scene-specific switch.

The initial integrated canonical implementation completed 61 synchronized pour frames (180–240), against a baseline that already contains the pressure-return deletion. The measured surface chain averaged 3.022 → 2.857 ms. Mean whole-frame GPU pass sum fell 48.127 → 47.923 ms; paired median wall saving was 0.169 ms and trimmed mean saving 0.158 ms. Final reported speed, volume, front and pressure residuals matched. Timing order remains material (baseline-first and candidate-first means disagree), so this supports the small stage win rather than a precise global FPS claim. [Integrated run](../../artifacts/nbflip-stage-analysis-2026-10-20/surface-canonical-integrated.json).

### Iteration 4: retain ownership while removing the extra traversals

Gather now initializes each brick's dead publication mark while gathering its existing window. Finish combines boundary publication with the original writer's revision update; its early exits still commit any routed changes. Boundary publication also queries the same capacity masks directly. The chain is now Gather → Far → canonical metric/publication → Finish, with no separate clear or commit traversal and no new allocation.

| Scene/state | Baseline | Control | Previous canonical | Compact canonical |
|---|---:|---:|---:|---:|
| Native pour 80 | 2.811 | 2.808 | 2.639 | **2.631** |
| Native pour 120 | 3.099 | 3.064 | 2.973 | **2.906** |
| Dam 20 | 0.214 | 0.212 | 0.223 | **0.191** |
| Dam 40 | 0.214 | 0.213 | 0.225 | **0.185** |

All four states passed exact physical/metric comparison and revision semantics. The adaptation turns the smaller-scene regression into a win while keeping the pour improvement. [Pour replay](../../artifacts/nbflip-stage-analysis-2026-10-20/surface-chain-compact-native.json), [dam replay](../../artifacts/nbflip-stage-analysis-2026-10-20/surface-chain-compact-dam.json).

The compact integrated run then completed the same 61 synchronized pour frames: surface chain **3.017 → 2.868 ms (4.9% less)**; whole-frame GPU sum 48.558 → 48.332 ms. Paired median wall saving was 0.250 ms and trimmed mean 0.209 ms. Final reported physical quantities matched exactly. Timing-order effects remain, so report the stage result more confidently than the global timing. [Compact integrated run](../../artifacts/nbflip-stage-analysis-2026-10-20/surface-compact-integrated.json).

### Equality of physical data versus opaque revision tokens

Baseline/control replays themselves differ in some live revision words. The current code updates these with `atomicMax(load(old)+1)` across competing writers; exact final token values depend on scheduling. The recorded examples lie in the live-revision interval, not interface textures. Subsequent comparisons still demand exact metric, interface and other state, but compare revision tokens by their observable contract: same changed/unchanged regions, validity/saturation class and monotonicity relative to the same initial state. Dead brick marks borrowed for publication routing are explicitly excluded only in the canonical candidate; the next table rebuild clears them before reuse.

Local evidence: [first fusion](../../artifacts/nbflip-stage-analysis-2026-10-20/surface-chain-native.json), [canonical ownership and mismatch examples](../../artifacts/nbflip-stage-analysis-2026-10-20/surface-chain-canonical-v2.json). These files retain their failed exact-word gate status; they are not relabelled as passing runs.

The subsequent [capacity-mask comparison](../../artifacts/nbflip-stage-analysis-2026-10-20/surface-chain-masks.json) records the revised, explicit equality contract and passes it.

## Distance demand: a useful bound on the next experiment

Let K(Q) denote the eight nodal corners around every query point. End-step distance demand is the union of phi publication, existing-particle start positions, seeding queries, shell-fitting probes and eligible P2G faces. Next-step particle movement samples metric only at its starting position; trajectory integration does not require a swept metric neighbourhood. Ordinary seeding stays in its cell. Only bootstrap, activated tiles and sources can enter shell fitting, whose probes admit a conservative radius-two cell dilation.

However, every reached, open cell with positive target heat already requires all eight corners: below eight particles, seeding queries depth to decide whether to replenish; at eight or more, existing particles query those corners next step. Phi publication and P2G enlarge the union. A demand representation therefore cannot delete most active-surface metric work merely by following particle occupancy. Useful exclusions are more likely cold/unheated regions and unused solid or coarse padding. A future census must count expensive distance-candidate evaluations removed, because excluded outer vertices often already take a cheap fallback.

This also rules out unsafe shortcuts: current trilinear sampling reads all eight words even at zero weight, so omitted scratch words can propagate NaNs. Retirement uses a separate, earlier metric epoch and cannot be removed using the final metric's demand proof.

## Adopted: remove unobserved band-owner pressure RHS

Band-owned root divergence is not read by the native root solver: its consumer excludes band owners. Wall RHS, warm pressure and minimum-state initialization remain live and are preserved. The candidate prepares the band map before RHS, then skips the redundant divergence and interior cut-face reductions for band owners.

The first full replay failed even in baseline/control because preparation allocates arena slots nondeterministically. Differences were confined to arena slot state, slot-indexed pool data and presented pressure; root pressure, velocities, boundaries and freeze/status channels matched. The adapted experiment executes original RHS and preparation once, snapshots that common allocation, saves original solve/project output as oracle, then replays the idempotent RHS and remaining chain. It keeps the strict buffer comparison and times the candidate map copy; common preparation is explicitly excluded.

| Native state | Baseline | Control | Skip dead band RHS |
|---|---:|---:|---:|
| Frame 80 | 6.089 | 5.994 | **5.850** |
| Frame 120 | 6.376 | 6.334 | **6.242** |

Every compared live word matched; only the proven-dead current-band root RHS words are excluded. [Original allocation-varying failure](../../artifacts/nbflip-stage-analysis-2026-10-20/pressure-rhs-chain-native.json), [fixed-allocation replay](../../artifacts/nbflip-stage-analysis-2026-10-20/pressure-rhs-chain-fixed-graph.json). The independent dam replay also passed: control/candidate 1.949/1.936 ms at frame 20 and 2.263/2.235 ms at frame 40. [Fixed-allocation dam replay](../../artifacts/nbflip-stage-analysis-2026-10-20/pressure-rhs-chain-fixed-dam.json). The first continuous run failed immediately after state synchronization. Investigation found that the history harness copied `Uniform compute blit tables`, an immutable instruction table whose offsets are cached in CPU objects private to each variant. Reordering copy commands changes those offsets; copying table bytes without the CPU cache changes the executed program. The candidate's band count fell to zero at the first synced frame. The corrected harness excludes exactly this instruction table and verifies remaining resource labels as well as shapes. The failed run is retained as [synchronization failure](../../artifacts/nbflip-stage-analysis-2026-10-20/pressure-rhs-integrated-failed.json). The corrected run completed all 61 synchronized frames with no errors and identical final reported physical quantities. RHS averaged **0.254 → 0.140 ms (45.0% less)**; total GPU pass sum 45.912 → 45.802 ms. Mean wall saving was only 0.027 ms, paired median 0.085 ms and trimmed mean 0.102 ms. The clean product implementation is now integrated, with no extra allocations and an explicit opt-in at the coupled frame; generic pressure callers retain their original behavior. [Corrected integrated run](../../artifacts/nbflip-stage-analysis-2026-10-20/pressure-rhs-integrated-fixed-sync.json).

## Next representation migration: nearest crossings as local offsets

The current nearest-crossing producer selects a nearby cell, converts it to a global storage key, and consumers repeatedly convert that key back to coordinates. The search radius bounds each coordinate difference to [-5, 5]. Its sufficient output is therefore the relative displacement δ from the queried cell, stored in the same one-word payload:

`encode(δ) = (δx + 5) | ((δy + 5) << 4) | ((δz + 5) << 8)`.

Keep `0xffffffff` as the absent sentinel. Every consumer knows its queried cell c and can use `c + decode(payload)` directly. Selection and tie-breaking remain before encoding. This removes address conversion at both producer and consumer, without another cache, allocation or demand pass.

This is a planned migration, not a measured gain. It must change both nearest producers, arena metric (including canonical publication), membership, refinement and legacy distance readers together. Reuse caches can retain the new payload under their existing topology/origin/crossing certificates, but format switching requires fresh or invalidated caches. Replay must compare decoded nearest coordinates and exact physical outputs; raw old/new nearest words intentionally differ. This is the next experiment that directly addresses a common producer/consumer representation.

## Reproducibility and status

The research checkout is `/Users/petersuggate/.codex/worktrees/nbflip-deposit-experiments/fluid`. Experiment source transformations and failed candidates remain there; only a validated, clean implementation belongs in product sources. JSON files record settings, source/shader hashes, repeats, comparison results and failures. Parser/layout failures are retained separately and are not numerical evidence.

Both pressure deletions, the compact canonical surface chain and conditional particle travel are integrated in the main checkout. Its final product-source replay also passes every physical output word, with 0.583/0.587 ms baseline/control versus 0.515 ms enabled: [product replay](../../artifacts/nbflip-stage-analysis-2026-10-20/pressure-return-product-replay.json). The final surface product-source replay passes, with baseline/control/candidate medians of 2.916/2.921/2.773 ms at frame 80 and 3.110/3.117/2.976 ms at frame 120: [surface product replay](../../artifacts/nbflip-stage-analysis-2026-10-20/surface-chain-product-replay.json). These product changes allocate no new GPU buffers. The conditional particle-travel helper is also integrated after exact pour and dam replays; its measured saving is small. The failed particle reductions and cache representations remain isolated in the research checkout. No unit tests were run during this experiment round.

The integrated pressure change can be replayed from the main checkout with:

```sh
lockf /private/tmp/claude-501/fluid-gpu.lock \
  node --import tsx tools/experiment-nbflip-pressure-chain.ts \
  --frames=80,120 --repeats=9 --out=artifacts/pressure-return-replay.json
```

The harness explicitly compiles a disabled oracle and an enabled candidate; the product default cannot silently turn the baseline into another enabled run.

The product surface replay is also available from the main checkout:

```sh
lockf /private/tmp/claude-501/fluid-gpu.lock \
  node --import tsx tools/experiment-nbflip-surface-chain.ts \
  --frames=80,120 --repeats=9 --out=artifacts/surface-replay.json
```

It disables canonical orchestration only on the replay solver, asserts the original four-stage capture and candidate entry names, saves baseline oracle outputs, and restores the baseline trajectory between selected frames. Product-source type checking passed after integration.

### Full-frame control-state correction

An unchanged-versus-unchanged run exposed a second benchmark defect: GPU-only synchronization retained independent CPU pressure-cycle histories. Identical source code consequently produced different pressure residuals on 15 of 61 measured frames, with maximum difference 12.062, and an apparent 0.165 ms GPU saving despite no code change. That control does not invalidate exact restored-state stage replays, but it makes the earlier full-frame numbers provisional. [Unchanged-code control](../../artifacts/nbflip-stage-analysis-2026-10-20/data-transform-unchanged-control.json).

The history harness now copies the pressure cycle window, its derived cycle bound, pressure launch history and ownership work hints alongside GPU state. The final control and product runs synchronize from frame 2, before independent histories accumulate. The immutable copy-instruction table remains local to each arm. This is the formal state's control history μ in practice: GPU arrays alone are not the complete state. The corrected unchanged-code control completes without errors; its maximum pressure-residual difference drops to 0.000134, and all final reported physical quantities match. Its mean apparent GPU saving is −0.071 ms and wall saving −0.048 ms, consistent with timing variation rather than a real improvement. [Corrected control](../../artifacts/nbflip-stage-analysis-2026-10-20/data-transform-control-complete-state.json).

### Final benchmark isolation

The first combined candidate snapshot also captured a concurrent, unrelated `transferOwners` hanging-face edit from the shared checkout. Its [run](../../artifacts/nbflip-stage-analysis-2026-10-20/data-transform-product-final.json) is retained but excluded from attribution to this work. The final candidate is rebuilt from `4acecd97` plus exactly the four reviewed patches in `/private/tmp/nbflip-data-transform-isolated`. The shared checkout retains the concurrent edit. The final history harness also preserves each arm's immutable compute-blit instruction records during state synchronization.

The final attribution run uses the ordinary product implementations, with no experiment switches. It supersedes the earlier GPU-only synchronization runs:

```sh
lockf /private/tmp/claude-501/fluid-gpu.lock \
  node --import tsx tools/benchmark-nbflip-history.ts data-transform-product-complete-state \
  --baseline=/private/tmp/nbflip-deposit-baseline-4acecd97 \
  --candidate=/private/tmp/nbflip-data-transform-isolated \
  --scene=nbflip-figure-7-pour --scene-file=/private/tmp/nbflip-pour-60.json \
  --transfer=flip --steps=240 --from=180 --sync=2 --profile=true \
  --baseline-values='{"timeStep":"scene"}' --candidate-values='{"timeStep":"scene"}' \
  --out=artifacts/nbflip-stage-analysis-2026-10-20
```
