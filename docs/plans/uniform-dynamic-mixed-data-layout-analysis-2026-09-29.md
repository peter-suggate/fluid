# Uniform Geometric dynamic mixed mode: data and shader performance audit

29 September 2026. Analysis of the working tree at `c82429537f492f39476699b1ddec45fd19c1b086`, including its uncommitted changes. This is an analysis, not an implemented optimization or a measured speedup. File hashes, allocation calculations and an extracted historical hardware capture are preserved in [the evidence file](../benchmarks/uniform-dynamic-mixed-analysis-2026-09-29.json).

The existing 4h tile design already provides spatial locality, contiguous fine-owner blocks and cooperative workgroup staging. This audit does not establish that replacing it would improve performance. The strongest investigation is where expensive consumers fail to exploit that locality, or have long arithmetic/dependency chains despite good locality. Candidate changes stay within that design: interpolation-ready views, tile-local coefficient layouts, exceptional seam connectivity, and alternative surface-trial evaluation. Pressure-cycle tuning and dispatch consolidation are outside this proposal. None of the estimates below assume fewer transport rounds, sharpening sweeps, characteristic steps, or weaker pressure acceptance.

## What is established, and what still needs measurement

Source inspection establishes several long dependency chains and strided buffer accesses. A stride is not by itself evidence of poor effective locality: each lane's record is contiguous, adjacent lanes stay in a compact tile block, and subsequent instructions can reuse cache lines. Source inspection does **not** establish register spilling, memory transaction sizes, cache hit rates, or a 50% achievable speedup. Low occupancy is a diagnostic signal, not proof that a data structure is bad: registers, divergence, too little eligible work, shared-memory limits and dependency latency can all contribute. Raising occupancy without lowering frame time is not success. Apple's [occupancy guidance](https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy) and [Metal compute discussion](https://developer.apple.com/videos/play/tech-talks/10580/) explain these distinctions and register-allocation thresholds.

An existing Figure 9 capture gives useful prioritization evidence:

- Scene `cm12-figure-9`, 128×128×64, dynamic mixed, Apple M1 Max; captured at 2026-09-29 01:31 UTC against older revision `d767fe7d` with a dirty profiling harness.
- Untraced harness result: 25.884 ms/advance over 200 advances. Traced harness result: 28.984 ms/advance. The separately analyzed 73-advance trace window averaged 36.847 ms, with 32.997 ms attributed to compute intervals. These are different windows; do not subtract them to derive CPU cost.
- The capture records Chrome GPU contention and only 70.2% exclusive GPU coverage. It has no shader register/spill records. Consequently it is **not a clean current baseline**, and its counters are pass aggregates, not measurements of individual kernels.
- No new GPU run was started for this audit: other Dawn tests/profiles held the repository lease during inspection. The evidence file identifies the original artifact and retains its caveats.

| Historical pass aggregate | Mean ms/advance | Sampled occupancy | Read GB/s reported by capture |
| --- | ---: | ---: | ---: |
| Global surface volume | 3.392 | 22.6% | 5.57 |
| Transport | 3.280 | 31.3% | 20.07 |
| Surface redistance | 3.234 | 11.3% | 1.51 |
| Extension | 3.224 | 16.5% | 8.73 |
| Geometric sharpening | 2.996 | 36.6% | 27.94 |
| Surface advection | 2.603 | 16.1% | 4.34 |
| Momentum | 2.421 | 14.5% | 3.70 |
| Pressure V-cycles | 1.138 | 5.4% | 5.92 |
| Pressure band solve | 3.300 | 13.3% | 24.73 |

The band solve is separate from the small all-4h multigrid solve; it must remain visible in accounting even though neither is the optimization target here. The sampling passes' low reported external bandwidth does not support calling them DRAM-bandwidth saturated. Cache/texture issue limits and dependent sampling are more credible hypotheses to investigate, not proven diagnoses. See Apple's [bandwidth measurement guidance](https://developer.apple.com/documentation/xcode/measuring-the-gpus-use-of-memory-bandwidth).

## The current data path

The maintained entry is `uniform-volume`. Dynamic coarsening is the default in `uniform-geometric-parameters.ts:67`. Figure 9 is authored in `cm12-paper-scenes.ts:541`: 1,048,576 finest cells, 16,384 tiles of 4³ cells, and 1,081,665 finest vertices. Its dropped ball is initial liquid geometry; it is not automatically a moving rigid-body workload.

`UniformMixedFrame.advance` (`uniform-mixed-frame.ts:359`) runs:

1. Adopt the previous GPU-built layout and remap changed cells/faces.
2. Build support/geometry and liquid phase, extend velocity, prepare regular/coarse/hanging sampling data.
3. Advect phi, trace volume departure boxes, redistance, resolve hanging phi.
4. Build receiver coefficients; perform three conservative normalization rounds and gather volume.
5. Cleanup, global surface-volume correction and geometry reconstruction.
6. Eight sharpening sweeps, momentum advection and forces.
7. Transfer to all-4h pressure, solve/project, transfer back, solve the h surface band.
8. Extend far states, classify the next dynamic band, and build its layout on the GPU.

This is not a conventional pressure-dominated Eulerian workload: surface reconstruction, geometric volume consistency and characteristic sampling occupy much of it.

### Topology is compact; field storage is mostly finest-grid addressed

The phrase “4h tile” needs two distinct meanings here. The physical 4×4×4 h tile is the locality unit even when it holds 64 fine owners; a coarse 4h owner occupies that whole tile. This is not a flat owner graph with arbitrary fine-cell ordering.

The locality machinery already present is substantial:

- Fine-owner workgroups visit contiguous 4³ blocks. Coarse owners are packed across lanes instead of allocating 64 lanes to one scalar cell.
- The frame plan certifies an entire characteristic's reach, including interpolation/search margins, for the regular fine path (`uniform-mixed-frame-plan.ts:48`). That path avoids generic mixed sampling, not just mixed dispatch sizing.
- Fine extension slots are `[tile][component][64 local cells]`, with separate coarse planes.
- Far extension publication cooperatively stages a 3³ tile neighborhood and source geometry (`uniform-mixed-extension.ts:377`).
- Surface-volume seeding stages 125 vertex values for a fine tile rather than loading 8 corners independently for each of its 64 cells (`uniform-mixed-surface-volume.ts:141`).
- Surface advection/redistance gives canonical candidate vertices their own lanes, and momentum separates components (`uniform-mixed-surface.ts:565`, `uniform-mixed-momentum.ts:131`).
- Coarse sampling and hanging fine taps are already materialized; resolved phi already removes reconstruction from many consumers.

These are advantages to preserve. The questions are whether subsequent inner-loop accesses remain efficient and whether instruction dependency/ALU cost, rather than locality, explains the remaining runtime. Timings must separate certified fine, general fine, coarse regular and coarse seam work before attributing a large stage's cost to seam traversal.

`uniform-mixed-layout.ts:186` assigns all h owners first, 64 contiguous owners per fine tile, then one owner per coarse tile. `uniform-mixed-topology.wgsl.ts:50` reads a packed tile descriptor containing width and owner base. Stencil masks freeze the 3×3×3 neighborhood's widths. The GPU builder reproduces this order using scans.

However, `uniform-mixed-faces.wgsl.ts:17` repeatedly converts position → tile → width/base → local owner, and face queries add neighbor ownership and patch geometry. Equal-width fine interiors can be arithmetic; general seam queries carry dependent metadata loads and branches into their consumers.

Persistent V, phi and MAC fields remain native 3D textures. A 4h cell stores V at its origin, and its three unsplit positive faces have different fine-coordinate anchors. They are not three adjacent scalar words in a compact coarse owner record. This is a sparse population of a dense address space. Texture swizzling means its exact memory transaction cost cannot be inferred from linear coordinates alone.

The code already mitigates this with a compact `(D/4+2)` coarse sampling cache, resolved phi, and a finest-sized `unitVelocity` interpolation cache. Recommendations must build on those rather than propose them as missing.

| Logical storage at Figure 9 resolution | Bytes | MiB |
| --- | ---: | ---: |
| Receiver coefficient capacity | 41,943,040 | 40.00 |
| Six donor limbs plus decoded sum | 29,360,128 | 28.00 |
| Mixed extension value/distance ping-pong range | 51,675,136 | 49.28 |
| Surface-volume scratch range | 14,243,108 | 13.58 |
| Dynamic velocity-bound cube table | 8,847,360 | 8.44 |
| Each finest-cell RGBA32 field | 16,777,216 | 16.00 |
| Each finest-cell R32 field | 4,194,304 | 4.00 |
| Two finest-vertex R32 phi fields | 8,653,320 | 8.25 |

These are source-derived logical sizes, not measured traffic or an additive allocation total. Extension, transport, sharpening and other stages reuse the scratch arena. Coarsening cuts the number of live owners without proportionally reducing these capacities. Merely allocating fewer bytes would not guarantee faster execution; the relevant changes must improve addresses actually touched and the instructions that compute them.

## Priority 1: interpolation and redistance representation

Relevant sources: `uniform-mixed-surface.ts:111,221,411,538`; `uniform-mixed-velocity-sampling.wgsl.ts:69,125,171`; `uniform-velocity-departure.wgsl.ts:5`; `uniform-mixed-momentum.ts:131`.

Surface RK2 samples a complete velocity twice. Each trilinear component uses eight scalar taps, so an ordinary unmixed RK2 trace requests 48 texture taps before phi sampling. Blended h/4h interpolation can evaluate both interpolants. Cubic phi uses a 4×4×4 footprint. These are source-level tap counts, not DRAM read counts: caches can satisfy repeated loads.

Momentum integrates up to 32 characteristic substeps. Every substep needs two full velocity samples, and its output determines the next sample position. The final advected component needs another interpolation. Neighboring lanes can take different numbers of steps. Surface redistance has up to eight dependent iterations; each requests six trilinear samples for a gradient, then a trial sample. Its ordinary trilinear path therefore requests up to 56 taps per iteration, or 448 at eight iterations, before ancillary tests. Near mixed-width coarse geometry the trial can instead be cubic. Retirement/contact paths add variable-length searches.

The candidate limit is latency along each characteristic and the amount of live state/control flow needed to hide it. That dependency remains even when the 4h tile gives excellent cache reuse; it is not evidence that the samples are random. Coherent translations can keep all lanes' departures coherent far beyond their originating tile. Deformation and different iteration counts can reduce that coherence, but their distribution must be measured. More dispatches cannot shorten the characteristic dependency graph. Momentum already assigns components to separate lanes; the old three-characteristics-per-lane problem is fixed.

Recommended representation experiments:

1. **A truly sampling-ready MAC view.** Compare the current RGBA32 view with component planes and explicit boundary halos. A component request currently selects one component of an RGBA texture; separate scalar planes may reduce unused component traffic and simplify boundary addressing. It is not automatically a 4× bandwidth gain, because cache behavior and packing may already help. Keep canonical conservation/pressure fields unchanged initially and measure the total cost of producing and consuming the view.
2. **Use the texture sampler for interpolation where supported.** The current bind layouts request `unfilterable-float` and all interpolation is manual. A `float32-filterable` experiment could replace eight software taps with a filtered lookup per component, with the correct staggered coordinates and negative-face halo. It still performs interpolation work in hardware; it does not mean eight physical reads become one. Filtered interpolation may round differently and must pass the numerical lanes. Feature negotiation is required by [WebGPU's format capabilities](https://www.w3.org/TR/webgpu/#float32-filterable). A half-precision cache is a separate, higher-risk numerical experiment, not a prerequisite.
3. **Represent local phi interpolation as reusable coefficients or a bounded tile patch.** Keep the same six finite-difference sample positions but load their union of corner data together, then evaluate locally. For certified footprints, neighbor lanes can cooperatively stage a patch. Do not assume a 5³ tile closure covers a characteristic: redistance can move ±4 owner widths, and advection can travel farther. A certificate or explicit cross-tile path is necessary.
4. **Make seam addressing explicit.** Keep arithmetic indexing within regular tiles; store compact neighbor-base/width descriptors and seam patch records for boundaries. This moves topology traversal out of every sample/face evaluation. Avoid a six-neighbor record for every fine cell: most neighbors are already `index ± 1/4/16`, and such a table could create more traffic than it removes.

Replacing finite-difference gradients with analytic trilinear gradients is a numerical-method change at interpolation-cell boundaries. It is not an equivalent optimization to assume silently. Similarly, dropping cubic interpolation or shortening characteristic loops is outside the proposed performance work.

## Priority 2: expose the surface-volume trials as parallel data

Relevant sources: `uniform-mixed-surface-volume.ts:290-351`; `uniform-surface-volume.wgsl.ts:3-20`.

One invocation loads eight phi values and eight shift scales, then serially evaluates **17 trial shifts**. `fill` sums six tetrahedral volumes, and each tetrahedron classifies four signs into dynamically indexed local arrays. Two refinement rounds can evaluate 204 tetrahedra per cut owner. Fully inside/outside owners already take a constant-result shortcut; band culling and shared seed corners are also already implemented.

The workgroup then holds 20 floats per lane in `sums`, 5,120 bytes for 64 lanes, and runs a reduction with barriers. Keeping 17 trials in one lane has an important advantage: its 16 corner/scale values are reused without repeatedly loading or exchanging them. Serial trial evaluation is therefore not intrinsically a mistake. The candidate pressure comes from local arrays, sign branches, division chains, and a long-lived per-owner trial workload. Register spilling remains unproven.

Compare the current intermediate, “one owner carries all 17 trials,” with **a tile-local batch of corner records plus trial-major results**. Expose the independent owner×trial dimension to lanes; cooperatively load corners, evaluate a few trials per lane, and reduce a fixed trial across owners. Test small tiles of owners/trials rather than naively launching 17 mostly empty subgroups. Retain the current scheme if corner reuse wins. A sign-mask tetrahedron evaluator with fixed scalar selections is another useful experiment against dynamic `n[count]`/`o[outside]` arrays; it can shorten the evaluator while retaining all current tile/lane ownership.

This retains all 17 samples, both refinement rounds and the same fill curve. It is a representation and work-granularity change, not removal of calculations. Preserve the existing reduction order where feasible; otherwise measure rounding changes in surface volume and long-run mass behavior. Extra corner-record traffic and shared memory can outweigh the benefit, so count producer and reduction costs in the result.

## Priority 3: transport coefficients and donor ownership

Relevant sources: `uniform-mixed-transport.wgsl.ts:33,59,70,114,143,171,337`; `uniform-volume-donor-sum.wgsl.ts:36`; `uniform-mixed-transport.ts:147`.

### Coefficients run across lanes with a 40-byte stride

For fine owners the row address is `tile*640 + lane*10` words: one packed base and nine weights. At fixed coefficient k, consecutive owner lanes access words ten apart. A block-transposed layout, `tileBase + coefficient*64 + lane`, makes that instruction contiguous while retaining all values and the packed base.

For illustration only, 32 aligned lanes accessing one current coefficient span ten 128-byte lines; a transposed coefficient spans one. This is **not** a prediction of 10× total traffic reduction: subsequent coefficient instructions reuse those same lines, and compiler scheduling/caches matter. The falsifiable claim is improved per-instruction locality and possibly lower cache/issue pressure.

Give coarse rows their own format. They require nine entries on an all-coarse donor footprint or up to 126 entries when sampling h donors. The current 2,560-byte tile reservation accommodates both 64 fine rows and a coarse row, but the coarse row uses only 508 bytes at maximum. An AoSoA fine layout plus separate coarse row descriptors is more natural than forcing both through one maximum-capacity tile block.

### Exact donor accumulation is scattered synchronization

Each nonzero weight adds to a donor using a six-limb representation of the exact nonnegative f32 sum. The common path performs one or two integer atomics, with bounded carry propagation on overflow; it is not six atomics for every contribution. Decode loads six limbs and rounds once. The three normalization rounds repeatedly revisit the same graph, and receivers can contend on shared donors. Coarse donors may be represented repeatedly in a fine-grain overlap row.

Mixed donor indices are already packed by owner, so proposing the native solver's brick-order optimization again would miss the point. The remaining issue is scatter contention, indirect addressing and the normalization graph's orientation.

The larger experiment is a **dual-view transport graph**: receiver-major edges for row normalization/gather, donor-major edge references for donor reductions. Build the transpose once from that frame's departure boxes, then reuse it in the three rounds. Give each donor a contiguous incoming segment and reduce its limbs locally before publication. Preserve every contribution, the self-fallback edge, zero-weight handling, capacities and final correctly rounded sum. The inverse graph must include a possible self edge before round-zero fallback activates it.

This adds O(E) references and a count/scan/scatter construction cost every frame. It is worthwhile only if the reused reductions save more than construction costs. It should follow the cheap layout experiment, not precede it. A simple unordered float reduction would change the current arithmetic contract and is not the first proposed implementation.

Coarse normalization already distributes edge work over lanes, but lane zero still sums up to 126 weights serially. `buildAt` and `gatherAt` remain per-owner loops. A coarse edge block can expose those independent overlaps too; changing summation order needs its own validation. The current atomic-appended live lists also do not guarantee tile order. Stable block compaction may help coarse neighbor locality; treat that as a data-order experiment, not a dispatch-count optimization.

## Priority 4: extension connectivity and live state

Relevant sources: `uniform-mixed-extension.ts:31,117,146,195,250,282,471`.

Extension already has a useful GPU-oriented layout: tile/component-major fine slots, a separate component-major coarse layer, and negative-wall planes. Fine neighbors within a tile are nearby. It should not be described as an entirely unstructured field.

The remaining general stencil work is expensive. One fine-owner lane loops through all three components. Each non-source component evaluates six neighbors, with seam cases passing through `umVelocitySite` and `umFace`, then carries six value/distance/spacing records through the Eikonal update. Coarse seams already distribute patches across lanes. Publish also has cooperative staging for far states.

A compact **canonical-patch record** should contain a value/distance index plus exceptional seam-neighbor indices and spacings. Geometry stays immutable through the sweeps, so the two sweep rounds can consume direct state references. Keep regular fine indexing implicit and generate records only at exceptional boundaries; compare the descriptor build cost against saved dependent queries.

The existing component-major slots also permit one component per lane for fine sweeps without first repacking data. This is a low-risk way to test whether the serial component evaluator is the occupancy limit, paired with compiler register/limiter evidence. Preserve equal-distance tie averaging, the distance cap, boundary seeds and the coarse nearest-source continuation. Packing all state into one compact owner array without a stable seam/face map would make the neighbor problem worse.

## Priority 5: sharpening's repeated stencil traffic

Relevant sources: `uniform-mixed-sharpening.ts:113-115,230-255,275-335,360`.

Sharpening runs eight rounds of propose/limit/commit. Per-owner budgets are six interleaved floats: give, take, distance, desired and two limit factors. Raw flux is `3*linearFineAnchor + axis`; geometry flags use another finest-grid anchor array. Thus packed owner indexing, fine-coordinate face indexing and tile-list indexing all coexist inside each stencil.

At a fixed budget field, neighboring packed owners read with a 24-byte stride. A tile-block SoA layout makes each requested field contiguous. Store flux component planes in tile order with explicit seam patches, alongside the seam neighbor records used by extension. Existing geometry admission caching should remain; the improvement is to make its consumers read compact addresses.

Coarse seam limit currently stages face terms in parallel and then has one lane sum up to 96 terms; commit retains serial per-side summations. A segmented tree/subgroup reduction can expose that reduction, with arithmetic-order validation. Keep global round boundaries: executing all eight sweeps locally in a tile without exchanging updated neighboring budgets would change the algorithm. Removing those boundaries is not an acceptable shortcut.

## Dynamic metadata and CPU waits

The current implementation is materially newer than the September 28 CPU-wait proposal:

- `webgpu-uniform-reference.ts:1504` encodes the census and GPU builder in the frame tail. Readbacks contain only counters/receipts and are lagged.
- `uniform-mixed-dynamic.ts:70` skips diagnostics when its three-slot ring is full.
- `webgpu-uniform-reference.ts:2414` and `uniform-mixed-remap.ts:88` adopt/remap the GPU generation without waiting for a CPU layout array. Default census cadence is one frame (`webgpu-uniform-reference.ts:418`).
- `uniform-mixed-frame.ts:82,440,490` still requires a checked pressure receipt from frame N−2 before encoding N. There are two receipt slots; the receipt provides failure checking and the lagged pressure plan. This is genuine bounded CPU feedback even though `advanceTo` itself is synchronous.
- `webgpu-uniform-reference.ts:2314` refuses another advance when both receipts are outstanding or an edit is draining frames. `awaitFrameCompletion` waits for receipts and queue completion, but it is not called unconditionally inside every mixed advance.
- `webgpu-renderer.ts:3339` limits pipelined presentation while one is in flight. Queue completion retires presentation asynchronously (`:3950`); it does not synchronously block the JavaScript thread. It can still create a GPU bubble if the producer is gated until a callback/rAF arrives. Timeline evidence is needed to quantify this.
- `tools/profile-uniform-geometric-dawn.ts:85` deliberately awaits frame completion, queue completion and stats per frame. It is useful for stage diagnosis, but cannot demonstrate receipt-independent sustained throughput.

Do not implement the old host-free-layout plan again: that work is present. Do not remove a map/fence without replacing its acceptance, resource-lifetime or presentation role. First measure submit→GPU-start, GPU-end→map callback, next eligible advance→submit, and time rejected by each gate. Attribute idle periods to a dependency, rather than counting promises. Only if receipt gating leaves material idle time should the pressure-plan contract move entirely to GPU state; increasing the ring alone still retains host feedback and changes failure latency.

There are also true serial metadata algorithms: census prefix kernels scan a line per invocation (`uniform-mixed-dynamic.ts:479`), and the layout builder scans block totals in a single workgroup (`uniform-mixed-layout-builder.ts:152`). At Figure 9 these lines are only 32/32/16 tiles and there are 64 builder blocks. They are secondary targets. For larger dimensions, use parallel blocked scans with contiguous intermediate layouts, while keeping the band and all safety margins identical. A GPU-resident algorithm can still have poor GPU parallelism.

## Can these changes halve frame time?

The seven principal non-pressure passes above total 21.150 of 32.997 historical GPU ms, or 64.1%. Amdahl's relation is `new/old = 1 − f + f/s`.

| Speedup across those seven passes | GPU time reduction with everything else fixed |
| --- | ---: |
| 2× | 32.0% |
| 3× | 42.7% |
| 4× | 48.1% |
| 4.55× | 50.0% |

These are arithmetic scenarios, not forecasts. The contaminated older trace cannot set an acceptance budget. Nevertheless it shows why one or two 20% kernel improvements will not achieve the request. A 50% simulation-frame reduction requires a broad representation change or very large wins across several of these families. End-to-end rendered frame time has a further unchanged extraction/rendering component; record that separately before promising 50% on the displayed frame.

Recommended implementation sequence:

1. Establish a clean, pinned Figure 9 dynamic baseline spanning startup, impact/spread and later motion, preserving full numerical settings and ownership census statistics. Use sustained advances without diagnostic fences for throughput; a separate bounded capture for counters and individual kernel attribution. Serialize GPU runs under the repository lease and exclude browser simulation. Attribute time and eligible lanes to certified fine/general fine/coarse regular/coarse seam paths; collect characteristic-iteration and cut-trial histograms, plus bytes/cache misses and compiler resource usage where available. Keep optional telemetry in GPU reductions with lagged readbacks. Without this split, do not assume general seam code dominates a stage whose common path already exploits tiles.
2. Prototype tile-transposed transport coefficients and tile-SoA sharpening budgets independently. These isolate concrete address changes and have limited numerical risk.
3. Prototype parallel surface-volume trials and explicit extension seam connectivity. Retain all work and numerical bounds. Measure register pressure, active lanes, cache/texture or buffer limiters, and producer+consumer time.
4. Investigate the sampling-ready field view across advection, redistance and momentum together. This is the broadest shared opportunity. Evaluate layout-only/manual interpolation first, then filtered sampling as a separately validated option.
5. Consider the donor-transposed graph and persistent coarse/fine field separation only when the preceding measurements show the required headroom. Stable tile page IDs can keep dynamic remaps local; keep topology generation, owner indexing, renderer bindings and halo validity explicit. Do not turn every fine-cell access into a page-table traversal.

For each candidate, use the same initial state and simulated horizon, compare whole-frame time and p90 as well as the targeted kernels, and report fine/coarse/seam counts. A changed trajectory can change the work population; a shorter kernel time with fewer live owners is not proof of a better access pattern. Include new cache construction, graph construction, remap and presentation costs in the result.

Correctness must retain transport mass/capacity/fallback behavior, mixed seams, solid boundaries, extension symmetry, long-dam progression, surface volume, pressure rejection and live relayout behavior. Run the applicable focused lanes, then the unchanged repository gate: `npm run check:types`, `npm run test:unit`, `npm run test:dawn`. No timing ceiling or numerical lane should be weakened.

This audit changed only its report and evidence artifact. It did not modify shaders or run the clean-repository gate, and it does not claim the existing dirty working tree passes that gate.
