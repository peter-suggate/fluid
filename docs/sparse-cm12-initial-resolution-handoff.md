# Sparse CM12 initial resolution — coarse-based vast spaces

## Current target and acceptance scene — updated 2026-09-05

**Ocean-seiche is the acceptance scene for a coarse spatial foundation with local,
reversible refinement.** Quiet water must be representable by cells spanning 16, 32,
64 and progressively larger numbers of finest cells per axis. Coarsening must not
stop at an 8³ or 16³ block of finest cells. Larger basins should primarily add coarse
coverage; they must not require a catalogue of finest-scale bricks across the basin.

**This is general adoption across all Sparse CM12 scenes.** Topology preparation,
conservative transfer, capacity handling and publication must share the ordinary
solver path. Ocean-seiche supplies acceptance evidence; no scene-name branch or
ocean-only simulation mode may substitute for the general capability. Clipped domain
boundaries, live scene edits, solid coupling and existing small scenes remain part
of the adoption contract.

Here `B8` means eight sample cells per brick axis, whereas a cell of width `32h`
covers a `32³` block of finest cells, with `h = finestCellSize_m`. The existing
formula `cellWidth / h = B × spanBricks / resolution` permits large physical cells
without raising B8/P8. A large macro brick alone is not evidence of coarse cells:
acceptance must report physical cell widths and the liquid volume they represent.

The required lifecycle is: initialize coarse macro coverage; refine locally as the
wave or another feature demands it; split macro leaves when their internal rungs
cannot provide local detail; merge compatible siblings when that detail is no
longer needed; reclaim the retired topology and presentation storage. Runtime
macro splitting and merging are **required scope**, even within the existing
integer lattice. Extending the address representation below the declared finest
cell remains outside this programme.

The first milestone is a testable ocean in the ordinary UI at the existing finest
cell size. Deeper finest lattices come later. A global minimum-cell-size region is
a useful comparison arm, but its hard refinement limit does not satisfy this target.

### Realtime requirement — supersedes the paused replacement milestone

The user rejected multi-second “Updating resolution” pauses. Whole-resident
replacement with simulation suspended during CPU packing, allocation or pipeline
compilation is **not an acceptable production solution**, even if its conservation
and publication tests pass. The implementation described below is a correctness
prototype until this requirement is met.

Preparation must run without stopping accepted physics or rendering. Expensive CPU
geometry/packing must run outside the advancing worker; GPU uploads and background
compilation must be bounded so they cannot monopolize the frame schedule. Publication
must consume the latest accepted fields, validate that its captured ownership still
matches, and switch all consumers at a bounded frame boundary. A stale candidate is
cancelled while accepted simulation continues. Simply removing the pause guard or
publishing fields captured seconds earlier is not correct.

Realtime acceptance must measure the actual worker/frame stalls during repeated
split/merge and require continued accepted-step progress while preparation runs.
Report maximum preparation slices and publication latency as well as average frame
cost. A UI spinner, an async function, or eventual completion is not realtime evidence.

### Authored region enforcement — 2026-09-06

Authored minimums constrain physical cell width (`B × span / resolution`),
including macro leaves, support coverage and topology closure. Intersecting leaves
inherit the minimum; maximum-size requests apply to contained leaves. Minimums win
when overlapping requests conflict. A live region edit schedules preparation on the
next advance; already satisfied maximums no longer starve unsatisfied requests in
the bounded admission queue or get undone by quiet merges.

The ordinary ocean GPU regression applies a global min8 after simulation starts,
then adds max8 to the same region. Through 129 steps and four publications, no
active cell inside the box falls below 8h, and the region converges to exactly 8h.
Cells outside the authored box remain unconstrained. Initial global min16 and
min32 also pass CPU mass/coverage checks and four-step GPU evolution checks;
ordinary initial liquid volume remains 1,853,440 finest-cell-volume units.

Initial coarse floors merge clipped coverage and grade its surrounding halo.
Live plans that cannot represent a requested floor with available complete coverage
and capacity defer publication; arbitrary partial live min32 edits are not yet an
acceptance claim. Enforcement evidence does not establish realtime latency: the
background compilation and maximum-slice problems below remain open.

The long ocean test also exposed duplicate dynamic directory allocations. New
directory entries now become visible in a separate publication dispatch, after
reservation payload writes. A GPU collision/retirement/reuse stress test checks
uniqueness, and the previously failing 129-step ocean run passes.

Validation receipt: 20 CPU planner/initialization tests pass (three Dawn-only
tests skipped in that CPU run); all three focused ocean GPU tests pass; the
production build passes. The final canonical Dawn run passes 15/16 lanes, with
all correctness lanes green. Mini32's native `ProcessEvents` crash was resolved
by retaining the Dawn GPU instance through the test's asynchronous work. Mini32
performance is 31.654 ms against 40 ms; mini64 is 55.116 ms against 50 ms and
remains a failing gate. A preceding run measured mini64 at 42.402 ms, so timing
variability remains material; no ceiling was changed. Repository typechecking
still reports existing errors outside the touched files.

### Production integration status — 2026-09-05

The ordinary solver now prepares bounded resident-generation replacements in the
background. This supersedes the historical “GPU adoption unimplemented” notes below.
Unbacked authored rungs and macro leaves use this path; fully backed small scenes
retain in-place rung changes and also check quiet sibling groups for macro merging.
No scene-name switch or ocean-only mode selects the implementation.

- Accepted fields remain on the GPU. Sparse clipped overlap maps transfer liquid
  mass, gamma, cell momentum, pressure and coplanar face flux into isolated storage.
  New internal faces derive velocity from the transferred cells.
- A shared owner publishes physics, pressure/transport connectivity, presentation,
  collision bindings and diagnostic sources together. CPU packing runs in a second
  worker; the simulation worker realizes resource commands with cooperative uploads
  and asynchronous pipeline compilation. Physics continues during preparation; only
  final validation/transfer/publication holds advances. Scene edits
  invalidate stale preparation; asynchronous readers and submitted work delay reclaim.
- Local rung demand, physical 2:1 closure and macro coverage closure can split leaves.
  A macro's own motion/thin-fluid demand can split beyond its finest internal rung.
  Complete quiet full-liquid sibling groups may merge into progressively larger cells.
  Clipped or open-world sibling merges remain conservative exclusions.
- Lifetime leaf, cell and GPU-buffer budgets are fixed from construction. Absorbed
  open-world pages continue to count against the original growth budget. A budget
  deferral retains the accepted generation and is visible in the transport status.
- Ordinary ocean construction now includes wet 32h cells. A 129-step normal-policy
  Dawn run (4.3 simulated seconds) published two generations with no activity or
  presentation faults. This is integration evidence; reference wave phase/amplitude
  comparison is still required for the numerical acceptance below.
- CPU gates cover local splitting, quiet 64h merging, physical grading, clipped and
  signed overlap geometry, and budget deferral. Device transfer gates check mass,
  gamma, momentum and exterior flux conservation.

The implementation replaces the whole bounded resident, rather than wiring the
standalone SCMT storage prototype directly into every resident buffer. That prototype
remains an independently tested storage component. Whole-resident construction is
still too expensive to satisfy the realtime requirement, despite background progress.
A separate quiet, non-ocean fixture publishes two generations from 512 span-one
leaves to 197 leaves spanning 1, 2 and 4 bricks. The coarsest rung now accumulates
quiet history so spatial merging is reachable under the ordinary policy.

Production-build browser observation (2026-09-05): ordinary ocean advanced to
38.7667 simulated seconds and published three replacements, with zero reported stale
candidates. The first preparation took 96.3 seconds; a later one took 54.2 seconds.
The largest measured realization CPU slice was 402.7 ms against a 2 ms scheduling
target. The paused pressure receipt was 4.09e-4 relative residual with 423,988 accepted
cells. These receipts prove live adoption, **not realtime acceptance**. An individual
synchronous resource operation can overrun the cooperative slice. Operation attribution
and maximum handover timing are now exposed for the next measurement. Temporary
shader handles are released at their last construction use instead of all being
retained until the complete replacement finishes.

The CPU construction profile records 189 pipeline requests per replacement. A
representative 64-leaf refinement with closure reuses only two exact pipeline
descriptors: generation-specific offsets/counts change the other shader sources.
Prioritize a stable shader interface and reusable storage for the realtime cutover;
an exact compilation cache alone does not remove the recurring cost. See
`docs/sparse-cm12-ocean-transport-cost-investigation-2026-09-05.md` for the measured
construction/reuse receipts. Current focused CPU checks pass (24 tests and the stage
timing contract); the latest GPU regression and instrumented browser run are pending.

Final realtime UI validation, the full post-integration regression receipt, reference wave
accuracy and vast-space scaling measurements are still being collected.

The work order is **WP4 → WP6a → first UI ocean milestone → WP6b → WP7**.
WP5 is conditional on measured pressure convergence. WP identifiers are retained
for continuity; their numeric order is not the dependency order.

The original investigation follows. Historical line references and measurements
describe the original snapshot unless explicitly updated; completed WP0–WP3 work
supersedes the original blockers in §§3–4. Sections 7 and 9 define current acceptance.

Original status: exploration, 2026-09-05. At the time no code had changed; every
claim below was read from HEAD or measured in that session, and inferences are marked.

Implementation update, 2026-09-05: WP0 and WP1 are complete. WP2 is also complete
on the production Sparse CM12 path: face-velocity support is keyed by bounded
resident cell capacity, production construction no longer builds LOD1, and BTI1's
persistent point-owner plane is a leaf-count-bounded dyadic origin hash. LOD1 remains
available only to its two explicit comparison oracles, and the exhaustive dense BTI1
point oracle remains test-only. The application and every scene still default to the
production B8/P8 Sparse CM12 solver; no opt-in UI flag is required.

WP3 is complete as well. Surface proof now walks every dyadic rung with a
level-indexed generation receipt and lease bitfield; its conservative restriction
stencil is factor-parameterized. Velocity thresholds are published as a complete
B1-through-B16 level array, quiescent component bootstrapping can select any rung,
and both CPU grading closures use physical span-aware widths with coarse-first
polarity. Ordinary CPU promotions are capped per epoch while quiet demotion no
longer pays a second 64-step residence gate. The production default consumes this
path directly. Its new gates cover all component rungs, macro faces, throttling, and
a bounded fine-start ocean-seiche census with mass/momentum transfer receipts. The
fixed all-fine/all-coarse selector and its runtime policy branches have been removed;
legacy UI values are discarded at normalization. A private initial-resolution seam
remains only for transition tests that must manufacture a chosen starting census.
The generalized proof fills only the candidate rung's `(B + 4)^3` stencil and
aggregates contiguous accepted-owner runs, avoiding the redundant finest-child
lookups that initially pushed the mini64 performance lane over its ceiling.

The proposal: stop defining a tank by its **maximum** resolution (a 128³ lattice that
caps every cell) and define it by an **initial** resolution, letting cells refine
downward as far as the scene needs.

The finding: the sparse CM12 ladder already represents a wide range of cell widths.
Moving its anchor also requires making that spatial hierarchy mutable at runtime
and making allocations follow the resident working set. Removing dense finest-lattice
structures is necessary but does not by itself deliver coarse-based vast spaces.

---

## 1. The premise correction

The method is not two-level, and has not been for some time.

- Four rungs live *inside* a brick — `1/2/4/8` cells per brick axis for B8
  (`lib/methods/adaptive-mass/sparse-brick-atlas.ts:50-58`). Level counts are already
  written `log2(brickFineResolution)`, not as a pair
  (`webgpu-sparse-cm12-resident.ts:1256-1258`); the init step calls itself
  *"Build four-rung and 2:1 seam topology templates"* (`:3890-3891`).
- Macro leaves span up to 2^k bricks. `maximumMacroSpanBricks` defaults to
  `Infinity` (`sparse-brick-atlas.ts:1334-1335`); a span > 1 macro is admitted at
  rung ≤ `ladder.coarseResolution` (`:1080-1084`); the presentation ABI carries
  `spanLog ≤ 30` (`webgpu-sparse-cm12-resident.ts:2192-2193`).
- Cell edge = `brickFineWidth × span / resolution`
  (`sparse-atlas-composite-projection.ts:388, 526, 596`). Accepted cell edges
  therefore already range **1 → 512 finest cells**.

A 512:1 adaptivity ratio is deeper than any published result surveyed in §8
(Ando & Batty run ~6–7 levels; Aanjaneya "two or three").

### The asymmetry

The structure is **unbounded upward and hard-floored downward**, and the floor is the
address space, not a policy.

Every geometric field in the solver is an integer count of finest cells —
`minimumFine`, `maximumFine`, `centerFine`, `widthsFine`, `volumeFineCells`,
`areaFineCells2` (`sparse-atlas-composite-projection.ts:39-60`). WGSL floors it:
`vec3u(cellWidths(cell))` (`webgpu-sparse-cm12-resident.wgsl.ts:2613`),
`spans = max(vec3f(1.0), spansInput)` (`:2625, 2724, 2757`). A brick's candidate arena
is exactly `B³` (`:375-376`). Refinement regions say it in the header:
*"it never refines through the cap"* (`sparse-cm12-refinement-regions.ts:1-9`).

**The declared finest lattice remains the refinement floor.** Runtime re-rung can
refine and coarsen within the supported catalogue, but there is no rung below
`finestCellSize_m`. That floor is compatible with the current vast-space target.

---

## 2. Three readings of "unlimited", with verdicts

| Reading | Verdict |
|---|---|
| **A. Truly unbounded depth, no declared floor** | Not recommended. Nobody ships it; the cost is the swept-support coupling in §6, and it buys nothing any paper can validate. |
| **B. Re-anchor — retain a declared floor, start coarse, materialize on demand** | **Required target.** Includes bounded residency and runtime macro split/merge within the existing lattice; it is more than a sizing change. |
| **C. Refine below the declared finest lattice or expand the address representation** | Separate programme. Not required to prove coarse-based ocean-seiche. |

Reading B is DCGrid's model (`docs/papers/raateland-2022-dcgrid.txt`): per-level sparse
grids, coarsest level dense, refine at the fine end, and **cap blocks rather than
depth** — refinement requests that do not fit are cancelled. That converts an unbounded
structural question into a bounded allocation question, which is the right shape for
this codebase.

Note what B does *not* change: the finest lattice remains the unit of length and every
width remains an integer ≥ 1. Declaring the floor deeper is a rescale of the integer
lattice, not a new representation. That is why it is tractable.

---

## 3. What already works — the discretization is ratio-generic

This is the substantive good news, and it is stronger than expected.

- **The pressure operator is a genuine finite-volume mortar.** `A = Gᵀ(W/θ)G` with
  `dualWeight = area × distance` and coefficients `±1/distance`
  (`webgpu-sparse-cm12-resident.wgsl.ts:5023-5090` SpMV, `:4738-4818` diagonal,
  `:5069-5087` RHS, `:5586-5597` projection). No uniform-h assumption outside the
  6-face fast path at `:5033-5052`, which only fires when the cell has no T-junction.
- **The seam row builder does honest overlap-area quadrature.** `appendBrickInterface`
  (`sparse-atlas-composite-projection.ts:622-748`) tiles the pairwise overlap rectangle
  by the coarser side's *physical* cell width and emits one term per cell with positive
  tangential overlap. There is no `2`, no `2×2`, and no subface constant in it. Row kind
  (`"brick-face"` vs `"mixed-seam"`) is *derived* by comparing widths (`:739-742`), not
  asserted from levels.
- **Verified at 4:1 this session.** An 8|2 pair was pushed through the production
  topology compilers with the grading gate bypassed. BTI1, its adversarial validator
  (cell partition + row partition + dense point-owner oracle) and BFP1 all passed,
  producing `mixedSeamTermCountHistogram [[17,4]]` — 17-term rows, exactly
  `4×4 fine + 1 coarse`, `rowsPerAddressHistogram [[1,1700]]`, zero collisions. Term
  counts scale 5 / 17 / 65 at ratios 2 / 4 / 8.
- **The row-term ABI was deliberately sized for 16:1.**
  `webgpu-sparse-cm12-resident.ts:1310-1319`: *"Nine count bits cover the widest
  supported B16 macro/fine face (256 fine endpoints plus its coarse endpoint)."*
- **Transport is volume-weighted, not ratio-2.** The conservative column condition
  `Σᵢ VᵢAᵢⱼ = Vⱼ` lives in shared numerics (`lib/core/cm12-numerics.ts:65-99`) and is
  used verbatim on the GPU (`webgpu-sparse-cm12-resident.wgsl.ts:3330-3335, 3380-3385`).
  Gamma diffusion is ratio-aware and exactly antisymmetric (`:3725-3765`).
- **24 seam consumers in the resident shader are arity-blind** — every one is a
  `rowTermOffset … + rowTermCount` loop. There are zero hardcoded seam fans.
- **The aggregate pressure hierarchy is already fully N-level**
  (`webgpu-sparse-cm12-resident.ts:3017-3105`,
  `sparse-cm12-persistent-pressure-cache-aggregate.wgsl.ts:117-125`). "Coarse" there
  means "aggregate", indexed by `hierarchyLevelCounts`/`hierarchyLevelOffsets`.
- **A four-rung B8/B4/B2/B1 ladder is an asserted CPU configuration** —
  `tests/sparse-atlas-cm12-transport.test.ts:44-61` asserts
  `mixedSeamRowCount === 4*4 + 2*2 + 1`.

**The seam is not the blocker.** Where 2:1 is genuinely exploited rather than merely
enforced is the *compiled artifact*: the GPU never builds seam geometry, it reads a
2-bit `rowKind` and selects a host-prepacked row, and the prebuilt library contains
adjacent-rung pairs only (`webgpu-sparse-cm12-resident.ts:1918-1937`).

### Original coverage caveat — addressed by WP0

Ratio-genericity is a property of the code, **not something the suite would catch a
regression in**. Every fixture and gate exercises only the 8/4 pair
(`tools/report-sparse-cm12-brick-tile-gate.ts:61`,
`tools/benchmark-sparse-cm12-brick-tile-services.ts:42`,
`tests/sparse-cm12-brick-tile-image.test.ts:64-87`). The four-rung ladder appears in
exactly one CPU test. **Any work here needs a mixed-ratio fixture first.**

---

## 4. What blocks re-anchoring (reading B)

### 4.1 Original dense finest-lattice structures — addressed by WP2

This is the wall. Nothing else matters until it is gone.

| Structure | Sizing | Measured |
|---|---|---|
| `faceVelocitySupport` | `4 floats × denseCellCount`, where `denseCellCount = atlas.dimensions[0]*[1]*[2]` (`webgpu-sparse-cm12-resident.ts:2704-2706`, call `:3931`) | 9.0 MiB long-dam, **37.5 MiB ocean-seiche** |
| LOD1 logical owner directory | 2 words per logical brick over the whole `brickDimensions` product (`sparse-cm12-logical-owner-directory.ts:180-183`) | built on host **always**; uploaded only in two QA modes |
| BTI1 brick/tile image | dense spatial-owner table over `finestDimensions/4`, with `BTI1_FINEST_DIMS` baked as WGSL literals (`sparse-cm12-brick-tile-image.wgsl.ts:45-48`) | probe/oracle today, B8-gated at `.ts:232-234` |

`faceVelocitySupport` scales 8× per level of extra depth, independent of sparsity.
Three more levels on ocean-seiche is 512× → ~19 GB. It is a deliberate read cache —
the comment at `webgpu-sparse-cm12-resident.wgsl.ts:1986-1989` records that resolving a
TEI owner per RK2 corner costs 2.4–5.4× more — so removing it has a real, measurable
price that must be paid deliberately, not assumed away.

WDR1 (`sparse-cm12-world-directory.ts`) is the right replacement shape for LOD1: signed
i32 coordinates plus `spanLog`, open-addressed at ≤50% load, free list, genuine
recycling. It is already production for ownership.

### 4.2 Original rung-policy limitations — addressed by WP3

Construction can already place every rung — `sparse-brick-atlas.ts:1578-1690` computes
`distanceRung = log2(policyFine) − (distance − rings + 1)` and is ladder-generic. The
*runtime* policy cannot follow:

- Promote is `min(brickFineResolution, 2 × current)`, demote is `resolution / 2` — one
  rung per epoch (`sparse-atlas-resolution-policy.ts:400-434`).
- **Surface bricks can only reach `fine/2`.** The B8→B4 representability proof is the
  single largest two-tier debt (~60% of it): storage is `vec2f(fine, coarse)`
  (`webgpu-sparse-cm12-resident.wgsl.ts:9369`), the lease is one bit
  (`ACTIVITY_SURFACE_B4_LEASE`, `webgpu-sparse-cm12-resident.ts:1132`), proof word 39
  holds exactly one value (`:9299-9385`), and the shader says so:
  *"an accepted B4 surface cannot descend further without a future proof
  implementation for that rung"* (`:6785`).
- `coarsenLargeQuiescentComponents` converts to `ladder.coarseResolution` only —
  exactly one rung below max (`sparse-brick-atlas.ts:1912-1919`).
- `velocityResolutionFloor` is a 3-entry `vec4f` whose fourth lane is already spent on
  thin-feature width (`webgpu-sparse-cm12-resident.wgsl.ts:5948-5954`, uniform comment
  at `:1144`). At `fine=16`, rung 2 is already unreachable.
- Promotion is unconditional every step; demotion is epoch-gated *and* throttled to
  `prepareBricksPerFrame = 64` (`:6836-6861`, `:6997-7001`). **Refinement is unthrottled
  on both axes** — the wrong bias for a design that starts coarse.

A deep anchor makes most of the domain want to be very coarse. Today it can only get
there one rung per epoch from an already-fine start, with the surface pinned at `fine/2`.

### 4.3 Original capacity blockers — WP4 remains in progress

Real demand-driven *residency* exists — WDR1 hash + free lists, a topology page LIFO,
a presentation page LIFO, and genuine retirement
(`retireUnsupportedEmptyBricks`, `webgpu-sparse-cm12-resident.wgsl.ts:8760-8794`).
What does not exist is demand-driven *capacity*:

- `GPU_TOPOLOGY_PAGE_POOL_MAXIMUM = 512` (`webgpu-sparse-cm12-resident.ts:1142`); a
  1024 parameter for curved initial liquid is staged in the working tree but
  uncommitted.
- No spill, no overflow, no fallback. Every exhaustion path fails closed and retries
  next frame. The topology fault word 29 is **never read back by host code**.
- Host rung catalogue budget: 2048 mutable bricks / 250k accepted cells / 750k accepted
  rows (`:1183-1218`). Above it, `mutableBrickKeys` is empty and **runtime re-rung is
  disabled entirely** — `tests/sparse-cm12-adaptive-resolution-lifecycle-dawn.test.ts:66-67`
  says *"Large domains intentionally skip host-built resolution variants."*
- Growth roots require B8: `allocateSparseWorldFrontier` early-returns unless the source
  leaf is at `brickFineResolution` (`:7141`). **A coarse frontier cannot page in new
  world** — this is the Figure 6 stall
  (`docs/SPARSE_CM12_FIGURE6_PAGING_HANDOFF.md:44-52`), and it bites a coarse-start
  design directly.

### 4.4 The presentation address ceiling

The page key is 11/10/11 bits, signed-biased
(`webgpu-sparse-cm12-resident.ts:2128-2162`): x ∈ [-1024,1023], y ∈ [-512,511],
z ∈ [-1024,1022] → 2048×1024×2047 pages = **16384×8192×16376 finest cells**. Y is the
tight axis at 8192. Checked at `:3907-3911`.

The renderer contract is one global `fineCellWidth`
(`lib/core/fine-levelset-brick-abi.ts:101`, packed at
`lib/core/compact-fine-levelset-phi.ts:228`) with an integer span multiplier that only
goes **up** (`compactSampleSpanScale`, `:169-173`). The render path can consume
coarser-than-page levels; it has **no representation for anything finer**. Re-anchoring
is compatible with this — a deeper floor just means a smaller `fineCellWidth` — but any
design that wants per-page world-unit scales does not fit the ABI.

---

## 5. What blocks the required runtime macro lifecycle

Local refinement inside a large macro leaf eventually requires replacing it with
children, because its candidate arena is exactly `B³`. This is required for reading B
even without changing the finest lattice. The reverse operation must merge compatible
sibling leaves and reclaim their resources. Three constraints need explicit handling:

1. **`SPARSE_CM12_FACTORED_AEI_PATCHES_PER_FACE = 4`**
   (`sparse-cm12-factored-aei-topology.ts:17`), throwing at `:691-693`. Mirrored through
   the IBO ABI — 24 bounded face records
   (`sparse-cm12-interned-boundary-operators.ts:22-30`), the throw at
   `sparse-cm12-interned-boundary-image.ts:205-207`, `side * 4 + local` addressing at
   `:195`/`:532`, 3-bit-per-side count packing at `:211`, WGSL
   `const IBO1_REFS_PER_FACE:u32=4u;` (`.wgsl.ts:80`).
   A patch is keyed `${sourceLeaf}/${side}/${targetLeaf}`, so it caps **distinct
   neighbour bricks per face, not rows** — which is exactly why 4:1 *resolution* seams
   pass and why splitting does not. The enumerator feeding it has no cap
   (`compileSparseCM12StableLeafFaceNeighbors:146-158`). **Neither throw is exercised by
   any test, tool, or doc in the original snapshot.** First test whether combined
   leaf-span and physical-cell grading can preserve the four-neighbour bound during
   split/merge closure. Extend the ABI only if the required layouts exceed it; physical
   cell-width grading alone must not be assumed to bound neighbouring leaf counts.
2. **Macro leaves are immutable at runtime.** *"A macro leaf may be rerung, but it
   cannot be spatially split after it is packed into the resident catalogue"*
   (`sparse-brick-atlas.ts:1374-1381`, `:1050-1056`); mutable bricks are span-1 only
   (`webgpu-sparse-cm12-resident.ts:3818-3820`).
3. **Cell identity is `brickKey × brickFineResolution³ + local`**
   (`sparse-atlas-composite-projection.ts:404`). Children need distinct identities and
   generation-safe recycling; they cannot reuse an expanded local index inside the
   parent. Keeping B8 per child avoids widening the local stride, but every ownership,
   row, pressure, transport and presentation reference must transition atomically.

Also standing in the way of raising B instead: the B8/P8 production pin throws in three
places (`webgpu-sparse-cm12-resident.ts:3809-3811`, `.wgsl.ts:371-373`,
`sparse-cm12-brick-tile-image.ts:232-234`), the 5-bit resolution *value* field caps at
31 (`sparse-cm12-row-access.wgsl.ts:59-60`), the 64-packets-per-leaf ABI saturates
exactly at R=16 and **silently aliases into neighbouring leaves at R=32**
(`sparse-cm12-transport-execution-image.ts:16-17, 52-55`), and
`array<vec2f,(P+2)³>` needs 46.6 KB of workgroup storage at P=16 — over the 16 KB
WebGPU limit, so the shader would not compile
(`webgpu-sparse-cm12-resident.wgsl.ts:5215`). B16/P16 was already tried and reverted
(`docs/sparse-cm12-b16-p16-frame-architecture.md`): 1.44× accepted cells, 1.55× frame,
2.34× long-dam regression.

---

## 6. The physics: dt and the swept-support coupling

**dt is a constant and there is no CFL limiter anywhere in the method.**
`CM12_PAPER_DT_S = 1/30` is the default (`lib/core/cm12-numerics.ts:10`,
`method.ts:139`); the alternative is an authored `scene.numerics.maxDt_s`. Nothing
derives it from velocity, cell size, or gravity
(`webgpu-adaptive-mass-solver.ts:811-816`). One global step, no substepping
(`sparse-atlas-dynamics.ts:1343`, `webgpu-adaptive-mass-solver.ts:964`).

This is deliberate and it is CM12's strongest advantage for this proposal. Every other
adaptive method surveyed pays 2× the step count per level — Ando & Batty state it
outright: *"doubling the effective resolution implies that the time step size must be
reduced by half."* CM12's advection is unconditionally stable and mass-conservative at
CFL 8–32 by design.

**But there is a CM12-specific version of the same wall, and it compounds worse.**
With `h → h/2ᵏ` at fixed dt:

| Quantity | Scaling |
|---|---|
| Advective travel per step, in cells | **× 2ᵏ** (`webgpu-sparse-cm12-resident.wgsl.ts:2652`) |
| Gravity impulse per step, in cells | **× 2ᵏ** (`:4364`) |
| RK2 substeps per trace | `min(2ᵏ·dt|u|/h₀, 16)` — **saturates at 16**, then accuracy falls ∝ 2ᵏ |
| Velocity-extension band, physical | **÷ 2ᵏ** (fixed at 8 *cells*, `sparse-cm12-velocity-extension.ts:23`) |
| Page-activation lookahead, physical | **÷ 2ᵏ** (clamped to ±1 brick, `webgpu-sparse-cm12-resident.ts:6176-6186`) |

Travel-in-cells grows ×2ᵏ while the residency horizon shrinks ×2ᵏ — **a 4ᵏ gap**. A
trace beyond it lands on `INVALID` owners whose stencil weights are zeroed and
renormalized: mass stays conserved and is deposited in the wrong place. Silent.

The CPU oracle already does the right thing and shows the contrast —
`sparse-atlas-dynamics.ts:1243-1252` grows the halo from the *measured* displacement,
`transportHaloBricks = ceil((dt·|u|max + receiverSpan)/brickFineResolution)`. The plan
already names it: *"Large-CFL support. A one-tile halo is insufficient"*
(`docs/adaptive-mass-conserving-method-plan.md:1416-1417`).

**Porting the velocity-derived halo to the GPU lane is mandatory, not optional.**

### Solver conditioning

The pressure solve is **diagonal-preconditioned CG on a fixed budget** — 128 iterations,
1e-3 relative L2, gate re-tested every 8 (`webgpu-sparse-cm12-resident.ts:1052-1054`).
There is no multigrid; the code is labelled MGPCG in six places but the V-cycle lives in
other methods. A coarse-grid correction was attempted and **removed** as non-SPD, with
the reasoning preserved in the comment at `:5000-5010`.

For a connected fluid region `N` cells across, CG needs `O(N)` iterations, where `N` is
measured in the *smallest* cell on the path. Adding `k` levels multiplies the iteration
requirement by up to 2ᵏ **while the budget does not move** — so the failure mode is a
progressively worse projection, not a slower frame. (Analysis from the code, not
measured.) A fixed 20:1 penalty rides on top from the θ floor
`CM12_GHOST_FLUID_THETA_MIN = 0.05` (`lib/core/cm12-numerics.ts:14`).

**This is the strongest argument for building the coarse-grid correction the ladder
would finally justify.** The aggregate hierarchy planes exist and are allocated;
`pressureHierarchyAOffset`/`BOffset`/`RhsOffset` are defined and **never read**.

---

## 7. Staged work programme

Each stage carries its own gate. Follow the dependency order in the current-target
section; implementation may proceed once its prerequisites pass. Use CPU checks
first, then isolated Dawn validation, then the UI. Never run Dawn concurrently with
the browser or another Dawn process. After substantial simulation, topology or
publication changes run `npm run test:dawn:sparse-cm12`; do not weaken lanes or raise
timing ceilings. Use paired ocean-seiche and a canonical control scene for performance
changes, retaining source fingerprints, configuration and hardware with each receipt.

### WP0 — Mixed-ratio fixture (prerequisite, cheap)

Add a CPU fixture and a Dawn gate exercising an 8|2 and an 8|1 seam, plus the four-rung
B8/B4/B2/B1 row. Today ratio-genericity is real but untested; every existing fixture is
the 8/4 pair. **Gate:** operator symmetry (`minimumRayleigh ≥ −tol`), conservative
column condition, and a raw-bit digest on symmetric expansion.

### WP1 — Fix the live B1 bug (independent, ship separately)

`sparse-atlas-cm12-transport.ts:487-489` walks face-neighbour aliases with
`for (let offset = 1; offset <= 4; offset += 1)` — exactly the widest cell in a 4/8
ladder. A B2 brick (width 4) clears with zero margin; **a B1 brick (width 8) or any
macro span leaves `neighbors[slot] = -1`**, which the fast-iterative-method velocity
extension reads as an absent axis (`:596-660`). No throw, no assert — the extension band
silently loses an axis. B1 is a shipped rung today. **Gate:** a B1 velocity-extension
oracle that currently fails.

### WP2 — Delete the dense finest-lattice structures

In order of size: `faceVelocitySupport` → sparse/hashed or accept the TEI resolve cost;
LOD1 → WDR1; BTI1's dense spatial table. **Gate:** bit-exact digest on ocean-seiche and
long-dam, plus a paired frame-time capture. Expect a regression on the
`faceVelocitySupport` arm — the 2.4–5.4× per-corner resolve cost is documented. Decide
deliberately whether to pay it or to keep a *bounded* cache keyed on the active set.

### WP3 — Invert the policy anchor

**Complete (2026-09-05).** The implementation uses one bounded fine/candidate
workgroup pair per brick and stores the durable result per level; this avoids
growing workgroup memory with ladder depth while providing the requested per-rung
receipt semantics.

Generalize the surface representability proof from B8→B4 to per-rung: `vec2f` → a
per-level receipt, one lease bit → a bitfield, proof word 39 → level-indexed, and the
restriction stencil parameterized by factor instead of the literals `2`/`8`. Then
`velocityResolutionFloor`'s 3-entry `vec4f` → a per-level array, and
`coarsenLargeQuiescentComponents` → any rung. Make `enforceTwoToOneTargets`
(`sparse-atlas-resolution-policy.ts:97-123`) span-aware and macro-aware, and reconcile
its polarity with `stronglyGradeSparseBricksByCoarsening` — the two closures currently
repair in **opposite directions**.

Also fix the throttling bias: demotion is doubly gated while promotion is unthrottled,
which is backwards for a coarse-start design.

**Gate:** ocean-seiche reaches a target coarse census from a fine start within a bounded
number of epochs, with mass and momentum receipts intact.

**Scope of the completed gate:** the compact, zero-gravity transition fixture described
above. Full-size gravity-driven ocean and spatial macro adaptation are WP6 gates.

### WP4 — Demand-driven capacity

**In progress; not accepted.** Make topology capacity an explicit bounded budget with
observable exhaustion. Cancel or defer a refinement transaction that cannot fit,
preserving accepted topology and the validity of its dependent grading closure.
Cancellation must not permit unsupported transport; see WP6a's support gate. Separate
ordinary capacity cancellation from allocator corruption and read both back.

Implementation finding (2026-09-05): admitted authored re-rung already owns complete
template cells, rows and incidence. The old `allocateCandidateTopologyPages` dispatch
returned immediately for these leaves, and its geometry-only fallback could not make
an unbacked leaf publishable. Both that dormant allocator and its synthesis dispatch
have been removed. Do not reintroduce a second page identity for already-backed
topology, or gate those transfers on spare world-growth capacity. Future demand-built
candidate topology still needs complete-transaction reservation before publication.

The physical growth budget is now `AdaptiveMassSolverOptions.topologyPageBudget`
(default 512, retaining the existing curved-volume 1024 default). Diagnostics expose
WDR's actual available pages, capacity and unfulfilled allocation count through
`adaptiveTopologyPageAllocator`; they no longer mistake the legacy, separate scratch
free-list header for world residency. The canonical `topology-page-budget` lane
checks conservative authored re-rung with zero, one and 32 growth pages, no borrowed
dynamic identities, and matching ordinary/QA allocator receipts. This is a regression
gate for allocator separation, not proof of bounded on-demand template preparation.

Validation of this slice: 51 focused CPU/source tests and the stage-timing contract
pass. Figure 6 passes its traversal gate beyond 512 issued page identities. The full
14-lane canonical Dawn suite passes in 117.4 s against its unchanged 180 s budget;
mini32/mini64 medians are 30.41/42.27 ms against unchanged 40/50 ms ceilings. These
are regression receipts, not a paired performance claim. Repository-wide type
checking still reports existing harness/probe errors, and managed-pipeline compliance
still reports three direct pipeline creation calls in `lib/svo/sparse-brick-octree.ts`.

Second implementation slice (2026-09-05): `prepareSparseCM12TopologyWorkingSet`
uses the shared SCMT serializer to retain accepted cells and add only requested
candidate rungs and their affected rows. It requires physical 2:1 closure, discovers
complete macro-face halos, and emits both generations' cell/row worklists, incidence,
candidate-face tables and pressure edges. Cell/row/serialized-byte budgets return
`deferred` without mutating the accepted grid; these bounds do not yet measure peak
transient host memory. Clipped-cell restriction uses physical-volume weights.
Tests compare accepted and candidate graphs with independent full builds, exercise
budget retry, and prepare macro cells at 64h/128h without changing leaf coverage.

The production-surface-bias ocean fixture has 139,760 accepted cells and 2,400
mutable leaves. Preparing its 70 B8→B4 changes produces 144,240 cells, 437,798 rows
and a 56,574,488-byte SCMT packet. The 4,480 added cells are exactly the requested
candidate cells. This is a CPU preparation receipt, not a runtime/performance claim.
Stable AEI face-neighbor discovery now indexes dyadic origins and occupied ancestors
instead of expanding macro volumes and faces. A randomized box-overlap oracle and
million-brick-edge leaves cover holes, signed coordinates and mixed spans.

Validation of the second slice: 25 focused CPU tests pass, including complete graph
and pressure-edge comparisons across multiple construction chunks. The stage-timing
contract passes. All 14 canonical Dawn lanes pass in 92.9 s; mini32/mini64 medians
are 31.85/42.27 ms against unchanged 40/50 ms ceilings. Repository-wide type checking
continues to report the existing harness/probe errors, with none in this slice's
changed files. These receipts do not exercise live adoption of the new CPU packet.

The next unimplemented prerequisite is resident adoption of this bounded packet:
reserve all dependent stores, remap live fields and face momentum, validate the
candidate generation, atomically publish, then reclaim the replaced storage. The
current GPU catalogue still uses its existing admission policy. Runtime macro
split/merge, initial coarse-size UI controls and the ocean/vast-space acceptance
milestones remain open.

GPU storage component (2026-09-05): `SparseCM12TopologyGenerationStore` owns bounded
SCMT/membership buffer generations. Accepted, staged, allocation-in-progress and
leased retired bytes share one explicit limit; receipts expose current/peak reserved
bytes, deferred requests and allocation failures. Preparation validates membership
and generation continuity, reserves both buffers together, and rolls back partial
allocation. Cancellation and destruction preserve leased buffers until their submitted
consumers finish. An old consumer may be encoded before storage publication and
submitted afterwards while its lease remains live. Reclamation permits a previously
deferred generation to proceed.

The `topology-generation-storage` Dawn lane tests those transitions, GPU byte
readback, an injected second-allocation failure, malformed membership, stale requests,
and destruction during preparation. **This component is not wired into the resident
yet.** Its storage commit must accompany the resident's validated field, pressure,
transport and presentation binding replacement; calling it alone does not constitute
simulation adoption. It introduces no ocean-only path and no new UI mode.

Validation: all 16 canonical Dawn lanes pass in 98.1 s; mini32/mini64 medians are
31.13/43.84 ms against unchanged 40/50 ms ceilings. The final storage receipt tests
pass after adding peak/reservation accounting. The 16 focused CPU tests and
stage-timing contract pass; type checking has no errors in this slice's new files
and continues to report the existing unrelated harness/probe errors.

General transfer prerequisite (2026-09-05): the production candidate transfer now
decodes compact cell IDs using each leaf's clipped live dimensions. Restriction
skips absent children; prolongation and B4→B8 presentation reconstruction use the
same compact addressing. Candidate mass receipts use packed physical cell volumes,
and exterior face patch widths include the macro span. This removes span-one/full
brick assumptions from this portion of field transfer; it does not grant macro
candidate admission or publish the bounded preparation packet.

The new canonical `clipped-topology-transfer` lane uses a 13×10×9 domain, ordinary
demotion, then live whole-domain B1 and B8 edits. Every leaf completes the round trip;
mass, gamma and momentum transfer receipts pass. Total mass changes by
0.0000322 finest-cell volume units out of 1,170, under the 0.0001 test limit. Running
the same clipped fixture against the previous transfer code produces transfer fault
bit 2. All existing scene gates retain their original thresholds.

Validation: all 15 canonical Dawn lanes pass in 104.8 s; mini32/mini64 medians are
31.06/43.32 ms against unchanged 40/50 ms ceilings. The 21 transaction/manifest CPU
tests and stage-timing contract pass. Type checking still reports existing unrelated
harness/probe errors, with no errors in the files changed for this transfer fix.

Allow coarse frontiers to acquire demanded world coverage with a valid seam on both
sides. The current working-tree route promotes the source to B8 to meet newly grown
B8 pages; this may unblock paging, but is not the final coarse macro growth policy.

Replace the all-or-nothing host catalogue admission with bounded preparation for
the actual changing region. The production ocean's 2,400 mutable leaves must remain
adaptive despite the existing 2,048-leaf ceiling. Do not solve this by prebuilding all
rungs throughout a larger domain or merely raising the ceiling. Record host peak
memory, GPU reserved/used capacity and deferred requests.

**Gate:** Figure 6 traverses its required course and exercises allocation beyond 512
page identities; report live, peak and cumulative allocations separately. The actual
full-size ocean re-rungs under normal policy. A deliberately small budget cancels a
closure safely, preserves accepted mass/topology, and subsequently makes progress
after reclamation. A helper admission test alone does not satisfy this gate.

### WP5 — The coarse-grid correction (conditional on convergence)

The aggregate hierarchy planes are allocated and unread. A deeper ladder is exactly the
condition under which diagonal-preconditioned CG on a fixed budget stops being adequate.
Losasso's `p_a = p_1` trick is the template: accept first-order pressure to keep the
operator symmetric — *"nonsymmetric formulations … easily lead to an order of magnitude
slowdown, or in the worst case scenario problems with robustly finding a solution at
all."*

**Gate:** iteration count at fixed residual, measured across ladder depth. Defer this
work while the ocean passes its pressure residual and wave-accuracy gates. If those
fail from conditioning, WP5 becomes a prerequisite to accepting the affected scale;
do not hide a failed solve behind a fixed iteration budget.

### WP6a — Coarse macro foundation and local refinement (required)

**Depends on WP4.** Construct large aligned macro coverage directly from authored
fluid and solid boundaries without enumerating finest bricks throughout the volume.
Select initial cell widths by physical representability and policy. Preserve the
raised slab and surface detail rather than forcing every region onto a coarse rung.

Enable macro re-rung and spatial parent-to-child replacement when local refinement
requires it. Keep B8/P8; define split eligibility, child identities, conservative mass
and momentum prolongation, surface proof, neighbour closure and capacity reservation.
Publish ownership, pressure/transport connectivity and presentation as one accepted
generation. No overlapping parent/child authority, gaps or stale identities may
become visible. Resolve the face-patch constraints in §5 with CPU fixtures first.

**Swept support is part of this stage.** Size the GPU support horizon from measured
displacement, dt and physical receiver span across macro boundaries, following the
CPU oracle's approach in §6. Account for the RK2 trace limit and velocity extension.
Detect incomplete support and defer or otherwise safely handle the affected advance;
mass conservation alone cannot validate a trace silently renormalized onto the wrong
receivers. Budget pressure must never silently become fluid loss or teleportation.

**Gate:** CPU and Dawn split fixtures cover physical widths 16h, 32h and 64h,
including boundary contacts, coarse/fine faces and rollback on insufficient capacity.
On ocean, a moving feature enters an initially coarse macro region, causes localized
refinement, and crosses it with conservative transfer and valid pressure/presentation.
Report spatial leaf span separately from physical cell width.

**First UI milestone:** after these gates and the canonical regression suite pass,
open the ordinary ocean scene with coarse initialization and normal activity policy.
Make initial physical cell size/coarse coverage and the finest permitted cell size
distinct in scene settings; an initial coarse size must allow later finer detail.
Expose the physical-width distribution, accepted cell/leaf counts, split/merge counts,
GPU memory, frame time and deferred requests in the existing diagnostics. This is a
testable intermediate delivery; automatic recovery of coarse coverage follows in WP6b.

### WP6b — Merge, reclaim and validate ocean-seiche in the UI (required)

**Depends on WP6a.** Merge complete compatible sibling sets when activity, surface
representability, solid geometry and grading allow it. Restrict mass and momentum
conservatively; publish the parent atomically; reclaim child topology, candidate,
pressure and presentation allocations only after their last consumers finish.
Use hysteresis to prevent repeated split/merge oscillation without permanently
retaining fine regions after a wave has passed. Partial child sets cannot merge.

**Gate:** repeat split→merge→split through several generations, including allocator
reuse, budget cancellation and macro boundaries. There must be no stale references,
monotonic capacity leak or loss of represented liquid. A quiet recovery fixture must
return to a bounded coarse census; the dynamic ocean need only merge where its
measured activity and representability permit it.

**Ocean acceptance:** use the authored 2.5 cm finest lattice, raised slab, gravity and
ordinary production policy. Run at least an outward crossing and return reflection
(approximately four simulated seconds for the original 8 m basin). Compare wave
phase, amplitude and seam reflection with a validated finer reference. Record mass
drift, transfer momentum errors, pressure residuals, solver iterations and support
faults. Agree and record numeric error tolerances from the reference before judging
the candidate; preserve existing stricter regression tolerances.

At initialization and after motion, report liquid-volume-weighted fractions in
physical-width bins 1h, 2h, 4h, 8h, 16h, 32h, 64h and larger, plus cell and leaf counts.
Target a majority of represented liquid volume at widths ≥16h where geometry and
wave accuracy permit; a thin fine surface should not disqualify coarse bulk coverage.
Demonstrate 32h and 64h wet cells in appropriately enlarged/deepened ocean variants
when the original basin's boundaries do not permit them. Do not count dry or unused
macro coverage as satisfying the coarse-liquid target.

Verify the same case in the UI: no cracks or disappearing deep water, refinement
follows the feature, eligible regions merge, and reset restores the coarse start.
Full-domain minimum-cell-size clamps, zero gravity and a compact replacement lattice
remain diagnostic comparisons, not substitutes for acceptance.

### WP7 — Vast-space scaling and optional deeper anchoring (required scaling gate)

**Depends on WP6b and on WP5 if convergence requires it.** Grow ocean basin extent
at fixed finest cell size and fixed local feature scale. Include larger/deeper variants
that admit 32h, 64h and progressively coarser wet cells. Preserve the authored wave's
world-space geometry when resizing; do not accidentally rescale brick-seeded liquid.
Run long enough for the wave to enter initially coarse coverage at each scale.

Record logical finest volume, coarse wet volume, resident leaves, active cells,
topology/template/presentation bytes, host initialization peak and time, simulation
GPU stage times and rendered frame p50/p95. Additional calm space should cost coarse
coverage and necessary boundary detail, with no full-domain fine catalogue or hidden
disabled adaptivity. Fix any dense initialization/presentation path exposed here.

Respect the signed coordinate and presentation address ceilings in §4.4. Declare the
supported physical extent at each finest cell size and reject unsupported extents
explicitly. Coarse cells do not remove address limits. Extend the address ABI only
when a required measured ocean scale reaches it; do not claim unbounded space.

Only after this gate, deepen the finest lattice if local feature requirements warrant
it, repeating swept-support, pressure and allocation gates. Larger quiet spaces do
not by themselves require a smaller finest cell.

---

## 8. Literature bearings

Read from `docs/papers/` this session.

- **CM12 itself is a one-level uniform method.** *"We discretize the simulation domain
  using a regular staggered grid"*; every result is a uniform box; the word "octree"
  does not appear. dt = 1/30 at **CFL 8–32** in every example. The entire brick ladder
  is a local invention — which means the paper offers zero validation for N-level seams,
  and also that adding rungs is not a bigger departure than what already ships.
- **Nobody ships unlimited depth.** Aanjaneya: *"two or three levels of refinement."*
  Ando & Batty: 2:1 graded, *"Allowing more than one level of change would increase the
  algorithmic difficulty and potentially introduce unexpected visual artifacts."*
  Narita 2025 names the physics reason: *"numerical reflection arising from the rapid
  change in the resolvable accuracy."* Only Losasso 2004 permits unrestricted jumps.
- **The 2:1 rule is a discretization requirement, not a code convenience.** A deep
  ladder therefore needs a graded skirt whose width grows with depth — cells you did not
  want to refine get refined anyway.
- **The dominant published cost of depth is dt**, and CM12 is the one method that
  escapes it — see §6 for the CM12-specific replacement.
- **The solve is usually not the wall.** Ando & Batty: projection 1.21 s of a 21.96 s
  step (5.5%); octree construction + sizing evaluation 5.36 s (24%). Losasso: pressure
  ~25%. Aanjaneya is the counterexample at 31–49%. This matches HEAD, where
  `resolution-planning` costs **1.049 ms on frames where nothing changes** and
  `scheduleTopologyPreparation` burns ~0.79 ms/frame on a single GPU lane
  (`docs/sparse-cm12-non-pressure-temporal-coherence-handoff.md:80-107`).
- **A hierarchy that is not sparse is worse than none.** DCGrid at full refinement:
  **2.19× the memory and 1.71× the time** of a plain uniform grid at the same effective
  resolution; break-even ~50% cell occupancy. This is the honest risk for a deep anchor
  on a scene that then refines everywhere.
- **DCGrid is the best structural template** — GPU-native, multi-level in the physics,
  demand-allocated, memory-budgeted, with precomputed apron indices so cross-level
  stencils compile to divergence-free code. Its nesting invariant (coarsest dense,
  refine at the fine end) **is** the initial-resolution model.

---

## 9. The proportionality trap

The known failure mode of this programme is already documented, and it is not about
depth. `docs/sparse-cm12-min8-domain-work-analysis.md:63-71`: a full-domain coarsening
region *"does not merge the 512 logical bricks, shrink the immutable multi-rung template
library, reduce pressure hierarchy capacity, or reduce the one-fine-page-per-brick
presentation allocation"* — **a 113× reduction in accepted cells bought only 1.69× frame
time.**

A coarse start will reproduce this unless storage follows macro coverage and the
changing region. WP2 and WP4 are prerequisites; WP6's merge/reclamation and WP7's
scaling measurements establish the missing spatial proportionality.

**Acceptance combines physical accuracy, interactive performance and sparse scaling.**
Before the UI acceptance run, record a numeric frame-time and memory budget on the
target hardware using the current ocean baseline. Keep simulation GPU time separate
from rendering and initialization. Fixed work means halving accepted cells need not
halve the entire rendered frame; report that fixed cost rather than hiding it in a
cell-count ratio. Budget values are still to be established, not passed by this plan.

For the extent sweep, plot measured allocations and stage times against resident
coarse coverage and local refined work, alongside logical finest volume. Acceptance
requires continued adaptivity, reclaimed resources and absence of finest-volume-sized
work in production. Increasing maximum span, reducing cell counts under a hard clamp,
or showing one large leaf is insufficient. Retain before/after receipts for both the
ordinary ocean and the larger variants so that a vast-space win cannot conceal an
unusable ordinary UI scene.

---

## 10. Open questions

1. What coarse initial physical width best preserves the ocean wave while keeping most
   liquid volume at ≥16h? Measure 16h/32h/64h coverage where basin geometry permits it;
   retain 2.5 cm as the first milestone's finest cell size.
2. Can leaf-span grading preserve the four-patches-per-face ABI through every required
   split/merge closure, or must the boundary representation change? Resolve with CPU
   fixtures before changing GPU packing.
3. What frame-time and memory budgets define a useful UI run on the target hardware,
   and what phase/amplitude tolerances does the validated reference support? Record
   these before candidate acceptance; do not move ceilings to accommodate regressions.
4. Which basin sizes first expose presentation coordinate ceilings, dense boundary
   initialization or pressure conditioning? Use the WP7 sweep to determine the next
   bottleneck rather than adding finer depth speculatively.
5. How much surface and support detail is physically necessary as macro span grows?
   Measure it in world units and physical cell widths; a fixed finest-brick band is
   not an adequate vast-space policy.
