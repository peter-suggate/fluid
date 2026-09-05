# Sparse CM12 initial resolution — anchor the ladder at the top, not the bottom

Original status: exploration, 2026-09-05. At the time no code had changed; every
claim below was read from HEAD or measured in that session, and inferences are marked.

Implementation update, 2026-09-05: WP0 and WP1 are complete. WP2 is also complete
on the production Sparse CM12 path: face-velocity support is keyed by bounded
resident cell capacity, production construction no longer builds LOD1, and BTI1's
persistent point-owner plane is a leaf-count-bounded dyadic origin hash. LOD1 remains
available only to its two explicit comparison oracles, and the exhaustive dense BTI1
point oracle remains test-only. The application and every scene still default to the
production B8/P8 Sparse CM12 solver; no opt-in UI flag is required.

The proposal: stop defining a tank by its **maximum** resolution (a 128³ lattice that
caps every cell) and define it by an **initial** resolution, letting cells refine
downward as far as the scene needs.

The finding: the sparse CM12 ladder is already about nine levels deep. It is anchored
at the bottom and grows upward. The work is not to build a deeper hierarchy — it is to
move the anchor, and to delete the three dense structures that are sized from the
finest lattice regardless of sparsity.

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

**Adaptivity in CM12 today is coarsening-only.** There is no rung below
`finestCellSize_m` and no address for one.

---

## 2. Three readings of "unlimited", with verdicts

| Reading | Verdict |
|---|---|
| **A. Truly unbounded depth, no declared floor** | Not recommended. Nobody ships it; the cost is the swept-support coupling in §6, and it buys nothing any paper can validate. |
| **B. Re-anchor — declare the floor deep, start coarse, materialize on demand** | **Achievable, and mostly a sizing problem.** This is the recommended target. |
| **C. Depth beyond `log2(B) + spanLog` by splitting bricks spatially** | Structurally hard. Separate programme; see §5. |

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

### Coverage caveat

Ratio-genericity is a property of the code, **not something the suite would catch a
regression in**. Every fixture and gate exercises only the 8/4 pair
(`tools/report-sparse-cm12-brick-tile-gate.ts:61`,
`tools/benchmark-sparse-cm12-brick-tile-services.ts:42`,
`tests/sparse-cm12-brick-tile-image.test.ts:64-87`). The four-rung ladder appears in
exactly one CPU test. **Any work here needs a mixed-ratio fixture first.**

---

## 4. What blocks re-anchoring (reading B)

### 4.1 Three dense structures sized from the finest lattice

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

### 4.2 The coarsening policy can only walk one rung

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

### 4.3 Capacity is fixed at construction and never grows

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

## 5. What blocks brick splitting (reading C)

Depth beyond `log2(B) + spanLog` requires splitting a brick into children, because the
candidate arena is exactly `B³`. Three walls, in the order they are hit:

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
   any test, tool, or doc.**
2. **Macro leaves are immutable at runtime.** *"A macro leaf may be rerung, but it
   cannot be spatially split after it is packed into the resident catalogue"*
   (`sparse-brick-atlas.ts:1374-1381`, `:1050-1056`); mutable bricks are span-1 only
   (`webgpu-sparse-cm12-resident.ts:3818-3820`).
3. **Cell identity is `brickKey × brickFineResolution³ + local`**
   (`sparse-atlas-composite-projection.ts:404`). Refining past B overflows the per-brick
   id stride.

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

Each stage carries its own gate. Do not start a stage before its predecessor's gate is
green. Per the benchmark discipline: one Dawn run per arm per scene, two scenes, CPU
checks first; never variant compiles on Dawn.

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

### WP4 — Demand-driven capacity

Raise the page pool from a construction constant to a budget with a **stated policy on
exhaustion**. DCGrid's answer is the right one: cancel the refinement request, do not
fail. Read back the topology fault word (currently never read). Lift the B8-only growth
root so a coarse frontier can page in world
(`webgpu-sparse-cm12-resident.wgsl.ts:7141`). Reconsider the host template catalogue —
at ~8× per rung it is already auto-disabled above 250k cells, which means the largest
scenes have **no runtime adaptivity at all** today.

**Gate:** Figure 6 paging passes above 512 pages; a scene above the template budget
still re-rungs.

### WP5 — The coarse-grid correction (optional, high value)

The aggregate hierarchy planes are allocated and unread. A deeper ladder is exactly the
condition under which diagonal-preconditioned CG on a fixed budget stops being adequate.
Losasso's `p_a = p_1` trick is the template: accept first-order pressure to keep the
operator symmetric — *"nonsymmetric formulations … easily lead to an order of magnitude
slowdown, or in the worst case scenario problems with robustly finding a solution at
all."*

**Gate:** iteration count at fixed residual, measured across ladder depth.

### WP6 — Brick splitting (separate programme, only if needed)

Reading C. The AEI/IBO 4-patches-per-face cap first, then macro splitting, then the cell
id stride. Do not start this until WP2–WP4 show that `log2(B) + spanLog` is genuinely
insufficient.

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

A deeper anchor that starts coarse will reproduce this exactly unless WP2 and WP4 land
first. Coarsening the physics does not coarsen the allocations, and the allocations are
what the finest lattice sizes.

**Acceptance for the whole programme should therefore be a proportionality oracle, not a
depth demonstration:** halving the accepted cell count must roughly halve the frame.

---

## 10. Open questions

1. What floor do you actually want? Six levels below today's initial resolution covers
   every scene in the library. Ten starts testing the presentation Y axis (8192 finest
   cells).
2. Is the `faceVelocitySupport` cache worth keeping in bounded form? The 2.4–5.4×
   per-RK2-corner resolve cost is documented but was measured against a dense cache, not
   against a hashed one sized to the active set.
3. Should the coarse-grid correction land before or after the anchor moves? Before is
   safer (the ladder is shallow, so the regression surface is small); after is cheaper
   (the deeper ladder gives the hierarchy something to do).
4. Does the surface band stay one brick thick? The B16 post-mortem found a larger B
   doubles the band's cell count for free. That mechanism **inverts in your favour** if
   depth is added downward from a smaller initial brick rather than upward from a bigger
   one — worth confirming before committing to a brick size.
