# Stage Collapse Plan — grouping the frame by data access, not algorithm order

**Written 2026-07-25, against e0441ee.** Measured state: 125.95 ms/advance
free-run on `benchmark:power-dam-ui` (189.28 attributed across 18
`splitProductionPhase` boundaries). Companion evidence: the five-stage
re-audit recorded in the perf handoff and session memory; every file:line
below comes from that audit of the current tree.

## 1. Thesis

The advance is currently organized as ~20 stages in *algorithmic* order
(topology → solve → extrapolation → advection → redistance → restriction →
publication). Each stage re-opens the same data structures, re-derives
structure that is invariant per topology generation, re-validates its inputs,
and publishes transactionally to the next stage as if crossing a trust
boundary. The measured consequences:

- ~60-70% of frame time is **structure work whose inputs did not change**
  (directory builds, descriptor emits, sorts, adjacency validation,
  selector-row resolution, graph re-emission).
- The fine lattice (262k samples) is **swept end-to-end ~15-20 times per
  frame** by separate kernels that each read committed phi and write one
  small derived output (JFA resolve/validate/commit, volume ×5, energy ledger
  ×3, restriction, summary measure, residency marks).
- Every stage seam pays transactional scaffolding: A/B arena copies, carry
  passes, validation reduces, per-probe generation re-checks — machinery that
  exists *only because the seams exist*.
- The 18 phase boundaries themselves cost ~0.5 ms each (~5 ms/frame measured:
  "Final pressure row assembly" is one 108-WG dispatch yet bills 0.60 ms).

The collapse: reorganize the frame into a small number of **engines, one per
data domain / access pattern**, and one **conditional structure epoch**. Work
moves to the engine that owns its data, not the algorithm step that wants its
result. Structure work runs only when its inputs change, and each engine
reads its domain once per frame.

This is also the paper's own shape: Aanjaneya et al.'s Table 2 shows
everything except the solve as near-free, because on their CPU
implementation extrapolation/publication are table reads over prebuilt
structure. The stage collapse is the GPU-native equivalent.

## 2. Data domains and their true invariants

| Domain | Contents | Changes when |
|---|---|---|
| **Catalog** | descriptors, tetra, vertex selectors, §6.3 coefficients, (to add: integer support/neighbor offset tables) | never (per sim) |
| **Coarse structure** | tiles, leaves, owner pages, row directory, identity tables, descriptors, faces+geometry, operator/CSR, RAP, band-row directory, adjacency/incidence, selectorRows, restriction links, summary structure, residency structure | topology generation (dam break: most frames, but with a *small delta*) |
| **Coarse values** | coarse phi, face/row velocities, pressure, band phi | every frame |
| **Fine structure** | page directory, resident set, support masks | interface crosses a brick (subset of frames; small delta) |
| **Fine values** | fine phi, transport work channels, CPT seeds | every frame |
| **Cross-scale measures** | restriction aggregates, summaries, volume/ledger measures, residency worklists | derived from values (measure) + structure (shape) |

The audit's central finding restated: the *values* rows of this table are
cheap (a handful of row/brick sweeps). Nearly all of the 126 ms lives in the
*structure* rows being recomputed per frame, or in value kernels paying
per-thread structure resolution (owner searches, directory probes) that the
structure rows should have precomputed.

## 3. Target frame graph

```
┌────────────────────────────────────────────────────────────────────┐
│ STRUCTURE EPOCH  (conditional: runs iff dirty counts ≠ 0;          │
│                   cost ∝ delta, not domain)                        │
│  E1 coarse topology: refine/balance/compact (already delta-driven) │
│  E2 directories: row dir, identity tables, selectorRows,           │
│     (level,cell)→row direct table — scatter-built, one pass each   │
│  E3 derived structure: descriptors, faces+coefficients, operator   │
│     rows, RAP, band directory (scatter + prefix over identity      │
│     table — replaces all 4 radix sorts), adjacency, incidence,     │
│     restriction links, summary shape, residency structure          │
│  one shadow-arena build → one validation reduce → one flip         │
└────────────────────────────────────────────────────────────────────┘
   then, every frame:
 ROW ENGINE A   coarse phi advect + external forces + face velocity
                completion (values only; neighbors via epoch tables)
 SOLVE          Galerkin, adaptive cycles, convergence-broken coarsest
 ROW ENGINE B   projection + velocity extension values + band phi
                values + publication values (one row/face sweep)
 BRICK ENGINE A transport m-segments + CPT seed, fused (one lattice
                read; seeds emitted from the same shared-memory tile)
 CPT WAVES      k JFA passes, k adaptive: warm-started collar repair
                (interface moves ≤ fineFactor cells/step ⇒ strides
                [4,2,1,+1], not [32,16,8,4,2,1,+1])
 BRICK ENGINE B "harvest": resolve+commit phi + volume measure +
                restriction aggregate + summary measure + residency
                marks + energy ledger — ONE fused lattice read,
                per-brick partials, one tiny combine dispatch
 EPOCH GATE     dirty counts written by the engines arm next frame's
                epoch (or same-frame via indirect encode)
```

~7 boundaries instead of 18-20. Two-level gating everywhere: an engine or
epoch *skips* when its input generation is unchanged, and when it runs its
cost is proportional to the published delta, never the domain.

## 4. What this deletes (the "hugely less work" inventory)

1. **Per-frame structure re-derivation** → epoch-only:
   - band-row rebuild: emit×4 + sort×4 over all core rows every frame with
     no gate (face-closest-point.ts:2645, :2705-2861). The canonical order
     (size-major, cell) is exactly the identity-table layout order — a
     prefix scan over the 24,576-slot table replaces the 4 single-WG radix
     sorts outright.
   - transient power graph emit+validate every step though geometry is fixed
     per generation (face-closest-point.ts:4365, :4368).
   - selectorRows rebuilt with 307k findSite searches per advance
     (power-coarse-levelset.ts:322-324) — epoch scatter of a direct table.
   - faces re-clipping Sutherland-Hodgman 2-3× per face per generation and
     never binding catalog direct tables (power-faces.ts:927-958).
   - adjacency full revalidation per frame (face-closest-point.ts:3155).
2. **Redundant full-lattice sweeps** → one harvest kernel: JFA
   validate (dead at residualTolerance:1) + resolve + commit tails, volume's
   3 reductions + 2 applies (owner() resolvable once per WG, analytic
   re-measure: volume.ts:228-232), 3 ungated ledger sweeps
   (octree.ts:3009-3016), restriction read, summary measure, residency marks.
3. **Seam scaffolding**: per-stage carry/validate/A-B copies
   (face-closest-point.ts:1536-1550 et al.), per-probe publication-header
   re-validation inside inner loops, per-stage generation stamps — replaced
   by one epoch commit point per domain.
4. **Fixed schedules** → adaptive: solve cycles 20→residual-driven
   (octree.ts:1200), coarsest 64-iteration CG convergence break
   (power-galerkin.ts:316-346), JFA 7→~4 warm-started passes, band-phi 16
   rounds → compacted-frontier with early-out, 8×2 repair waves →
   indirect-gated on unresolved count.
5. **Boundary tax**: 18 splits → ~7 (≈ −4-5 ms by itself). Keep a
   `FLUID_ENGINE_SPLIT=fine` env that re-inserts the old boundaries for
   attribution work.

## 5. Load-bearing assumptions (verify before each phase)

- **JFA warm start** assumes fine phi from frame N−1 is a valid distance
  field within the maintained band and the interface moves ≤ fineFactor
  fine cells (already enforced: octree.ts:2992-2994; measured displacement
  already published: transport.ts:474-475). New bricks entering the band
  still need full-stride seeding — seed them from the coarse phi restriction
  (already computed) and let the collar repair refine.
- **Volume analytic re-measure** assumes the correction is a uniform shift
  clamped to ±0.5h (it is: volume.ts docstring) so ΔV = shift·A from the
  first measurement suffices; keep the full path behind a debug env.
- **Reduction reordering changes float results.** Bit-exact parity with the
  current pipeline is not a goal; the acceptance bars are the existing ones:
  `benchmark:power-dam-ui` gates, zero validation errors, IoU parity vs the
  Dawn smoke baseline, energy-ledger drift bounds. Snapshot both before
  starting.
- **Fused kernels vs limits**: engines need more bindings per kernel; group
  buffers into per-domain arenas (offsets in a control word) rather than
  raising binding counts. Watch register pressure — fuse within a domain,
  never across domains (a row kernel never touches the fine lattice).
- **Divergence**: where a fused engine would branch per row class
  (interior/transition/boundary), prefer specialized pipelines dispatched
  over class-partitioned index ranges (the epoch can emit class partitions
  for free during compaction).

## 6. Migration order (each step ships green)

| Phase | Work | Est. after |
|---|---|---|
| 0 | Engine-tagged instrumentation + `FLUID_ENGINE_SPLIT`; collapse phase boundaries to 7 | ~120 ms |
| 1 | **Harvest fusion** (Brick Engine B): fold JFA resolve/commit, volume measure (per-WG owner, analytic), ledger (gated), restriction, summary measure, residency marks into one sweep + combine | ~95-100 ms |
| 2 | **CPT warm start** + adaptive strides + delete dead validate; band 23→16 | ~80-85 ms |
| 3 | **Structure epoch v1**: gate band rebuild + transient graph + adjacency validation + selectorRows on generation delta; sorts → identity-table prefix compaction; epsilon dirty-marking upstream so the existing skip machinery finally fires (topology.ts:1531-1534, power-coarse-levelset.ts:487) | ~55-65 ms |
| 4 | **Solve adaptivity**: cycles residual-driven, coarsest CG break, RAP refresh gated on operator delta | ~42-50 ms |
| 5 | **Row engines**: fuse per-row value passes; catalog integer offset tables (kills float-transform + ownerAt-discard paths, face-closest-point.ts:2925, :2586) | ~35-42 ms |
| 6 | **Transport+seed fusion** + directOwner via owner-topology O(1); sparse residency insertion parallelized (dense-arm pattern, fluid-brick-residency.ts:891-967) | ~28-35 ms |

Endgame after collapse: the "nothing moved" frame is epoch-skip + 4 value
sweeps + a short solve — the single-digit-ms structure the representation-V2
plan targets.

## 7. Non-goals

- No change to the paper's algorithm, discretization, or the fail-closed
  *outcome* (a rejected epoch still refuses to flip).
- No speculative megakernel across domains; fusion stops at domain edges.
- No bit-exact preservation across reduction reordering (IoU/energy bars
  instead).
