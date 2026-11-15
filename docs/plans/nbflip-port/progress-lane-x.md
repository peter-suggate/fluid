# NB-FLIP port, lane X: structural redesign (progress)

**Brief.** The user's direction is "adapt the narrowband flip method at a structural level to be more like the
optimized version". Lane X is design work only: no changes under `lib/`.
- **Deliverable:** `docs/plans/nbflip-structural-2026-11-15.md`.
- **Scratch:** `~/code/me/fluid-scratch/lane-x/`.
  - `bench-addressing.mts` is the addressing microbench.
  - `bench-addressing.v1.mts` is the same bench without the staged variant S.

## 2026-11-15 STARTED

Base 5763fa6f. Read:
- lane A's attribution and the technique catalogue;
- the seep rewrite plan and lane S's progress;
- the one-pass (2026-11-02) and brick-resident (2026-11-08) plans and their learnings;
- the seep solver's record();
- NB-FLIP's address, arena, bulk, page-pool, fine-slot, texture-page, ownership and layout headers, and the
  frame's advance().

An Explore agent took a feature inventory of the uniform tree: bodies, solids and editing, sources, volume control,
tiers, the band, the level set, extension, presentation, and the gate scene configs.

## 2026-11-15 GO/NO-GO 1: addressing cost (check slot, shared)

Bench: `bench-addressing.mts`, on the seep's box of 64 × 32 × 8 bricks with a 2048-brick active slab:
- 1,048,576 cells and 262,144 particles;
- `adv` is a semi-Lagrangian MAC advect at 96 taps a cell;
- `pmove` is RK3 at 72 taps a particle.

Command, from the fluid directory, with the environment variable inline and never exported:
```
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js tools/gpu/fluid-gpu-run.sh check node --import tsx \
  ../fluid-scratch/lane-x/bench-addressing.mts --reps=40 --variants=D,BT,BTo,T2o,R,H,T
```

Results over four runs (40 and 60 reps, both variant orders):
- `adv` ns a cell, with ×D:

  | variant | ns a cell | ×D |
  |---|---|---|
  | D | 1.11 | 1.00 |
  | BTo | 1.48 | 1.30–1.33 |
  | BT | 1.52 | 1.36–1.40 |
  | T2o | 1.63 | 1.34–1.46 |
  | NB-FLIP's chain T | 1.80 | 1.59–1.62 |
  | R | 1.87 | 1.66–1.68 |
  | H | 2.9 | 2.6 |

- `pmove` is noisy on the shared slot (GPU memory in use was 3.9–5.4 GB at start). D is 3.5–4.1 ns a particle. The
  sparse variants are 1.05–2.0x and T is 1.18–2.26x.

**Reading.**
- NB-FLIP's own address chain is 1.6x dense, while its move is 16–27 ns a particle against a bare RK3 at 4–8 ns.
- So addressing is not the 5x gap. Directory choice: T2o, which is memory ∝ resident pages and has no domain-sized
  table.

## 2026-11-15 GO/NO-GO 1b: the staged brick (variant S)

Variant S stages the brick: a workgroup a brick (128 lanes) loads its brick plus a one-cell ring, 11³ × 3 components
(16 KB), through the 27-neighbour record of a shuffled pool, once. Taps then read workgroup memory, falling back to
the record path outside the ring.

Two quiet check-slot runs (1.1 GB in use at start), 40 reps, orders D,S,R,T2o,T and T,T2o,R,S,D, rc 0 on both:

| variant | adv ns a cell (×D) | move ns a particle (×D) |
|---|---|---|
| D | 1.11–1.12 | 3.54 |
| S | 1.92–1.93 (1.73) | 2.90–2.93 (0.82–0.83) |
| R | 1.85–1.86 (1.66–1.67) | 3.73 (1.05) |
| T2o | 1.60 (1.43) | 3.83 (1.08) |
| T | 1.77–1.78 (1.59) | 4.16–4.17 (1.18) |

**Reading.**
- Staging wins for particle kernels: faster than the seep's dense move. 128 particles × 72 taps share one stage.
- Staging loses for cell kernels, against T2o's direct addressing: workgroup memory costs occupancy, and the cache
  already serves the stencil's reuse.
- Design rule: particle kernels stage, cell kernels address directly through T2o.
- Shuffled slots cost about 1.15x over spatial ones (R against T2o), so the free stacks allocate page-locally.

## 2026-11-15 DESIGN WRITTEN

`docs/plans/nbflip-structural-2026-11-15.md`:
- §1: the structural diff.
- §2: the target, about 132 encoded and 60–75 real dispatches.
- §3: strategy C, a new brick engine behind the method id.
- §4: go/no-go results, and milestones M0–M6 with expected ms.
- §5: lanes E, F and P, with P1 folded into P and P2 into F.
- §6: the end state, 2.5–2.9x on all four scenes (seep 13/12, dam 18/17, letters 31/28, paddleboats 19/17 ms,
  fenced/pipelined).
- §7: decisions for the coordinator.

The decision raised: should the two-sided drift gates (G7, L1, P1, D4) become one-sided?

P2's move data is folded in at §1.4 and §2.4:
- the move is split by register budget into `pTrace`, `pBand`, `pSolid` and the rebin;
- the p1b collapse is an M0 precondition;
- the parked census-split patch is superseded by §2.11's single param ring.
