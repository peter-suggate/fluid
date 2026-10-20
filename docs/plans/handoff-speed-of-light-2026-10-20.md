# Handoff: NB-FLIP pour at speed of light

## The scene
- **Target:** `nbflip-figure-7-pour` at 1/60 s on the M1 Max, under 17 ms a frame. Every change is live in the UI and on by default.
- **Focus: the late frames, the churn case** (pour f200–212):
  - about 1.2M particles and about 10k fine tiles (read the count from the GPU; `workDomainInfo` goes stale);
  - 54.9 ms of GPU pass time plus 7.1 ms of idle.

  | Stage | ms |
  |---|---|
  | Surface | 16.8 |
  | Pressure | 10.3 |
  | Particles | 10.2 |
  | RFE | 10.0 |
  | Other | 6.4 |
  | Momentum | 1.3 |

- **Physics gate:** the letters A/B (`nbflip-figure-8-letters`), two repeats per arm. Particle counts, volume drift and occupancy must stay within repeat noise.

## The approach: speed of light, worked backwards
1. **Define the ideal first.** For each stage, write down the irreducible job at late-frame counts: which elements the physics actually needs, and what each one must read and write.
2. **Price it at measured hardware rates** (`m1-max-measured-ceilings`):

   | What | Rate |
   |---|---|
   | Streaming | 360 GB/s |
   | Gathers | 34 / 15.5 / 6.9 G/s (within 4 KB / over 4 MB / over 32 MB) |
   | Texture filtering | 28.4 G samples/s (rgba16f), 13.5 (rgba32f) |
   | Workgroup memory | 719 G/s |
   | Atomics | 218 G/s workgroup, 5.7 G/s device |
   | Per pass | 2.3 µs |

   The result is the bound.
3. **Work backwards to the algorithm shape:**
   - the minimum work that produces the answer, with no more elements than the physics needs;
   - change-driven where inputs persist;
   - one trace step, no substeps.
4. **Then the data shape:** the layout that puts every access on its fastest path.
   - compact lists of only the needed elements;
   - hardware-filtered textures for interpolated sampling;
   - per-tile context resolved once, not per element;
   - accumulators produced in the pass that already touches the data.
5. **Apply that shape across stages.**
   - Classify once per step, build each list once, and let every stage read it.
   - Don't re-derive ownership or geometry in each kernel.
6. **Verify before projecting.** Read the kernel, run knock-outs at the fixed late state, then make one bold change. Measure absolute pass times at f200–212, run the letters A/B, commit.
7. **Reassess priors whenever a result is more than 2× off its bound.**
   - Name the assumption that broke: tap rate, element count, layout, or the algorithm.
   - Try a different approach. Never accept the gap as "near today's per-element rate", and never retune the same idea.

## What produced the big wins
- **Lean split kernels over compact lists instead of fat merged kernels.** Metal inlines everything, so fat kernels lose occupancy.
  - Surface trace: 3.7 → 0.83 ms.
  - Momentum as classify plus a listed trace: 3.4 → 1.1 ms.
  - RFE: moving about 3 serial "plane corner" jobs into their own launch takes remapCells 1.63 → 0.20 (knock-out).
- **Exact lists of only the needed elements.** The coarse-corner list cut the coarse surface passes from 1.01 to 0.40 ms.
- **Texture-unit sampling.** Particle trace: 1.48 → 0.90 ms. Surface trace: 0.86 → 0.52 ms together with one step and no substeps.
- **Knock-outs before projections.** They overturned every stale assumption: the splat was already workgroup-atomic, coupleFine doesn't gather particles, and R checks at most 8 candidates.

## What failed, so don't repeat it
- **Workgroup window staging on the thin h band.** About 103 of every 125 window tiles are 4h, and the staged sAdvect lost: 9.6 ms against 6.0.
- **Fat per-brick kernels.**
- **A per-cell shared velocity channel.** At 0.19 particles per cell, filling it costs as much as the per-particle path it replaced.
- **Projections from notes rather than code.**
- **Reading a knock-out's delta as memory latency.** Removing code also frees registers and raises occupancy. In the particle move, the knocked-out lookups promised −1.0 ns/p. Serving 99% of those lookups from a staged window saved only 0.23. Confirm with counters, and treat a large knock-out delta as a sign the kernel is fat, which calls for a lean split kernel.
- **Retiring the root solve (55c16a3a).** The band's coarse correction through the shared tail stalls on letters: residual p90 0.56 against 0.11 before. Recommended: `git revert` it, then re-land it after a band-twin analysis.

## Rules
- No indirect dispatch; zero per-frame allocation; no compile during simulation; allocate upfront and fail fast.
- One call site per heavy WGSL function.
- No step limits and no trace substeps.
- Physics changes go behind NB-FLIP params (update: "solver").
- GPU runs go under `lockf /private/tmp/claude-501/fluid-gpu.lock`.
- The dev server serves the main tree. Diagnostics go in `git archive` trees; switch-overs are short and committed straight after.
- Never stash, checkout or reset. Commit with `git commit -- <paths>`, or `git apply --cached` when a file has peer hunks.

## State
- **Docs:**
  - `speed-of-light-plan-2026-10-20.md`: umbrella and late-frame table;
  - `brick-arena-core-2026-10-20.md` §5–6: surface;
  - `pressure-rfe-speed-of-light-2026-10-20.md` §1.8 and §2.5: pressure and RFE;
  - `particle-stage-gap-2026-10-20.md`: particles.
- **Late-frame baseline:** `scratchpad/P/late1.json`; probe at `scratchpad/wt-late/probe-late.mts`.
- **Paused WIP** (scratchpad patches):
  - surface: `surface-trace-onestep-wip.patch`; delete `scratchpad/PAUSE` before resuming;
  - pressure: `wip-tail-cycles.patch`;
  - particles: the move brick-context work;
  - RFE: item 1, plane corners.
- **Next levers by late-frame ms:**

  | Stage | Lever |
  |---|---|
  | Surface | One-step trace (needs its letters result); per-tile cell table for coupleFine; per-brick setup in arenaDistance |
  | Particles | Move brick context (3.33 ms pass); seed hysteresis (churn) |
  | RFE | Items 1–4 of §2.5; a new structure for body forces and extension |
  | Pressure | Revert decision; pRows rebuilt for dirty tiles only; authority once per step |
  | Other | Ownership census (2.2) and local speed (1.25), unowned |
