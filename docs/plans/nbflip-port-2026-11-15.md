# Port the seep rewrite's speed wins into Narrow-band FLIP (2026-11-15)

## 0. Status board

| item | state |
|---|---|
| Phase 0: close out the seep rewrite (S2q gate round, technique catalogue) | in progress |
| Phase 1A: NB-FLIP cost attribution across scenes, ranked port list (lane A) | in progress (2026-11-15) |
| Phase 1G: cross-scene gates and NB-FLIP references, dam break added (lane G) | in progress (2026-11-15) |
| Phase 2: port lanes P1 (pressure) and P2 (submission, dispatch, kernels) | todo |
| Combined cross-scene round | todo |

## 1. Goal and contract

The user chose to port the seep rewrite's improvements into the general method
instead of writing another single-scene solver. The high-resolution dam break joins
as the sloshy acceptance scene.

Success means **NB-FLIP gets substantially faster on every gate scene while its
physics stays the same within measured tolerance**. In detail:
- **Gate scenes:**
  - `reservoir-seep` (1200 steps);
  - `high-resolution-dam-break` (128³, 4096-brick ceiling, 4 particles per fine cell;
    the step count is set by lane G from the scene's 20 s duration and GPU budget);
  - letters (`nbflip-figure-8-letters`, 240 steps; see `tools/benchmark-nbflip-letters.ts`);
  - paddleboats (`docs/plans/paddleboats-60fps-2026-11-10.md`), timing and drift.
- **Physics:** for each scene, the observables of NB-FLIP after the port match NB-FLIP
  before it, within the before/before repeat spread S. NB-FLIP is nondeterministic,
  so every scene's reference runs twice. This is the same method as the seep
  rewrite's §4.
- **Speed:** step time, fenced and pipelined, measured on a quiet GPU by lane G's
  timing round. It is reported per scene against frozen baselines. No single-scene
  win counts: a change that helps one scene and hurts another is reverted, not
  compensated (memory: cross-scene gates, improvements generalize).
- **Unchanged rules:**
  - memory and work stay proportional to the fluid (no domain-sized allocation or
    dispatch);
  - no substeps (dt 1/60);
  - all work on the GPU, with no per-step CPU stages or readbacks;
  - no hacks or clamps;
  - no runtime pool growth;
  - every numerical change cites a paper.
- **Per optimisation:** a fixed-state, one-step equivalence check (bitwise, or a
  stated ulp bound) before any trajectory comparison, wherever the change is meant
  to be exact. Pressure-solver changes that alter the iterate instead pass the same
  residual tolerance, and then the trajectory gates.

## 2. What transfers from the seep rewrite (catalogue)

These are measured in `docs/plans/seep-rewrite/progress-lane-s.md`, S2c–S2q. Lane S
writes the definitive per-technique catalogue as Phase 0. The likely transfers:

1. **Dispatch economics under Dawn.**
   - Every encoded dispatch costs about 3.5 µs.
   - Dawn validates every indirect dispatch with an injected one-workgroup dispatch,
     so an empty indirect slot costs about 50 µs per PCG cycle slot. Command
     translation at submit is CPU time that fenced mode waits on.
   - Levers:
     - fewer dispatches;
     - a convergence check placed before the V-cycle, so the converged iteration
       skips its down sweep;
     - fewer cycle slots;
     - a two-part submit, so the GPU starts while Dawn translates the rest.
   - NB-FLIP issues about 300 dispatches a step, so this applies at full scale.
2. **Pressure preconditioner.**
   - A staggered block Schwarz L0 smoother: the brick partition, then an xz shift by
     half a brick, with exact reverse post-order (Smith, Bjørstad & Gropp 1996). It
     took mean cycles from 5.17 to 3.32 and the maximum from 12 to 7.
   - An exact coarsest solve (Gauss-Jordan once per step) instead of 41 sweeps.
   - A Jacobi warm start (Saad 2003).
   - Block-Jacobi on a coarse level, with restriction fused into pre-smoothing.
   - Symmetry is proven by a CPU Float64 unit test (`seep-smoother.test.ts`).
   - Additive/BPX cycles were measured and rejected: 30–40 iterations against 7–10.
3. **Kernel shape, guided by occupancy and register pressure.**
   - Split register-heavy traces out of merged kernels, compact the work so full SIMD
     groups trace, and use 64-cell slabs per brick instead of 512-thread workgroups on
     lightly loaded bricks.
   - Spread one-thread-per-item loops (for example, interface faces) over 16 lanes.
   - Pack 8 sparse bricks per 64-lane workgroup.
4. **Level-set maintenance.**
   - Skip tiles that are a fixed point: all air, all −PHI_MAX, or no unknown face.
   - Use packed sign-code tiles for cut/frozen tests.
   - Compact true cut cells before the tetrahedral fill.
5. **Particle transfer.**
   - Transposed lanes in P2G, to avoid same-word shared atomics.
   - Flag stores instead of contended atomicAdds.
   - Skip redundant atomicOr.
6. **Physics detail.** Emit at an outlet with the face's own normal velocity and the
   donor cell's tangential velocity. This is relevant wherever NB-FLIP seeds
   particles from grid velocity next to closed faces.

**Seep-only items that do not transfer:**
- dense brick-major storage of the whole box (about 1.05 GB), which violates memory
  scaling;
- baked BULK rows and the throat block;
- particles kept to one chamber;
- the scene-fixed 32×32 coarsest level, which needs a general equivalent.

## 3. Phase 1 (parallel)

### Lane A: attribution and ranked port list
Profile current NB-FLIP on seep, high-resolution dam break, letters and paddleboats.
For each scene, report:
- per-kernel GPU time (timestamp profile) and dispatch counts (direct and indirect);
- empty or zero-workgroup indirect dispatches per step;
- CPU encode and submit time;
- the pressure cycle distribution and cap;
- xctrace occupancy and limiters for the top 10 kernels, using the
  `gpu-frame-profile` skill on a quiet GPU.

Map every cost centre against §2, then produce `docs/plans/nbflip-port/attribution.md`.
It ranks candidate ports by expected ms saved, per scene, with an implementation
sketch and risk for each, and names the NB-FLIP files each port touches. No changes
to `lib/`.

### Lane G: cross-scene gates
Generalise the seep harness into a per-scene gate runner for NB-FLIP before and after:
- **seep:** reuse G1–G8. The candidate is now NB-FLIP itself, so every gate is
  NB-new against NB-reference within the repeat spread.
- **high-resolution dam break:**
  - front position x_f(t) and residual column height h(t) (Martin & Moyce 1952
    form);
  - total volume drift;
  - kinetic and potential energy over time;
  - wet-floor cells;
  - a coarse surface-height profile at fixed times;
  - particle count;
  - step time.
- **letters** (240 steps): drift and step time per the cross-scene table, plus a
  shape observable.
- **paddleboats:** drift and step time.
- Two NB-FLIP reference runs per scene at a frozen baseline commit, giving S.
- Gates of the form |new − ref| ≤ max(rel%, 3·S), written down **before** any port
  lands.
- A timing round script with the quiet-GPU gate, generalised from lane H's
  `timing-round.sh`.

Outputs:
- `tools/nbflip-gates.ts` (or an extension of the seep tools);
- references under `artifacts/nbflip-port/ref/`;
- gate definitions in §4 of this document.

## 4. Gates
Written by lane G before Phase 2 lands anything. Once written, they are never
loosened after the fact. A measurement defect found in the reference, as with
seep §4.1, may be corrected only with evidence, and the original row stays printed.

## 5. Phase 2: port lanes
Two implementation lanes take items from lane A's ranked list:
- **P1, pressure:** the smoother, coarsest solve, check placement and cycle slots,
  in `uniform-pressure-arena.ts` and its relatives. A CPU Float64 symmetry test is
  required.
- **P2, submission, dispatch count and kernel shape:** two-part submit, fewer
  dispatches, register splits, tile skips and particle atomics.

Each landing needs:
- the equivalence check;
- the lane G gates on every scene;
- an interleaved A/B on every scene;
- a commit through the private index guard.

A coordinator-approved round by lane G times the combined result on a quiet GPU.

## 6. Process
- Agents implement; the coordinator plans and judges.
- GPU work only through `tools/gpu/fluid-gpu-run.sh`; rc 76, 77 or 78 means the run is
  invalid.
- Run `npm run check:types` (only the 3 pre-existing lane-n errors) and
  `npm run test:unit` before every commit.
- Use a private index for every commit, then `git reset -q`.
- Each lane writes `docs/plans/nbflip-port/progress-lane-<x>.md`.
