# Lane A handoff: seep fidelity (2026-10-25)

Previous owner: Claude lane A (W0, absorbing W5 and W8). Everything below is merged to main (the lane's tip 59ce5a32 landed as 6992ac44). Paths under the previous owner's scratchpad (`w0/…`) may be gone. The probe readers are now in `tools/gpu/probe-readers/`, the runner is `tools/gpu/fluid-gpu-run.sh`, and the gate bundle is `tools/gpu/gate-bundle.sh`.

## Branch and tip

- Branch: `worktree-agent-abc5be47e02ee923f`
- Worktree: `/Users/petersuggate/code/me/fluid/.claude/worktrees/agent-abc5be47e02ee923f`
- Tip: `59ce5a32`. This merges main `9785c8fe`, which contains `88eebb1f` (lane B's arena pool split).
- Checks at the tip: `npm run check:types` is clean apart from the stray `app/app/*` files, which are not ours. `npm run test:unit` passes 1213/1213.

Commits since the package landed (main `78c729be`):

| SHA | What | Default | Gated |
|---|---|---|---|
| `c25404bb` | crowdedCells "merge": a deep cell over its cap folds the excess into its kept samples as mass (up to 8). Merged samples split again in shallow or under-filled cells. Excess the gatherer refuses stays live. | added as retire | yes |
| `70905acf` | crowdedCells default flipped to merge, with the near-settled fallback | **merge** | yes; on main as `b0bce708` |
| `ca2d251b` | coldSurface "still": a free surface may stay COLD while its omitted transport is within f32 hydrostatic round-off (k=256) a step and 0.01 cell summed. Region state grows 18 to 19 words. | opt-in, default stays warm | gated as a default; failed on letters (see below) |
| `e70d0a3a` | still wake fix and its proof conditions. (1) rhSpread neighbour rule: a still surface beside any non-COLD region is WARM. (2) The round-off bound scales with the liquid's extent along gravity at the last spread, instead of the domain height. (3) Halo reach bound: if the fastest liquid travels 16 h (one region) or more a step, every still surface warms that frame, and the frame is counted. The probe's heatState stride now follows UNIFORM_REGION_HEAT_STATE_WORDS. | opt-in (coldSurface still only) | **UNGATED**. Only check-mode probes were run. No gate bundle or paired timing. |

Gate evidence (logs were under the previous owner's scratchpad `w0/`):

- **`mgates/`**: tip `e37ebb3d` against main `759f912d` passes splash, seep and letters, so the retire default was unchanged. The merge-default snapshot fails on particle count only for splash and seep, and on tiles in one window (frames 51-60) for letters. The paired cost of merge against retire is +0.46 ms on seep and +0.55 ms on letters.
- **`fgates/`**: A/A within the merge default (`70905acf`) passes everywhere: splash worst 0.10, seep 0.40, letters 0.36.
- **`sgates/`**: still-default (`ca2d251b`, without the wake fix) against main `759f912d`.
  - Splash passes, with cost unchanged.
  - Seep passes.
  - Rest saves 6.20 ms (9.40 to 3.13).
  - Letters fails: bandBricks over in 3/6 windows, particles in 2/6, tiles in 1/6, at ratios 3.4-6. Cost on letters is neutral.

## Open problems

### 1. coldSurface still: flip blocked on letters (wake lag)

- **Where the letters divergence starts.** The probe is `wake*-still.log` against `wake-warm.log` (every frame to 36).
  - Frames 0-30 are identical apart from the region tiers: under still about 230 more regions are COLD (at frame 30: 388 vs 159).
  - First field divergence is at frame 31: 1004 vs 964 h tiles.
  - Particles follow at frame 32: 20.8k vs 9.8k.
  - **This is not particle "activation".** No particle exists in either arm before frame 32, and NB_ACTIVITY_NEW is census-driven particle heat, not region heat.
- **What did not fix it:**
  - **Neighbour rule (`e70d0a3a`).** It warms about 110 more regions before the event (174 vs 64 WARM at frame 28), but frames 31/32 still diverge the same way.
  - **Head-scaled bound.** On letters the liquid's extent is 8-9 regions, because the airborne letters count, so the bound barely tightens. Some letters regions sit at about 7e-4 h/step of omission, just under the domain-height bound (7.5e-4), and stay COLD. That omission is very likely real flux, not round-off.
- **Best current understanding.**
  - Regions that warm forces every frame are COLD under still at the frame-30 event (letters materialising; 197 regions are HOT from frame 28 in both arms).
  - Whatever differs in the census or transport across the COLD/WARM boundary at that event produces the extra h tiles and seeds.
- **Next steps:**
  - (a) Find the extra frame-31 h tiles by tile and see which regions they sit in. The heatState listing is a subset, so add a tile-level diff.
  - (b) Use a head that counts only liquid connected to the region's column, not airborne liquid, or the region's own pressure. Then the 7e-4 regions warm.
  - (c) Gate again. The flip is worth 6.2 ms on rest.
- **Halo reach bound.** Implemented in `e70d0a3a`. The tiers words after the active list, `UNIFORM_REGION_HEAT_STILL_WORD`, hold:
  - +0: the liquid's extent along gravity, in regions;
  - +3: this frame's maximum travel, in h per step;
  - +4: the number of breach frames.
  - The probe prints these as `still:{extentRegions,travel_h,breachFrames}`.
  - Measured: letters travel at most about 4 h/step and seep about 13 h/step at 5 mm (3.97 m/s). Both are under 16, with 0 breaches. Seep is the closest to the bound.
  - Travel uses the frame dt and region velocity extremes, which is conservative.
  - A breach warms every still surface rather than widening the halo. That is safe but costs the saving on that frame.

### 2. crowdedCells merge: mergedMassRetired / overCap

- **Fallback retirements past mass 8: 0 by construction.** An excess sample that would push its gatherer past 8 stays live. The excess lane repeats the gatherer's walk (`nbMergeAbsorbed`) to decide this.
  - This is counted as overCap: 156k sample-steps on 300 seep steps and 624k on 200 letters steps.
  - It means some deep cells hold more than 8 times the cap. My reading is FLIP clustering at stagnation points such as the right floor jet impact. Nothing is deleted for it.
  - Open: whether to spread clustered mass, for example by allowing a split into a neighbouring under-filled cell even when deep.
- **mergedMassRetired.** This is the mass past 1 that the existing deep and held-heat retirements take from merged samples: 4.3k on seep and 1.13M on letters (deep 1.27M, heat 1.0M retirements).
  - It is the same NB-FLIP band rule that retires mass-1 samples in retire mode, and the volume is phi-based.
  - Open: whether a merged sample crossing the band depth should split rather than retire with its whole mass, which would cut how much mass merged samples lose there.
- **Results.** Seep at 300 steps, merge against cap-20 retire:
  - right chamber 1.28 L vs 0.25 L;
  - drift -0.022% vs -0.279%;
  - particles 110k vs 18.8k.
  - On letters drift is the same (-1.508%).

### 3. Milestone A status (5 mm seep, 20 s): NOT yet demonstrated

- **No 1200-step (20 s) run exists** on the current defaults: package plus merge plus the 4-cycle band.
- **300-step merge run (`seep5-merge2.log`):**
  - no fatal;
  - cdInstant 0.81 (throat u 2.86 m/s, against Torricelli 3.52 m/s at head 0.631 m);
  - cdRight 0.54 to 0.66 late, so the right chamber receives about 80% of the throat flux;
  - max speed 3.79-3.97 m/s, so the fast-jet throat stays near 3.96;
  - right chamber 1.28 L and filling.
- **Still to show:** the left level's fall against Torricelli at the measured Cd within 10% over 20 s, and the right chamber continuing to fill with no fatal. Older 1200-step runs (`seep5-r1-1200`) predate the package and are not comparable.

## Probes (tools/probe-tier-budget.ts)

Run every GPU job through the shared lock, from the worktree:

```
tools/gpu/fluid-gpu-run.sh check env WEBGPU_NODE_MODULE=/Users/petersuggate/code/me/fluid/node_modules/webgpu/index.js /Users/petersuggate/.nvm/versions/node/v22.22.1/bin/node --import tsx tools/probe-tier-budget.ts NAME [options] > LOG 2>&1
```

- `--scene=ID` (default reservoir-seep), `--region=rest|cold|warm|splash|hot` (the region-tier scenes, quality balanced), `--steps=N`, `--every=N`, `--cell=`, `--pool=BLOCKS`, `--set=key:value,...` (method params).
- `--retire`: retirement counts by reason (diagnostic words at slab state 32-47).
- `--retireSplit=X`: as `--retire`, with crowded / all / seeds offered counted at cells x >= X. Use 256 for the seep's right chamber and 0 for everything.
  - The merge counters print as `merged, overCap, splits, mergedMassRetired`.
- `--heatState`: every surface-reader region's freeze evidence, as `surf/mom/pres/froz/disp/trig/tier`.
  - Summarise with `tools/gpu/probe-readers/flux-print.py LOG`. Compare two runs with `tools/gpu/probe-readers/wake-print.py A B` and `tools/gpu/probe-readers/wake-diff.py A B STEP...`.
- `--flowStages=s1,s2`: per-stage velocity maxima and zone volumes (left/tube/right) for those steps. `--phiStages=` does the same for the phi stages.
- Band (arena) header: printed every sample as `{"bandHeader":...}`, with tiles, cycles, initial and final residual, cycle history, and native initial and final.
- Every row also carries `still:{extentRegions,travel_h,breachFrames}` (coldSurface still words) and the discharge block (`level_m, head_m, uThroat, qRight, qTorricelli, cdRight, cdInstant`).
- Gates:
  - `tools/gate-repeat-spread.ts NAME --baseline=SNAP --candidate=SNAP --gate="tools/gpu/fluid-gpu-run.sh MODE" [--differ=keys] [--set=]`. Use mode gate for splash and timing for seep and letters.
  - Cost: `tools/benchmark-region-tier-paired.ts` with `--set` and `--set-candidate`.
  - Snapshots and the three-scene bundle: `tools/gpu/gate-bundle.sh NAME BASE_REV CAND_REV [--differ=…] [--set=…]` builds `git archive` snapshots itself.

## Seep milestone run (exact commands)

On main, with defaults (merge on, coldSurface warm), 20 s at 60 Hz = 1200 steps:

```
cd <your worktree at main>
tools/gpu/fluid-gpu-run.sh check env WEBGPU_NODE_MODULE=/Users/petersuggate/code/me/fluid/node_modules/webgpu/index.js /Users/petersuggate/.nvm/versions/node/v22.22.1/bin/node --import tsx tools/probe-tier-budget.ts seep5-milestone --steps=1200 --every=60 --retireSplit=256 --out=<dir>/art > <dir>/seep5-milestone.log 2>&1
```

- About 30 ms a step, so 10-15 min. It needs about 3.5 GiB and runs in check mode.
- Read per sample:
  - `discharge.level_m` over t, against Torricelli at the measured `cdInstant`;
  - `liquid_m3.right` rising;
  - `volume.maxSpeed_m_s` near 3.96;
  - `pool.fatal` absent;
  - `retired.overCap` and `mergedMassRetired`.
- For timing, rerun in `timing` mode, exclusive. ms from check mode don't count.
