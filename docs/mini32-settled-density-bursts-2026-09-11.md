# Mini32 min/max1 settled density bursts

## Finding

Reproduced on the current working tree with the registered/UI mini32 scene,
whole-domain minimum/maximum cell size 1, balanced defaults and paper timestep
1/30 s. Native Dawn/Metal runs cover 330 steps (11 simulated seconds), with
per-frame accepted-field captures from step 240. No browser ran alongside Dawn.

The wall/corner bursts are a feedback loop involving gamma diffusion, false-air
pressure classification, and conservative transport. A concrete boundary error
in `pressureCellSubmerged` defeats the existing protection against that loop:
`if(count<2u){return false;}` treats an impermeable one-sided wall row as if it
were an open sparse-air port. This explains why the problem favours walls and
corners even with every active cell at the finest resolution.

This is a diagnosis, not a production fix. The diagnostic guard experiment also
shows substantial underlying density depletion; suppressing the bursts alone
would not establish correct settled behaviour.

## Work backwards from one event

Coordinates below are zero-based finest cells, not brick coordinates. Corner
(0,0,31) borders the floor and two walls. Its adjacent interior cell is (1,1,30).

| Step / time | Observation |
| --- | --- |
| 296 / 9.867 s | Corner density 0.572607, pressure 3110.48, speed about 0.006 m/s. |
| 297 / 9.900 s, transport | Corner density 0.574768. |
| 297, gamma diffusion | Corner density falls to 0.475327, crossing the 0.5 pressure-membership threshold. |
| 297, end of step | Corner pressure is exactly zero. Velocity becomes (-1.1035,-1.0871,+1.0896) m/s: approximately 1.894 m/s toward the corner. |
| 298 / 9.933 s, transport | Corner density jumps to 4.33230. |
| 298, diffusion | Corner density falls to 2.4040. |
| 298, sharpening | No change at this corner. |
| 298, capacity repair | Corner density becomes 1.09297; adjacent interior cell reaches 1.20750. |
| 298, publication | Preserves the repaired scalar field. |

At steps 299 and 300 the analogous transport spikes are 4.7177 at (31,0,31)
and 4.4176 at (31,0,0). Repair leaves visible interior peaks of 1.25429 and
1.20563. The late trajectory repeats approximately every four steps.

Stage copies reproduce the baseline trajectory exactly at steps 297–304,
including the pressure residuals. Thus this is present in accepted simulation
fields and is not caused by the renderer or by diagnostic stage copies.

## Causal controls

**Disable diffusion immediately before step 297**, after an identical 296-step
history. Corner density stays at 0.574768, pressure stays at 3165.93, and speed
stays below 0.01 m/s. At step 298 density is 0.572725 rather than 1.09297, with
no transport pile-up. All four bottom corner speeds remain below 0.038 m/s
through step 304. This isolates the threshold-crossing trigger; it is not a
recommendation to turn off diffusion in production.

**Exempt closed one-sided rows in the submerged guard**, from reset, preserving
open one-sided air ports. Across steps 240–330 the maximum bottom-corner speed
falls from 2.425 to 0.211 m/s and maximum corner density from 1.151 to 0.567.
However, minimum corner density falls to 0.0567, many corners remain at zero
pressure, and other motion persists. The guard requires previous membership
and liquid neighbours, so it cannot generally recover an already depleted
connected region. This experiment does not validate a complete correction.

There is an additional feedback mechanism in the captured scalar state:
transport raises corner gamma from 17.5095 to 74.0179 at step 298; diffusion
reduces it to 41.0729, while capacity repair subsequently redistributes density
without redistributing gamma. That leaves a high-gamma, density-depleted corner
which continues donating density through gamma diffusion. The observations
establish this loop; they do not establish the earliest origin of the gamma
inhomogeneity or prove that changing gamma transport is the right remedy.

## Relevant implementation

- `lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts`:
  `pressureCellSubmerged`, `pressureCellMembershipFromDensity`,
  `classifyPressureCell`: wall exclusion and zero-pressure hole.
- Same file: `scatterGammaRow`, `finalizeGammaCell`: density/gamma diffusion.
- Same file: `gatherConservativeDensity`: next-frame concentration.
- Same file: `finalizeDensityCapacityRepair`: redistributes density while
  gamma retains the preceding diffusion result.

A production repair needs to distinguish impermeable walls from genuine air
and separating boundaries, handle depleted submerged regions rather than
merely their previous membership, and retain correct free-surface/ceiling
release. It also needs to address or account for the persistent density/gamma
mismatch. Do not accept a change solely because the yellow bursts disappear.

## Reproduction and evidence

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
  node --import tsx tools/probe-mini32-settled-bursts-dawn.ts

WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
  BURST_STEPS=304 BURST_FROM=296 \
  BURST_AUDIT_STEPS=297,298,299,300,301,302,303,304 \
  BURST_OUTPUT=artifacts/mini32-settled-bursts/audit \
  node --import tsx tools/probe-mini32-settled-bursts-dawn.ts

WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
  BURST_STEPS=304 BURST_FROM=296 BURST_SWITCH_STEP=297 \
  BURST_SWITCH_OVERRIDES='{"gammaDiffusion":"off"}' \
  BURST_OUTPUT=artifacts/mini32-settled-bursts/no-diffusion-late \
  node --import tsx tools/probe-mini32-settled-bursts-dawn.ts

WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
  BURST_WALL_GUARD=1 \
  BURST_OUTPUT=artifacts/mini32-settled-bursts/closed-wall-guard \
  node --import tsx tools/probe-mini32-settled-bursts-dawn.ts

python3 tools/analyze-mini32-settled-bursts.py
```

The probe acquires the repository GPU lease. Each arm saves configuration,
accepted density/gamma/velocity/pressure/divergence, and scalar stage captures
when requested. Summary: `artifacts/mini32-settled-bursts/summary.json`.
Pressure values above are raw diagnostic solver values, not converted to Pa.

The investigation above initially added only diagnostic tools and this report.

## Follow-up: isolated guard correction rolled back

The production correction exempted closed one-term rows while retaining open
ports and separating contact. Its 12-case GPU membership test passed. The user
then observed visible retreat of liquid from the corners, consistent with the
large corner-density depletion already found in the diagnostic ablation.
The isolated production change was therefore reverted. Production retains the
original guard until a correction also maintains appropriate corner density.

Keeping a depleted cell in the pressure solve removes the pressure-driven
refill event, but does not stop gamma diffusion draining density. Therefore
correcting the wall/air distinction alone suppresses one symptom while allowing
another visible failure. Future validation must check both corner depletion
and density/velocity bursts, not merely the pressure-membership predicate.

The targeted 12-case production-WGSL fixture is retained as the explicit
`tools/probe-sparse-cm12-submerged-wall-dawn.ts` diagnostic, outside automatic
test discovery. Against the restored guard it intentionally exposes the wall
membership error. It is not a passing regression in the restored implementation.

The full Dawn gate with the proposed correction passed 14/17 lanes. The
remaining failures were topology-page-budget timeout and both performance
ceilings. The ceiling-slab test also failed its refinement assertion. All four
failures reproduced with the original guard in the same checkout; no timing
threshold or baseline was changed. These results do not excuse the visible
retreat that caused rollback.

`BURST_LEGACY_WALL_GUARD=1` explicitly selects the original guard;
`BURST_WALL_GUARD=1` selects the earlier closed-wall diagnostic ablation.
Ordinary probe runs now use the restored production guard.
