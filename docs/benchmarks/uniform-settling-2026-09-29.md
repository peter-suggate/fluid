# Uniform Geometric: why minidam64 keeps moving

Investigation on 29 September 2026, Dawn/Metal, `minimal-power-dam-break-64`,
Uniform Geometric defaults. No production solver or default was changed.
Existing working-tree performance edits were preserved; base commit
`6cfb9b776a49fa9a07e7c8cb971a19b04697e347`.

**The recurring large speed peaks are predominantly persistent sloshing, not
proof of fresh energy being injected each time.** There are also two real
numerical problems: extreme local compression in volume transport, followed by
pressure-driven volume repair, and a loose default pressure target that makes
initially still water move. Increasing pressure work alone does not settle the
dam break.

![Settling measurements](uniform-settling-2026-09-29.png)

## What the default run does

The 0.8 m tank has free-slip walls, viscosity 0.001002 Pa s, density
998.2 kg/m³, zero surface tension, no inflow and no bodies. Its physical
viscosity is very weak; the wall condition adds no no-slip shear damping.
The solver advances by 1/30 s on a 12.5 mm finest lattice.

Over seconds 10–20, the volume-weighted velocity RMS proxy ranges from
0.148 to 0.719 m/s and averages 0.471 m/s. Its kinetic peaks recur about every
0.6 s. Kinetic and gravitational potential energy proxies are strongly
anticorrelated (−0.983 over seconds 5–20): water speeds up as its centre of mass
falls, then slows as it climbs again.

The mean total-energy proxy per mass is 1.708, 1.717 and 1.713 m²/s² over
seconds 5–10, 10–15 and 15–20. This is essentially a persistent oscillation
rather than an escalating total-energy envelope. It does **not** prove the
numerical scheme is energy conserving: numerical forcing could offset
numerical dissipation, and this diagnostic is not an exact energy budget.

A matched control switches gravity, viscosity and surface tension off at
step 301. During the next 60 steps, projection never increases the measured
kinetic proxy relative to its same-step input. The proxy falls from 23,005 to
5,498 (unscaled owner-volume units). Thus these observations do not establish
a recurring net pressure-energy injection. Local pressure impulses and local
jets can still occur while the global kinetic measure decreases.

## Numerical compression and repair are substantial

The detailed repeat is identical to the baseline at all 202 shared samples.
It captures transport output before cleanup/sharpening and the fields before
scratch reuse.

At t=0.6 s, cell (55,38,53) holds **137.525 cell-volumes**. The value is already
present immediately after transport and remains identical after sharpening.
Its centre phi is negative, so this example is not merely stale air storage.
Even after ten seconds, sampled maximum V reaches 64.29; aggregate excess
above capacity averages 2.02% of the water over seconds 10–20.

The relevant mechanism is:

1. `UniformMixedTransportStage.encodeTransport` performs three row/donor
   balancing rounds, then `gatherAt` applies final donor normalization.
   This conserves donor mass, but does not impose a final receiver-capacity
   bound. The measured overfill is created here. Whether more balancing or a
   better departure map is the primary remedy remains untested.
2. `uvVolumeCorrectionAmountAt` requests removal of half the excess per 1/30 s,
   capped at one cell-volume per step. The pressure authority divides this
   by dt and balances expansion with contraction in underfilled surface cells.
   At the default dt the requested divergence repeatedly saturates at 30 s⁻¹.
   These are numerical volume-repair motions, even when the pressure system
   is solved accurately.
3. Phi and V also disagree: at t=12 s the largest V is 21.15 in an owner whose
   centre phi is +15.61h. Pressure-liquid membership follows phi's sign.
   Detached mass therefore does not automatically become a pressure row.
   This is another reason a small pressure residual is not a bounded-volume
   or settling guarantee.

Turning volume correction off at ten seconds lowers subsequent mean RMS,
from 0.471 to 0.315 m/s, but the final excess rises to **29.6% of total water**.
This is a diagnostic disruption, not a usable fix or a clean energy comparison.
Disabling the global surface shift instead gives mean RMS 0.425 m/s and still
substantial overfill; it is not sufficient either.

Reducing dt to 1/120 s lowers mean excess during seconds 5–10 from 1.96% to
0.66%, and mean RMS from 0.468 to 0.366 m/s. Peak V in that window falls from
51.49 to 8.95. The initial impact still reaches V=59.03. Smaller steps also
quadruple resampling and sharpening frequency, so this is supporting evidence
for timestep sensitivity, not an isolated transport proof or a recommended
change to the paper-calibrated defaults.

## Pressure accuracy is a separate failure at rest

The same tank initialized with a flat, still fill develops motion under the
default coarse pressure target of 5 s⁻¹. At ten seconds:

| Control | Velocity RMS proxy | Peak wet-owner speed |
|---|---:|---:|
| Defaults | 0.076455 m/s | 0.205913 m/s |
| Coarse pressure target 0.01 | 0.000107 m/s | 0.000885 m/s |

The default resets pressure to zero and usually accepts one coarse cycle.
Tightening the target uses three cycles on the final rest frame. This directly
establishes pressure-driven parasitic motion in this fixture.

It does not explain away the large dam oscillation:

| Dam control | Mean RMS, seconds 10–20 | Mean fine-band residual |
|---|---:|---:|
| Defaults | 0.4706 m/s | 0.2718 s⁻¹ |
| Coarse target 0.01 | 0.4673 m/s | 0.1858 s⁻¹ |
| 12 fine-band cycles instead of 4 | 0.4565 m/s | 0.0005 s⁻¹ |
| Full h simulation ownership | 0.4261 m/s | 0.8053 s⁻¹ |

Full h simulation ownership still uses the current coarse-plus-fine-band
pressure architecture; it is not an independent monolithic fine-pressure
reference. The larger-band-cycle arm is an explicit diagnostic override;
production budgets and timing ceilings remain unchanged.

A related acceptance gap: the fine-band target is written as zero, so it runs
its fixed four cycles. Its residual is reported, but `UniformMixedFrame.check`
checks coarse convergence and band topology/capacity failures, not fine-band
residual convergence. This deserves correction independently; the 12-cycle
control shows it is not the main explanation for sustained dam motion.

## Where to focus a fix

The subsequent [local recovery experiment](uniform-local-recovery-2026-09-29.md)
tested gentler repair in nominally quiet regions and rejected it: reduced
motion came with accumulating excess, and the still-pool result worsened.
The [updated recovery plan](../plans/uniform-local-volume-recovery.md) records
that decision and a frozen-transport capacity diagnostic to separate
balancing error from departure-map limitations. Repair-driven regeneration
remains a hypothesis rather than an isolated cause.

- Add a sustained still-water regression using actual defaults. Improve the
  coarse stopping policy so repeated gravity projection does not drive rest
  currents, while preserving explicit pressure failure handling.
- Diagnose receiver-capacity errors in the conservative transport balance and
  departure mapping. Preserve mass; clipping V would conceal the problem.
  Any experiment with more balancing rounds must also extend the compact
  live-set dependency closure, which currently assumes three rounds.
- Measure exact MAC-face kinetic energy and per-stage pressure/source work
  before claiming an energy-injection fix. Track overfill, detached mass and
  the sloshing envelope alongside residuals. A pressure-only test misses them.
- Treat faster visual settling as a separate physical/damping choice. The
  current nearly inviscid free-slip scene should not be expected to stop
  quickly merely because pressure converges.

## Evidence and reproduction

[Trace and configuration receipts](uniform-settling-2026-09-29.json),
[diagnostic probe](../../tools/probe-uniform-settling-dawn.ts).
The probe acquires the repository GPU lease and awaits each frame. Captures
are diagnostic only. Velocity energy uses owner V and area-averaged positive
MAC faces; it omits negative boundary slabs and is not the variational kinetic
energy. Potential energy uses owner-centre heights. Do not interpret its
original-gravity height measure as physical potential energy in the unforced
control after its switch.

```sh
node --import tsx tools/probe-uniform-settling-dawn.ts --frames=600 --out=/tmp/settling.json
node --import tsx tools/probe-uniform-settling-dawn.ts --frames=600 --values='{"pressureResidualTolerance":0.01}' --out=/tmp/settling-tight.json
node --import tsx tools/probe-uniform-settling-dawn.ts --frames=600 --band-cycles=12 --out=/tmp/settling-band12.json
node --import tsx tools/probe-uniform-settling-dawn.ts --initial=rest --frames=300 --out=/tmp/settling-rest.json
node --import tsx tools/probe-uniform-settling-dawn.ts --frames=600 --control=no-correction-late --switch-frame=301 --out=/tmp/settling-no-correction.json
node --import tsx tools/probe-uniform-settling-dawn.ts --frames=1200 --every=12 --dt=0.008333333333333333 --values='{"timeStep":"scene"}' --out=/tmp/settling-small-dt.json
```

All diagnostic runs completed without uncaptured GPU errors. Type checking
passed. The unit suite had 1,526 passes, 88 skips and two failures in
`anchored-flyout-placement.test.ts` (vertical-origin and height-cap assertions),
outside this investigation's changes. The full Dawn suite was not run; this
is not a clean-repository gate claim.
