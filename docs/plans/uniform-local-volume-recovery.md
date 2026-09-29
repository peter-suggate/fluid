# Local volume recovery for Uniform Geometric

Status: steps 1–2 tested 29 September 2026 and **not adopted**; see the
[experiment report](../benchmarks/uniform-local-recovery-2026-09-29.md).
Fresh excess is produced every step in nearly every tile that holds excess
(about 20% of the standing excess per step), so it cannot identify quiet
liquid. A strain-confirmed quiet rate preserves the impact response but more
than doubles standing excess, both in the dam and in a still pool. The
tile-activity implementation is kept as an unapplied patch beside the report.

## Decision after steps 1–2

Do not adopt the slower quiet-repair policy. Its premise was that some
persistent excess could be repaired more slowly because little new excess
was arriving. The experiment instead finds sustained excess generation, and
slower repair increases the stored error. The strain gate preserves wall
climb but does not satisfy the volume-quality acceptance rule. In the still
pool it worsens both excess and motion.

Do not proceed to step 3 as a way to rescue this policy. Weighting the global
phi shift cannot substitute for removing local V excess. Keep the current
repair response while investigating the source of excess; the activity patch
remains an experimental artifact, not a candidate for integration.

The proposed self-feeding repair/transport loop remains a hypothesis. Similar
fresh-generation rates across repair settings do not isolate its contribution.
Likewise, preserving baseline wall climb is an important regression check,
but does not independently establish that repair-driven run-up is physically
accurate.

## Replacement investigation: isolate the transport operator

The next local diagnostic should measure receiver-capacity error directly,
rather than try to classify quiet liquid from the excess it produces.

Freeze representative frames' ownership, geometry, incoming V and departure
map, then replay balancing and gathering without advancing the scene. Let
\(m_i\) denote extensive liquid volume, \(C_i\) extensive open capacity, and
\(B\) the resulting nonnegative transport matrix, so \(m'=Bm\). For static
geometry without sources, two distinct properties matter:

- Donor conservation: the columns of \(B\) sum to one.
- Capacity preservation: \(BC=C\).

The local defect \(d_i=(BC)_i-C_i\) identifies capacity error in the completed
transport operator. If \(0\leq m\leq C\), positivity gives \(Bm\leq BC\);
preserving capacity therefore prevents new overfill from initially bounded
input. With inherited overfill, the same frozen operator permits separate
gathers of bounded input and excess to attribute their contributions.

Compare the current three balancing rounds with additional rounds, measuring
both donor conservation and receiver defects after the final donor
normalization. Use full support or extend the compact dependency closure for
each round count: its present three-round closure is not sufficient evidence
for a longer balancing experiment. Include h/4h volume weights, open capacities
and all relevant donor/receiver support.

If the defect falls substantially with more rounds, investigate balancing
convergence and a useful residual-based stopping rule. If it stalls, inspect
the departure-induced overlap graph: additional iterations cannot repair
missing transport connections or incompatible capacity constraints. Test
alternative departure maps on the same frozen inputs before returning to
coupled scene runs.

Separately retain the tight-coarse-pressure still-pool control to establish
how much excess generation is driven by pressure error. Do not change repair,
transport and pressure simultaneously. Any candidate must reduce fresh and
standing excess while preserving mass, impact quality and the existing
performance gates.

**Result (29 September 2026, [operator census](../benchmarks/uniform-transport-operator-2026-09-29.md)).**
The defect stalls: 16 rounds cut it by only 21–30%. It correlates with the
departure map's own compression (r 0.6–0.77), scales linearly with travel,
and barely depends on thin donor coverage, walls or seams. Removing local
repair makes the traced field nearly divergence-free in deep liquid (RMS
0.036 dt/h) but leaves the map's compression (0.105) and the fresh overfill
unchanged. The next candidate is a departure map whose box volume follows the
MAC divergence (face-traced boxes), tested on the same frozen inputs.

The remaining sections preserve the original proposal and its acceptance
criteria for reference; they are not the current implementation plan.

## Objective

Preserve the aggressive response to transport overfill during impacts, such
as a dam front climbing a wall, while making recovery gentler in quiet liquid.
The decision should be local: an active splash and a settling pool may coexist
in the same scene.

The proposed policy uses **locally renewed compression**, with explicit impact
protection, to select a volume-repair rate. Persistent overfill in a quiet
region receives slower, finite recovery. Fresh compression during an impact
retains the current response.

This proposal addresses numerical recovery. It does not promise to damp away
physical sloshing in a nearly inviscid, free-slip scene.

## Evidence and existing mechanisms

The [minidam64 settling investigation](../benchmarks/uniform-settling-2026-09-29.md)
found persistent exchanges between kinetic and gravitational potential energy,
severe local transport overfill, and pressure-driven motion in initially still
water. Increasing pressure work alone did not settle the dam. Disabling volume
correction reduced motion but allowed unacceptable excess to accumulate.

Two existing mechanisms serve different purposes:

- **Local volume repair** supplies a divergence target to pressure. The
  [current relaxation](../../lib/methods/uniform/uniform-volume-correction.wgsl.ts)
  removes half the excess per 1/30 second, capped at one owner-capacity per
  step. Surface-deficit balancing supplies compensating contraction where
  eligible capacity is available.
- **Global surface-volume correction** shifts phi to match represented surface
  volume to total conservative V, subject to its displacement bound and
  available surface. It does not directly redistribute V out of an overfilled
  cell.

There is therefore no direct exchange of “less local repair” for “more global
surface correction.” Both errors must remain accounted for. A quieter-looking
surface is insufficient if local excess continues to grow.

## Local assessment

Assess activity over a small neighbourhood, initially a tile plus nearby
support. Use a consistent physical footprint across h/4h ownership changes;
otherwise changing resolution could itself switch the recovery policy.

| Local evidence | Interpretation | Intended response |
|---|---|---|
| Surface liquid approaching a wall or solid, or an upward wall sheet | Impact or splash continuation | Preserve aggressive recovery |
| Converging relative velocities accompanied by fresh overfill | Active compression, including liquid–liquid impact | Increase recovery activity |
| Persistent excess with little fresh compression and weak relative motion | Quiet residual error | Gentle, finite recovery |
| Fast translation with little relative compression | Speed alone does not establish an impact | Do not activate solely from speed |

Reuse the wall-approach and lift logic in the
[dynamic classifier](../../lib/methods/uniform/uniform-mixed-dynamic.ts), while
keeping recovery activity separate from the decision to refine a tile. The
signals should be evaluated for the current frame, before applying its new
volume correction. Divergence alone is insufficient: a projected flow can
have strong converging and stretching directions while remaining nearly
divergence-free.

### Distinguish fresh compression from transported excess

Let the intensive excess field be

\[
e_i=\max(V_i-c_i,0),
\]

where \(c_i\) is the owner's open capacity. Transport the previous excess with
the same conservative weights used for V. Immediately after transport, before
cleanup and sharpening, measure

\[
e_{\mathrm{fresh},i}
=\max\left(0,
e_{\mathrm{after\ transport},i}-\mathcal T(e_{\mathrm{previous}})_i
\right).
\]

This separates new excess from an old overfilled packet arriving at a new
cell. An Eulerian difference between consecutive cell values cannot make that
distinction. The transported excess must obey the same owner-volume scaling
and conservative remap rules as V.

Fresh excess is a numerical diagnostic, not proof of physical impact. Combine
it with wall contact or converging relative motion. Repeated numerical
compression must not automatically keep a quiet region in impact mode.

Start by recording this scalar in the diagnostic probe. Its usefulness and
runtime cost should be measured before adding persistent production storage.

### Avoid a self-sustaining detector

Raw speed and overfill magnitude are unsuitable as the sole activity signal.
Recovery itself can produce fast motion, which could activate more aggressive
recovery on the next step. Measuring before the current correction helps but
does not remove the previous frame's corrective velocity.

Validation must explicitly test this feedback. If wall/strain signals and
fresh-excess history cannot distinguish it reliably, additional attribution
of correction-induced motion may be necessary. The detector should not be
described as independent of recovery until that has been demonstrated.

## Blend relaxation rates

Define activity \(a_i\in[0,1]\) and blend rates rather than per-frame fractions:

\[
k_i=(1-a_i)k_{\mathrm{quiet}}+a_i k_{\mathrm{impact}},
\qquad
\Delta V_i=e_i\left(1-\exp(-k_i\Delta t)\right).
\]

Use the current impact half-life of 1/30 second:
\(k_{\mathrm{impact}}=30\ln 2\). A quiet half-life of 0.3–0.5 seconds is an
initial experimental range, not a selected default. Keep the existing
capacity limiter for the first comparison. The exponential relaxation is
timestep independent; the existing per-step cap is a separate limitation and
must be considered when comparing different timesteps.

Policy safeguards:

- **Immediate activation, gradual release.** Impact evidence activates the
  neighbourhood immediately. Activity decays over physical time, with
  hysteresis, to avoid rapid switching or checkerboard correction strengths.
- **Finite quiet recovery.** Quiet excess continues to decay. Persistent
  growth or extreme excess overrides gentle recovery; thresholds require
  calibration rather than an arbitrary hard cutoff copied into production.
- **Consistent volume accounting.** Compute expansion totals after activity
  weighting and limiting. Recompute deficit balancing from those actual
  amounts, retaining the existing available-deficit bounds.
- **Consistent pressure levels.** Derive the requested repair volume once and
  restrict its integrated amounts conservatively into the coarse solve.
  Independently classifying h and 4h sources risks inconsistent corrections
  at the fine-band boundary.

## Broader recovery on quiet surface patches

As a separate experiment, redistribute the existing global surface shift so
quiet patches accept more of it and active splash tips accept less. One
possible form is

\[
\delta\phi_i=-\lambda\,w_i\,|\nabla\phi_i|,
\]

where \(w_i\) is a bounded mobility derived from local activity. Solve the
single multiplier \(\lambda\) against the actual represented-volume change,
including displacement limits. Freeze the mobility during that solve so the
volume response remains well defined. Simply multiplying the existing final
shift by local weights would generally break the volume constraint.

Preserve a fallback when little quiet surface is available. Do not silently
move representation between disconnected liquid bodies: component- or
basin-aware allocation is preferable, and requires separate design if those
identities are not already available.

This remains a correction to surface representation. It does not replace
local V repair, and any change to subsequent sharpening must be measured.

## Experiment sequence and acceptance

1. **Instrument the unchanged solver.** Record fresh and inherited excess,
   wall/strain activity, selected rate, integrated correction volume and its
   spatial distribution. Attribute motion with an exact MAC-face energy
   budget where possible; retain the existing energy proxy only as a proxy.
2. **Change only local relaxation.** Leave global surface correction and
   pressure settings unchanged. Compare candidate quiet rates and activity
   decay times against the current aggressive response.
3. **Test weighted surface recovery separately.** Add it only after the local
   policy's effects are understood. Run an ablation with each mechanism alone
   and both together.

Required comparisons include wall-impact dam fronts, rising and separating
sheets, liquid–liquid impacts, translating liquid, initially still pools,
long settling runs, and a scene with an active splash beside quiet liquid.
Include moving solids, disconnected bodies and h/4h ownership transitions.

Judge impact quality by front arrival, wall-climb height, sheet survival and
rebound. Judge quiet recovery by excess magnitude and age, pressure/source
work, parasitic velocity, surface drift and total mass. A reduction in visible
motion must not conceal accumulating V or weaker impact response. Check that
activity deactivates after an impact and cannot remain active solely from its
own correction history.

Track added storage, dispatches and frame cost. Preserve the maintained Uniform
lanes and existing timing ceilings. Before integrating an implementation, run
`npm run check:types`, `npm run test:unit` and `npm run test:dawn`, with Dawn
exclusive of browser and other GPU runs.

The first implementation decision should follow the diagnostic evidence:
whether local activity can reliably separate fresh impacts from quiet residual
error. Global surface weighting is a follow-up, not a prerequisite for testing
that distinction.
