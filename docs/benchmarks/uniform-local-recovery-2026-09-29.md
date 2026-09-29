# Uniform Geometric: local volume recovery experiment

29 September 2026, Dawn/Metal, `minimal-power-dam-break-64`, Uniform Geometric
defaults, dt 1/30 s. This tests steps 1 and 2 of
[local volume recovery](../plans/uniform-local-volume-recovery.md) using the
[recovery probe](../../tools/probe-uniform-recovery-dawn.ts).
**Result: not adopted.** A slower quiet repair rate reduces motion only by
letting conservative-V excess accumulate. This fails the plan's acceptance
rule: less visible motion must not conceal accumulating V.

The runs used a HEAD (`301e76c6`) snapshot. The working tree held another
session's uncommitted h-band edits, and those abort this scene with a
`hangingCapacity` layout-builder fatal. The experimental solver change is kept
as [a patch](uniform-local-recovery-2026-09-29.patch) against HEAD and is not
applied to the tree. Results are in the
[compact JSON](uniform-local-recovery-2026-09-29.json).

## Step 1: fresh versus inherited excess (unchanged solver)

Transport is linear in V for a frame's weights (rows and donor sums are
geometric), so the probe re-runs the frame's own gather on max(V_in − 1, 0).
This gives the inherited excess T(e_prev) exactly. Fresh excess per owner is
max(0, excess after transport − inherited).

| Window | Standing excess (% of water) | Fresh per step | Stale excess (tile had no fresh) |
|---|---:|---:|---:|
| 0–1 s | 6.3% | 1316 cell volumes | 6% |
| 1–2 s | 4.3% | 679 | 15% |
| 5–10 s | 2.0% | 412 | 13% |
| 10–20 s | 2.0% | 420 | 14% |

**Fresh excess cannot tell impact from quiet liquid.** It is produced every
step in nearly every tile that holds excess. Detectors triggered by fresh
excess (≥ 0.05 or 0.25 cell volumes per tile, over a 3×3×3 tile
neighbourhood) mark 99–100% of the excess active in every window.
Part of this fresh excess is probably the repair itself: its expansion pushes
volume into full neighbours, which the next transport records as new excess.
That is the self-sustaining loop the plan warned about. This interpretation
was not isolated.

Confirming fresh excess with compressive normal strain of u* separates only
partly. A closed wall face counts as zero normal velocity, so liquid driven
into a wall compresses. With a 10 s⁻¹ threshold, active tiles hold 91% of the
excess during the impact and 51–54% during late sloshing. Median, 90th and
99th percentile liquid-tile strain is 2.8/10/26 s⁻¹ during the impact and
1.6/7/12 s⁻¹ from 5 s on. The dam is still sloshing at that point; it is not
a quiet region.

## Step 2: rate blending (global surface correction and pressure unchanged)

The patch adds a per-4h-tile activity word:
1. The transport gather accumulates each tile's fresh excess.
2. After forces, a tile triggers on fresh excess plus strain.
3. Activity dilates over one tile and releases over 0.25 s.
4. Both pressure authorities (h/4h simulation and all-4h split) blend
   k = mix(k_quiet, 30 ln 2, activity), so the h and 4h sources come from one
   rate.

Deficit balancing sums the actual rate-weighted amounts, and the per-step cap
is unchanged. With the policy off, the patch diverges from HEAD at the ulp
level from step 2, consistent with Metal FMA contraction from the changed
gather. Clean HEAD is bitwise repeatable (202 of 202 samples).

RMS is the owner-volume-weighted speed of the projected velocity, a proxy.
Climb is the highest cell with V > 0.5 in the four-cell x-wall slabs; the
tank is 0.8 m tall.

| Dam arm | RMS 5–20 s | Excess 5–20 s | Peak excess | Climb 1–2 s (low/high wall) |
|---|---:|---:|---:|---:|
| Baseline | 0.43 m/s | 2.0% | 6.3% | 0.80 / 0.80 m |
| All quiet, half-life 0.4 s | 0.25 | 4.9% | 11.8% | 0.78 / 0.49 |
| Strain ≥ 10 s⁻¹ policy, quiet 0.4 s | 0.37 | 4.3–4.6%, still rising | 7.1% | 0.80 / 0.80 |

| Still-pool arm (initially flat) | RMS 5–10 s | Excess 5–10 s | Fresh per step |
|---|---:|---:|---:|
| Baseline | 0.055 m/s | 1.2% | 53 |
| Strain policy | 0.060 | 5.4% | 136 |

- **Aggressive repair drives the wall run-up.** With all-quiet repair, the
  first far-wall climb falls from 0.80 to 0.49 m. The strain policy keeps it.
- **Fresh generation barely depends on the repair rate** on the dam (396–442
  per step against 420), so standing excess rises as repair slows. In the
  still pool, generation even rises (53 → 136 per step) while excess grows
  4.5×. There, motion comes from the loose coarse pressure target
  (settling report), not from repair, and slower repair makes it slightly
  worse.
- **Most of the motion reduction is traded for V excess.** The strain policy
  removes about 12% of late RMS for about 2.3× the standing excess. All-quiet
  removes 42% for about 2.4× the excess, plus a weaker impact.
- The total-V drift over 20 s (−1.4%) is the same in every arm.

## What this points to

Local recovery treats a symptom whose source is steady: transport creates
about 20% of the standing excess again every step. The plan's quiet regime,
where persistent excess exists with little fresh compression, does not occur
here: only 13–15% of the excess sits in tiles without fresh excess. Removing
the generation comes before choosing a repair rate:

- the receiver-capacity residual after three row/donor balancing rounds;
- the departure map;
- the coarse pressure target that drives motion in still water.

Those are the settling report's first two recommendations. The detector could
be revisited once fresh generation is small. The patch then provides the
tile-activity path, and the probe provides the census.

## Reproduction

The probe acquires the repository GPU lease and awaits each frame. Policy arms
need the patch applied to a HEAD snapshot; `--recovery` is otherwise ignored.

```sh
node --import tsx tools/probe-uniform-recovery-dawn.ts --frames=600 --out=/tmp/rec-dam.json
node --import tsx tools/probe-uniform-recovery-dawn.ts --frames=300 --initial=rest --out=/tmp/rec-rest.json
# with the patch:
node --import tsx tools/probe-uniform-recovery-dawn.ts --frames=600 --recovery='{"quietHalfLife":0.4,"release":0.25,"fresh":1e9,"strain":0,"extreme":1e9}' --out=/tmp/rec-qall.json
node --import tsx tools/probe-uniform-recovery-dawn.ts --frames=600 --recovery='{"quietHalfLife":0.4,"release":0.25,"fresh":0.05,"strain":10,"extreme":1e9}' --out=/tmp/rec-qs10.json
```

No production file changed. The clean-repository gate was not run, because
only a tool and docs were added. The patched snapshot type-checked and passed
the CPU WGSL preflight (naga parses every module).
