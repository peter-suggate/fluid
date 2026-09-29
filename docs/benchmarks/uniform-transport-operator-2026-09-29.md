# Uniform Geometric: frozen transport-operator census

29 September 2026, Dawn/Metal, `minimal-power-dam-break-64`, Uniform Geometric
defaults, dt 1/30 s, HEAD `301e76c6` snapshot (the working tree's h-band edits
abort this scene). This follows "Replacement investigation: isolate the
transport operator" in
[local volume recovery](../plans/uniform-local-volume-recovery.md), using the
[operator probe](../../tools/probe-uniform-transport-operator-dawn.ts).
Results are in the [compact JSON](uniform-transport-operator-2026-09-29.json).

**Result.** The balancing is not the source of the excess, and more rounds do
not remove it. The capacity defect d = BC − C follows the compression of the
departure map. That compression persists in deep liquid when the traced MAC
field is nearly divergence-free, and it scales linearly with travel. New
overfill from bounded input is about 1.2–1.7k cell volumes per step at 0.6 s.
Neither disabling local repair nor a tight pressure tolerance reduces it. The
generator is the departure map from the cell-centred RK2 trace, not repair,
pressure or the number of balancing rounds.

## Method

The probe wraps the frame's transport at chosen steps. After the real
transport, in the same command encoder, it replays clear, build, R balancing
rounds and gather on the same ownership, departures and incoming V. The replay
uses full support: every tile is both a row and a donor. Each replay gathers:

- C (unit capacity), which gives the defect BC − C;
- the incoming V;
- the bounded part clamp(V, 0, 1). Its overfill above 1 is new excess from
  initially bounded input.

Checks:
- Full-support R = 3 equals the live transport exactly (|full − live| = 0), so
  the compact three-round closure loses nothing.
- ΣBC − ΣC and ΣBV − ΣV are about 2–5·10⁻³ cell volumes (quantization), so
  donor conservation holds.
- A control replay of the unmodified map through a copied departure texture
  matches R = 3 exactly.

A replay after the frame does not work: the stage's scratch state has changed
by then and returns garbage around 6·10²¹. All replays therefore run inside
the frozen frame's encoder.

## More rounds: stall

| Step | R | Positive defect | Max defect | Bounded excess |
|---|---:|---:|---:|---:|
| 18 (0.6 s) | 1 / 3 / 16 | 26,127 / 22,118 / 17,418 | 8.0 / 8.0 / 8.0 | — / 1,225 / 1,086 |
| 30 (1.0 s) | 3 / 16 | 32,125 / 26,523 | 8.0 | 2,053 / ~1,743 (R8–R12) |
| 360 (12 s) | 3 / 16 | 13,371 / 9,415 | 5.5 / 5.0 | 486 / 425 |

Bounded excess reaches its minimum at R 8–12 and then rises slightly. Sixteen
rounds cut the defect by 21–30% and the overfill by about 12%, which is not
convergence to BC = C.

## Where it sits

- **Walls and h/4h seams are minor.** Owners within one cell of a closed wall,
  or in a tile touching a tile of the other width, hold under 10% of the
  bounded excess.
- **Step 18:** interior liquid (V_in ≥ 0.5) holds 639 of 1,225 and air or
  partial cells 586.
- **Step 360:** interior liquid holds 323 of 486.
- **Defect:** 60–70% sits in air and partial owners.
- **Worst receivers** are air cells (V_in = 0) with about 2–3 cells of travel.
  They gather 3–6 cell volumes of liquid (BC up to 7). Top-receiver lists are
  in the JSON.

## Alternative departure maps on the same frozen inputs

All variants use R = 3. Cells are positive defect / bounded excess.

| Variant | Step 18 | Step 360 |
|---|---:|---:|
| Base | 22,118 / 1,225 | 13,371 / 486 |
| Displacement smoothed over 3³ h owners | 20,806 / 1,270 | 12,706 / 487 |
| Smoothed twice | 19,906 / 1,319 | 12,331 / 523 |
| Travel × 0.5 | 11,629 / 396 | 6,963 / 183 |
| Travel × 0.25 | 5,911 / 164 | 3,672 / 95 |

- **Smoothing** removes most of the thinly covered donors (liquid on donors
  with coverage < 0.1 falls 851 → 133) but barely changes the defect.
  Overfill rises. Under-coverage in the map's fine structure is not the
  cause.
- **The defect is linear in travel.** Overfill falls faster than that.
  Scaling travel changes the physics; it is a diagnostic, not a candidate.

## The defect is departure-map compression

Column normalization gives a receiver about det J of capacity, where
det J ≈ 1 + div(dep − x). The table compares the per-owner R = 3 defect with
that central-difference divergence, for h owners with six h neighbours.
Correlation is r.

| Class | Step 18 r | Step 360 r | Step 18 RMS div(dep − x) | Step 360 RMS div(dep − x) |
|---|---:|---:|---:|---:|
| Deep liquid (self and six neighbours V ≥ 0.5) | 0.73 | 0.61 | 0.117 | 0.078 |
| Surface liquid | 0.54 | 0.74 | 0.41 | 0.33 |
| Partial | 0.72 | 0.77 | 0.54 | 0.41 |
| Air | 0.74 | 0.76 | 0.69 | 0.39 |

## Repair and pressure do not drive it

Four arms, each a separate trajectory frozen at the same steps:
- **Base:** defaults.
- **Tight:** `pressureResidualTolerance` 0.01.
- **No correction:** `uvVolumeCorrectionAmountAt` returns 0.
- **Both:** tight and no correction together.

"MAC" is the RMS divergence of the traced field (`velocityScratch`, the
extension that the RK2 trace samples) in dt/h units. "Map" is RMS
div(dep − x). Both are measured in deep liquid.

| Arm | Step 18 MAC | Step 18 map | Step 18 bounded excess | Step 360 MAC | Step 360 map | Step 360 bounded excess |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.164 | 0.117 | 1,225 | 0.081 | 0.078 | 486 |
| Tight | 0.180 | 0.140 | 1,251 | 0.069 | 0.052 | 315 |
| No correction | **0.036** | 0.105 | 1,311 | 0.089 | 0.069 | 566 |
| Both | 0.114 | 0.122 | 1,712 | 0.049 | 0.068 | 174 |

- **Repair dominates the deep-liquid MAC divergence.** Without repair it falls
  from 0.164 to 0.036 at step 18. The repair's divergence target is
  intentional.
- **The departure-map divergence and the fresh overfill barely move without
  repair** (0.105; 1,311). The trace compresses the map at about three times
  the divergence of the field it samples.
- **The tight tolerance changes little.** Late overfill in the "both" arm is
  lower (174 against 486), but that trajectory is calmer and still stores 13k
  of excess, because nothing repairs it.

Interpretation: the RK2 midpoint trace through the interpolated MAC field
yields cell-centre departure points. Their discrete divergence is a
first-order interpolation error wherever the velocity varies at cell scale,
and it is not the MAC divergence. The box remap then turns that compression
directly into capacity error. Balancing cannot remove it: donor
conservation forces the compressed receivers to take the capacity.

## Next

Test a departure map whose box volume change equals the MAC divergence by
construction, on the same frozen inputs, before any coupled run. One option
is a face-traced box: each receiver box face moves by its own face velocity,
so the box volume changes by dt·div_MAC and the defect is zero where the
field is projected. The build kernel currently assumes a cube of the owner
width at the departure point, so this needs a second departure texture
(lower and upper corners) and a build change in a scratch copy. Local repair
and pressure stay unchanged.

The still-pool tight-pressure control from the plan was not run in this
round.

## Reproduction

```sh
node --import tsx tools/probe-uniform-transport-operator-dawn.ts --steps=18,30,360 --rounds=1,2,3,4,6,8,12,16 --out=/tmp/op.json
node --import tsx tools/probe-uniform-transport-operator-dawn.ts --steps=18,360 --rounds=3 --out=/tmp/op-base.json
node --import tsx tools/probe-uniform-transport-operator-dawn.ts --steps=18,360 --rounds=3 --values='{"pressureResidualTolerance":0.01}' --out=/tmp/op-tight.json
node --import tsx tools/probe-uniform-transport-operator-dawn.ts --steps=18,360 --rounds=3 --control=no-correction --out=/tmp/op-nocorr.json
```

No production file changed. Only the targeted probe ran on Dawn; no Dawn
suite was run.
