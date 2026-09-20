# Empty hero pond: Dawn reproduction

2026-09-20. Production `hero-garden-hose` document, empty initial fill,
12.5 mm cells, Uniform Geometric balanced defaults, Metal Dawn, 144×96×96,
90 fixed 1/30 s steps. Browser unloaded before Dawn; each run acquired the
repository WebGPU lease. No uncaptured GPU errors. Initial experiments preceded the source-gate fix below.

## Finding

The apparent floor leak reproduces without rendering. Liquid quantity accumulates
above the floor while the independent level set loses the accumulating pool.
It is not terrain penetration in this reproduction.

At 3 seconds, quantities below are in cell volumes (one cell = 1.953125 mL):

| Measurement | Default | Correction enabled during inflow | Volume pressure rows: abandoned |
|---|---:|---:|---:|
| Expected hose input | 2384.646 | 2384.646 | 2384.646 |
| Stored V | 2387.830 | 2387.802 | 2387.742 |
| V below terrain | 0 | 0 | 0 |
| V in fully closed static cells | 0 | 0 | 0 |
| V where centre phi says air | 2295.895 | 1132.699 | 2147.186 |
| Total V beyond local open capacity | 1607.737 | 269.090 | 247.551 |
| Maximum V in one cell | 55.215 | 14.570 | 40.694 |
| Geometric surface volume, static open capacity | 102.673 | 2387.752 | not measured |
| UI smoothed occupancy estimate | 187.387 | 3742.883 | 284.314 |

Default stores 4.664 L versus 4.657 L injected, but encloses only 0.201 L
in the geometric surface measurement. 96.15% of V occupies cells whose centre
phi is nonnegative. At step 20 (0.667 s), this is already 81%; at step 30 a
single cell holds 55.45 cell volumes. At step 50 the maximum reaches 95.45.
The retained samples include the early steps and vertical volume histograms.

## Mechanism and controlled comparisons

`webgpu-uniform-reference.ts:writeParams` sets `surfaceVolumeHasSource` on
all positive-inflow steps. `encodeGeometricVolumeTransport` then skips
`UniformSurfaceVolumeCorrection` while that flag is true, even when the UI
says Total surface volume is on. The entire running hose bypasses the feedback.

Conservative transport preserves V; it does not guarantee local capacity.
Default pressure membership follows phi rather than V
(`pressureSurfacePhi`, volumePressureRows off). Once phi loses a floor film,
its stored volume is not sufficient to claim pressure rows. Sharpening is also
limited by a phi band and full-open-cell admission; it is not a general repair
for arbitrarily concentrated volume whose surface has disappeared.

The probe-only `--correct-sources` arm bypasses just the host source gate.
At 3 s the geometric surface matches stored V to about 0.0021%, versus a
95.7% deficit in the baseline. This establishes the source gate as a material
cause of the total-surface collapse. It does NOT establish a complete local
solution: over-capacity cells and phi/V spatial disagreement remain.

The independent abandoned-pressure-rows arm reduces total excess but leaves
most V outside phi-liquid cells. Pressure membership alone is not enough.
Neither experimental arm was promoted to product defaults.

## Measurement caveats

The UI's represented-volume metric is a four-cell smoothed centre-phi
occupancy, without solid-capacity clipping; it is NOT enclosed surface volume.
Its +57% result in the correction arm does not mean the actual geometric
surface overshot. The independent measurement integrates the six linear
phi tetrahedra per cell and clips by static open capacity, matching the
surface correction's geometric convention. It is not a rendered mesh audit.
Capacity classification here includes terrain and static SolidWorld occupancy,
not dynamic rigid-body fractions. Below-terrain and fully closed static-cell
V are read directly; values at or below 1e-8 cell volumes are excluded from
spatial classification. Raw total V includes them.

## Reproduction

Unload the fluid browser first; do not run these concurrently.

```sh
node --import tsx tools/probe-hero-hose-floor-dawn.ts
node --import tsx tools/probe-hero-hose-floor-dawn.ts '--values={"totalSurfaceVolume":"off"}' --out=artifacts/hero-hose-floor/correction-off.json
node --import tsx tools/probe-hero-hose-floor-dawn.ts '--values={"volumePressureRows":"abandoned"}' --out=artifacts/hero-hose-floor/abandoned-rows.json
```

The probe reads live GPU volume and vertex-phi textures. The original
`--correct-sources` experiment bypassed the private source gate without editing
the solver. After the fix that probe override was removed; use the explicit
`totalSurfaceVolume: off` control above for comparison. JSON evidence alongside this report retains the method values,
per-frame metrics and vertical profiles. The baseline was run twice and its
reported frame-90 values reproduced exactly.

Next implementation work should keep source-aware volume feedback active,
validate local capacity and pressure membership at first floor contact, and
replace or relabel the misleading UI represented-volume metric. A global
surface match alone is not sufficient acceptance for this scene.

## Source-gate fix

Removed `surfaceVolumeHasSource` and the source-step exception. The enabled
constraint now executes after gather on inflow and drop steps, using V that
already includes the injected liquid. The explicit off setting still works.

The normal product path was rerun for 90 steps: geometric surface 2387.75244,
stored V 2387.80169 cell volumes, relative mismatch 0.0021%, with zero detected
static-solid/terrain penetration. This reproduces the experimental source-on
result without an override; see fixed.json. Local excess (maximum 14.57 cell
volumes) remains unresolved and must not be described as fixed.

The existing Dawn shrunken-pool test now exercises both inactive and active
source-step strengths, checking that the constraint restores phi while leaving
V unchanged. The full-scene probe supplies the actual hose-input coverage.
