# Spatial performance layer

Open a 3D Uniform scene, enter Edit, then choose **Visual layers → Performance**.
Run the simulation to collect a sample. Choose X/Y/Z and a slice position, or
choose 3D in the Performance controls. Aggregation is additive: Cells, 4h tiles,
and 16h regions preserve the same mapped cost. Select a stage to isolate it.

## Reading the map

The fill represents attributed milliseconds divided by `frame wall ms / number
of displayed entities`. A value of 1× is one average share of the **whole frame**,
including shared time. Changing the selected stage does not renormalize the
colour scale. Changing aggregation changes the average share, with its absolute
microsecond value shown in the legend. The ramp has labelled anchors at 0, ¼×,
1×, 4×, and 16×; values above 16× saturate. Opacity is adjustable.

Blue, amber and red rims independently indicate cold, warm and hot regions.
**Work tiers** replaces the fill with those categorical colours. Cold does not
mean free: global pressure and other required work can still contribute there.
In 3D transparent fills and cost-coloured wireframes reveal the whole volume.
Depth does not occlude interior cells; face opacity decreases with entity density
to keep fine views readable. Tier colours form a faint halo around the wires.
Overlapping colours blend, so use a slice for precise colour comparison. Costs that have not passed frame-model validation are
hatched, and the legend explicitly labels them provisional.

The legend reports the sampled frame, measured wall latency, mapped time and
shared/unmodelled time. These always close to the observed wall time. Individual
stages also report their total and their shared portion. A sample is immutable:
work evidence and timing come from that same frame. It remains visible when
paused and refreshes asynchronously at most once every 350 ms while running.

## Measurement and accuracy

Wall latency starts at `advanceTo` entry and ends when the frame's submitted GPU
work completes. It includes encoding and queue contention. CPU and GPU overlap,
so their durations are not added. Diagnostic stage times use the existing GPU
completion-frontier recorder across submitted segments. Overlap is counted once;
valid zero-duration segments represent fully skipped stages. Missing hardware
counters are rejected. The snapshot copy/readback is submitted after the wall
completion fence. Diagnostic instrumentation itself still affects the workload.

The spatial attribution is an estimate, **not a per-cell hardware timer or a
measurement of marginal speedup from deleting a cell**. GPU occupancy, caching,
parallel work and nonlocal pressure prevent such an interpretation. Cell mode
splits a tile's cost equally between its actual solver owners; differences
within an h tile are unresolved. Surface store counts are measured per 16h region
and distributed among its owners. Shared/unexplained time stays outside the map.

A non-negative regression learns stage costs from producer-owned work evidence.
Every fifth timing observation is held out. The history is bounded to 80 samples.
Prediction requires at least 24 training and 6 held-out observations, ≤15%
aggregate absolute error, and features within the observed range (20% margin).
A fitted fixed overhead is subtracted only when training includes a zero-work
reference; a good fit over a narrow nonzero workload cannot establish overhead.
Identifying spatial coefficients additionally requires varying work. Constant
work can validate a stage-total prediction without validating its spatial split.
Frame-total validation does not prove the local spatial allocation.

**Live estimate** samples diagnostic frames until the stage models and the
surface-work classifier pass validation, then evaluates their small compiled
rules between diagnostic refreshes (at least every 2.5 seconds). A changed or
unvalidated workload stays on diagnostic sampling. **Measure + learn** always
uses measured stages on sampled frames. Neither mode instruments every rendered
frame. Turning off the layer stops collection and the diagnostic producer
counters. A solver parameter generation change resets learning. No device-specific
coefficients are shipped as universal constants.

## Extension points

- `lib/core/spatial-performance.ts`: solver-independent plugin vocabulary,
  work records, fitting, validation and conservative wall accounting.
- `lib/methods/uniform/uniform-spatial-performance.ts`: the Uniform adapter.
  Each stage owns explicit, disjoint producer trace labels and feature names.
  Existing ownership, tier, particle-bin, pressure, cycle-count and band buffers
  provide cheap evidence. Surface diagnostic store counters train a regional
  work classifier. No shader implementation-name matching is used.
- `lib/features/spatial-performance/`: persisted controls, aggregation, legend
  and instanced WebGPU drawing. The drawing code has no solver imports.
- `GPUSolverInstance.setSpatialPerformanceEnabled` and `spatialPerformanceSnapshot`
  are optional provider hooks. Another solver supplies the same snapshot shape.

When a producer changes, update its adapter features/labels and bump the plugin
revision. Keep any new diagnostic-only work behind `performanceDiagnostics`.
New local evidence should be independently validated; do not interpret a low
frame prediction error as proof that every cell's share is right.

## Reproducible experiments

Run on an otherwise idle GPU; other browser scenes and GPU benchmarks contaminate
wall latency. The benchmark uses the production solver and saves observations,
compiled coefficients, validation error, feature totals, tier totals and the final
spatial snapshot under the requested output directory.

```sh
node --import tsx tools/benchmark-spatial-performance.ts --enabled --steps=96 --out=artifacts/spatial-performance/dynamic
node --import tsx tools/benchmark-spatial-performance.ts --enabled --settled --quiet --steps=96 --out=artifacts/spatial-performance/cold
node --import tsx tools/benchmark-spatial-performance.ts --steps=96 --out=artifacts/spatial-performance/baseline
node --import tsx tools/probe-spatial-performance.ts
npm run check:types
npm run test:unit
```

Use `--diagnostic` to keep timing every sampled frame. `--settled` constructs a
level tank fill; `--quiet` removes gravity for a stable cold workload. A/B pacing
is identical; compare repeated runs, excluding construction and initial warmup.
The native render probe exercises X/Y/Z, all aggregations, 3D, stage filtering,
tier paint, hide/restore, GPU validation and deterministic restoration.

Initial count-only experiments had roughly 27–41% held-out errors on several
stages. They did **not** justify trusted compiled predictions. This led to adding
producer store modes, pressure/band evidence and executed cycle counts, and to
requiring validation before entering compiled mode. Provisional results remain
visibly provisional rather than being presented as exact timings.

### Local verification, 2026-10-18

The 96-step quiet run remained all-cold after settling and retained a nonzero
pressure contribution. It exposed fully skipped GPU segments; equal, nonzero
hardware boundaries now count as zero work instead of invalidating the sample.
A stricter 32-step dynamic run then published a matching valid snapshot for every
step, with no GPU validation errors. Floating-point trace remainders around
1e-15 ms are clamped; materially negative timings still fail closed.

In that dynamic run, held-out stage errors at sample 32 were approximately 13%
layout, 2% support, 11% surface, 34% geometry, 10% momentum, 8% pressure and 3%
band. **This did not validate every stage**, so automatic mode correctly retained
measured stage sampling. No claim of independently validated cell timings is made.
The pressure workload in that run did not vary enough to identify its spatial
coefficients, despite a reasonable stage-total prediction.

A single paired run (32 steps each; first eight excluded) gave median host frame
completion of 27.74 ms with sampling every step versus 22.18 ms without sampling.
These are diagnostic-frame costs, including instrumentation and collection, not
the ordinary UI's amortized overhead: UI sampling is capped at once per 350 ms.
Repeat A/B runs on an idle device before treating this as a performance budget.

The native rendering probe passed all nine views, including zero pixels when
hidden and identical pixels after restoration. Browser checks covered live
updates, paused inspection, stage selection, X/Y/Z, cells/tiles/regions, 3D and
work-tier paint. Slices default to 100% opacity so background colour cannot distort their heat
scale. The 3D view intentionally uses transparency and wireframes to expose the
interior; its opacity slider adjusts both fill and wires.
