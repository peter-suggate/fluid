# Garden SVO lighting at 1500×1500 — where the frame goes

Date: 2026-08-01 · M1 Max · Dawn/Metal

Question this answers: the UI renders `garden-svo-lighting` at about 19 fps with
cone tracing and about 25 fps with cone tracing, shadows, and AO switched off.
The 6 fps gap invites the conclusion that cone tracing is the cost. It is not.

## Capture

`tools/profile-svo-render-xctrace.ts` attaches xctrace to the render-only
dry-frame worker from outside the process, so the shipping pass graph is
observed undistorted. Production configuration is
`SparseVoxelDrySceneRenderer(…, "canonical-parametric", "off", "split", 0,
"static-primary", …)` with `DEFAULT_SVO_RENDER_TUNING`.

```bash
node --import tsx tools/profile-svo-render-xctrace.ts \
  --scene=garden-svo-lighting --resolution=1500x1500 \
  --traversal=canonical-parametric --shading=split \
  --cone-scale=0.5 --radiance-reconstruction=full-res-relight \
  --cone-tracing=cones --counter-seconds=3 \
  --variant=garden-1500-cones --out=artifacts/xctrace-garden-1500-cones

# cones-off arm: --cone-tracing=off (implies --cone-scale=1)
```

`--cone-tracing` and its worker knob `FLUID_SVO_DRY_FRAME_CONE_TRACING`
(`cones | exact | off`) were added for this A/B — the UI's `SvoConeTracingMode`
switch previously had no headless equivalent. The non-`cones` arms are asserted
to the external-profiler lane, because withholding the cone stages removes the
reduced-rate plane the benchmark's A/B, image-comparison, and attribution lanes
are defined against.

## Frame structure

| | cones | cones off |
|---|---:|---:|
| Frames analysed | 66 | 100 |
| Frame wall / frame | 50.80 ms (p10 48.12, p90 51.52) | 35.19 ms (p10 32.30, p90 37.18) |
| GPU busy | 48.37 ms = 95.2 % of wall | 32.72 ms = 93.0 % of wall |
| Idle gaps | 2.43 ms across 13 | 2.47 ms across 4 |
| Encoders / passes per frame | 4 / 4 | 2 / 2 |
| Exact label attribution | 100 % | 100 % |
| Fragment / Compute / Vertex | 37.38 / 13.51 / 0.39 ms | 32.70 / — / 0.12 ms |
| Mean occupancy (ALU) | 10.6 % (21.2 %) | 21.6 % (32.7 %) |
| Counter coverage | 29.5 % uncontended | 11.3 % uncontended |

19.7 fps and 28.4 fps respectively — the cones arm reproduces the UI's 19 fps
exactly; the off arm runs a little ahead of the UI's 25 fps because the harness
carries no browser present/composite.

## Trace breakdown

Per-frame GPU time by exactly isolated task. Occupancy is Fragment Occupancy;
bandwidth columns are the GPU read/write counters already expressed as a rate.

**Cone tracing on — 48.75 ms attributed**

| Pass | ms | share | occ | ALU | LLC | read | write | placement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sparse voxel primary visibility | 26.95 | 55.3 % | 22.2 % | 34.6 % | 28.6 % | 3.7 GB/s | 14.6 GB/s | 1.01× |
| Sparse voxel deferred dry lighting | 10.83 | 22.2 % | 12.4 % | 13.4 % | 30.8 % | 25.2 GB/s | 39.9 GB/s | 1.00× |
| Sparse voxel compact cone visibility | 9.14 | 18.7 % | **4.7 %** | 23.2 % | 18.9 % | 6.1 GB/s | 4.3 GB/s | 1.10× |
| Sparse voxel persistent world GI cache | 1.84 | 3.8 % | 3.2 % | 57.2 % | 34.9 % | 14.0 GB/s | 2.2 GB/s | 1.35× |

**Cone tracing off — 32.82 ms attributed**

| Pass | ms | share | occ | ALU | LLC | read | write |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sparse voxel primary visibility | 27.36 | 83.4 % | 22.2 % | 34.9 % | 29.5 % | 3.7 GB/s | 14.7 GB/s |
| Sparse voxel deferred dry lighting | 5.46 | 16.6 % | 16.5 % | 14.5 % | 28.8 % | 23.6 GB/s | 32.8 GB/s |

**Delta attributable to cone tracing: +15.93 ms**

| | ms |
|---|---:|
| `compact cone visibility` (new pass) | +9.14 |
| `persistent world GI cache` (new pass) | +1.84 |
| `deferred dry lighting` growth — relight now reads and upsamples the cone plane | +5.36 |
| `primary visibility` | −0.41 (noise) |

## Findings

1. **Cone tracing costs 16 ms, not the 12 ms the fps delta implies.** Frame-rate
   differences compress at low framerates; score lighting work on the pass
   table, never on fps.

2. **`primary visibility` is 27 ms and is unmoved by every lighting switch.** It
   is 83 % of the cones-off frame. With all visibility work removed the frame
   still cannot beat ~28 fps at this resolution, because the majority of it is
   one full-screen SVO traversal filling the G-buffer. Cone tracing is layered
   on top of a frame that was already primary-bound.

3. **Nothing is bandwidth bound.** `primary visibility` reads at 3.7 GB/s against
   the ~78 GB/s peak observed on this part, with LLC at 29 %. The GPU is busy
   95 % of the frame wall while ~78 % of its thread slots sit empty. This is the
   register-pressure ceiling documented in
   `SVO_DAWN_OCCUPANCY_EXPERIMENTS_2026-07-31.md`, now the dominant term rather
   than a 6 % tail.

4. **`compact cone visibility` is the worst-decomposed pass in the frame** —
   9.14 ms at 4.7 % occupancy and 23 % ALU, spread evenly across partitions
   (1.10×). It is starved per core, not pinned to one quarter of the machine, so
   the fix is per-lane state, not more workgroups.

5. **Cost is pixel-linear.** Fixed overhead amortises slightly, but there is no
   knee: ~19–22 ms/Mpixel with cones, ~13–15 ms/Mpixel without.

   | Square resolution | cones | cones off | ratio |
   |---|---:|---:|---:|
   | 750 | 16.69 ms (59.9 fps) | 10.73 ms (93.2 fps) | 1.56× |
   | 1060 | 28.57 ms (35.0 fps) | 18.97 ms (52.7 fps) | 1.51× |
   | 1500 | 49.61 ms (20.2 fps) | 33.54 ms (29.8 fps) | 1.48× |
   | 2000 | 76.54 ms (13.1 fps) | 52.51 ms (19.0 fps) | 1.46× |

## The lever that is already implemented and switched off

`stationaryPrimaryReuseEnabled` (VisualPanel "Reuse stationary visibility") is
`false` in `DEFAULT_SVO_RENDER_TUNING`. It skips `primary visibility` entirely
while camera and scene are unchanged. Measured at 1500×1500:

| Configuration | Median | fps |
|---|---:|---:|
| cone 0.5, reuse off (current default) | 47.58 ms | 21.0 |
| cone 0.5, **reuse on** | 20.67 ms | 48.4 |
| cone 0.25, reuse off | 41.75 ms | 24.0 |
| cone 0.25, **reuse on** | 14.73 ms | 67.9 |

The saving is 26.9 ms against a profiled `primary visibility` cost of 26.95 ms —
the two measurements are independent and agree, which is the check that the
attribution above is real.

Two constraints. Reuse holds only while the camera is still; orbiting pays full
price every frame. And `svoDryPrimaryCoherenceDecision` requires
`useSplit && usePrepass`, so it works with cone tracing **on** and not with it
off — scale 1 still owns checkerboard shadow-deferred flags and must retrace.
For a lighting-study scene this is worth far more than disabling cone tracing
ever could be.

The second lever is `resolutionScale` (default 0.72), which by finding 5 buys
back time proportionally with no correctness question attached.

## Measurement integrity

Per-pass GPU time filters by process and is exact. Occupancy and ALU are
device-wide counters sampled only where no other process held GPU work, and
coverage was 29.5 % / 11.3 % (WindowServer 3.0 s, Codex 0.57 s, Cursor 0.17 s of
GPU in the window). Treat those as the softer numbers — though `primary
visibility` reported 22.2 % occupancy identically across both arms and all four
partitions, which is the behaviour of a real ceiling rather than a sampling
artefact.

Reports: `artifacts/xctrace-garden-1500-cones/report.html`,
`artifacts/xctrace-garden-1500-nocones/report.html`.
