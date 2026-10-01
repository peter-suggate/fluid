# Uniform Geometric stage analysis (2026-10-01)

Peter asked to "optimize the heck out of" Uniform Geometric (`uniform-volume`). The first pass removes unneeded
shader code, packs data more tightly, deletes unneeded storage and looks for denser layouts. Each stage was analysed
for what it does, the data it needs and its access patterns, with particular suspicion of chained lookups and loops
over indirect memory.

The analysis was done by five read-only reviews at HEAD c38adea3. They used code reading plus a mock-device WGSL
capture (`tools/capture-uniform-wgsl.mts`), with no GPU runs. Per-group reports:
[head](head.md) (census, layout, ownership, copies, full allocation inventory), [levelset](levelset.md),
[volume](volume.md), [velocity](velocity.md), [pressure](pressure.md). Per-kernel millisecond figures in the reports
are estimates from load counts. Only the stage totals below were measured.

## Baseline (HEAD c38adea3, M1 Max, Dawn/Metal)

| | fig-9 (cm12-figure-9, 128x128x64) | dam64 (minimal-power-dam-break-64) |
|---|---|---|
| Pipelined wall per advance | 19.43 ms (splash 47-128: 23.64, tail 129-245: 16.39) | 11.41 ms |
| Fenced GPU per frame | 22.42 ms | 12.09 ms |
| Allocated | 369 MiB in 388 resources | 93 MiB |

Fenced stage means for fig-9 (frames 5-130), in ms:

| Stage | ms |
|---|---|
| Vertex phi transport and redistance | 3.65 |
| Mixed extension sweeps + hierarchy fill | 3.04 |
| h pressure band solve + projection | 2.33 |
| Velocity advection + body forces | 2.22 |
| Geometric volume coupling | 2.15 |
| Conservative volume sharpening | 2.06 |
| Surface volume constraint + geometry | 1.73 |
| Census + layout build + remap | 1.56 |
| CM11a V-cycles | 1.51 |
| CM11a topology + RHS | 1.07 |

## Cross-cutting findings

- **The frame is launch- and latency-bound, not bandwidth-bound.**
  - Fig-9 frame 2 encodes 1214 dispatches in 79 compute passes and 20 blit runs.
  - dam64 has a quarter of the cells but costs half the GPU time.
  - The head stage fits roughly 0.70 ms fixed plus 0.05 µs per tile.
- **Skipped dispatches still cost time.** Pressure encodes 3 slots of 74 dispatches each and runs only one. The two
  closed slots still cost 0.61 ms, about 4 µs per gated dispatch. So every dispatch removed from the slot body saves
  about 14 µs per frame.
- **Dependent-load chains are the per-lane cost.**
  - A regular momentum face does about 63 loads in 7 dependent round trips.
  - Each velocity sample loads a support word before its 24 rgba32f taps.
  - The surface sampler lost its regular-texture fast path (`regularTexture` has not been passed since c9e9fe17).
  - Redistance re-loads a tile width for each of the 2744 window entries, although the window spans only 64 tiles.
  - The sharpening sweeps read read-only lists with atomics.
  - Seam extension faces chain 4-5 topology loads.
- **Dead shader code costs compile time and memory, not frame time.** Tint prunes each pipeline to its entry point.
  - Module #6 (CM11a hierarchy, 333 KB) is about 19% reachable.
  - Module #1 (reference kernels, 254 KB) dispatches only at t=0.
  - The transport module compiles 10 dead pipelines.
  - Module #5 compiles 15 legacy pipelines.
  - The solid twins of the band and transport pipelines are kept on purpose, so a live voxel edit can switch without a
    compile.
- **About 150 MB of the 369 MiB allocation is deletable or oversized:**
  - band capacity sized for 16384 tiles against an observed max of 3582: about 76 MB
  - reference transportA: 17.8 MB
  - front convergence receipts: 16.8 MB
  - remapped-extension texture, which can alias a texture holding nothing live: 16.8 MB
  - hanging vertex-cache words: 8.2 MB
  - extrapolation level 0 and origins: about 12.6 MB
  - CM11a L0 pressure A: 4.5 MB

Packing 4h data by itself is still expected to be null on Apple's tiled 3D textures. Wins come from deleting
loads, launches, copies and atomics.

## Round 1 workstreams (exact or ulp-level only)

| Workstream | Main items | Estimated fig-9 GPU gain |
|---|---|---|
| A1 levelset | redistance window width table, sharpening without atomics, direct resolved-phi loads, surface sampler fast path, cubic reuse | 0.7-1.6 ms |
| A2 velocity | certificate requires fine support (drop the support load), face-centre first RK2 sample, extension reads staged words, restrictBand lanes, umFace width from the h mask | 0.6-1.3 ms |
| B volume | dead pipelines, tier-merged transport launches, transport ping-pong (no copy), surface-volume dispatch trims, coarse rows one per lane | 0.4-0.7 ms |
| C pressure | band register path up to 4096 rows, band capacity bound, fused level transitions and checkpoints, grids sized to resident pages, mgCopy removal | 0.4-0.5 ms |
| D head+storage | storage deletions, blit-run consolidation, grids sized to the bound, census and builder scans, hanging restricted to 4h seams | 0.3-0.6 ms |

Deferred because they change results and need Peter's call:
- a band residual target (−0.4 ms per cycle saved)
- `extensionSweeps` 2→1 (about −0.75 ms)
- an f16 trace-only velocity copy with hardware filtering (about −0.45 ms)
- a surface-volume band derived from phi
- skipping advection of far h vertices

Results are recorded below as they land. (Moved from docs/research, which is gitignored.)

## Results

Pipelined wall in ms per advance, one run per arm (`wall.sh`: warm 45 + 200 measured on fig-9, warm 20 + 150 on dam64).
Two base runs put noise at about 0.1 ms on both scenes, and base fig-9 follows a deterministic tile trajectory (peak h 6679).

| Arm | fig-9 all | splash | tail | dam64 | Notes |
|---|---|---|---|---|---|
| base c38adea3 | 19.43 | 23.64 | 16.39 | 11.41 | |
| base rerun | 19.50 | 23.89 | 16.34 | 11.32 | |
| A1 levelset (8 commits) | 18.59 | 23.25 | 15.25 | 10.31 | **win**; trajectory differs (peak h 7215); represented drift −1.77% vs −2.01% |
| A2 velocity (8 commits) | 20.44 | 24.03 | 17.84 | 10.99 | **loss**, bisected below |
| B volume (7 commits + uniformity fix) | 19.88 | 23.59 | 17.19 | 11.68 | **loss**; transport stage +0.11 ms despite 19 fewer dispatches |
| C pressure (9 commits) | 19.25 | 22.86 | 16.63 | 11.27 | splash win (band coarse-solve register path), tail +0.27 |
| D head+storage (12 commits) | 19.05 | 23.18 | 16.07 | 10.80 | **win**; −71 MB |
| cand1 = A1 + C + D + ae62e5ba + B 6b4ff6c0/ec2ee928/1f59b7e9 | 17.91 | 21.90 | 15.03 | 10.22 | 356 pipelines / 208 dispatched (base 391/220) |
| **cand2** = cand1 + B v2 a887821f/e89c1353 (no tier merge) | **17.73** | **21.84** | **14.78** | **10.11** | **landed** (uncommitted); 350 / 203; represented drift −1.99% vs base −2.01% |

Fenced fig-9 stage profiles (base GPU 22.42 ms) for single-commit arms:

| Arm | Stage effect | GPU ms |
|---|---|---|
| A2 certificate requires fine support (7bbba564+b468068b+d76394c2+ba7a75c9) | census +0.12 | 22.56 |
| A2 face-centre first RK2 sample alone (8c6bade6) | null | 22.41 |
| A2 face-centre + shared texels (8c6bade6+693bb5ed) | momentum +0.26 | 22.79 |
| A2 staged extension words alone (1ff66303) | extension +0.45 | 22.89 |
| A2 restrictBand lanes alone (ae62e5ba) | extension −0.13 | 22.30 |
| A2 umFace width from stencil + authority phi skip (d76394c2+ba7a75c9) | census +0.10 | 22.48 |
| B whole branch | volume coupling +0.11, surface volume −0.05 | 22.49 |
| **cand2 (landed)** | phi −0.60, band −0.19, sharpening −0.19, census −0.17, extension −0.09, SV −0.08 | **20.97** |

Lesson: load count alone is a poor predictor on the M1 Max.
- Staging that replaced a few loads per lane with a barrier and fallback logic lost (extension staged words). So did
  holding more live registers (the 8-texel shared core in momentum).
- Staging that removed thousands of redundant loads per job won (A1 redistance window widths, wall reach).
- Wins came from deleting redundant work (A1), copies and passes (D), and global re-reads in a one-workgroup solve
  (C's band coarse-solve register path).
- Merging two kernel shapes into one launch (B's tier merge) cost more in registers than it saved in launches.

## Landed state

cand2 was copied into the main tree uncommitted (38 files, 37 commits' worth, based on c38adea3).
- fig-9 19.47 → 17.73 ms per advance (−8.9%); dam64 11.37 → 10.11 ms (−11%); fenced fig-9 GPU 22.42 → 20.97 ms.
- Memory 369 → about 301 MiB (D), plus the pressure band bound below.
- Checks: `check:types` clean, naga parses every module on both scenes, `tests/uniform-scratch-layout.test.ts` passes
  (its stale offsets test was deleted per policy), Dawn lane `uniform-mixed-solid-parity` passes.
- Pressure band capacity is now `min(tiles, max(4096, ceil(tiles/2)))` and fails loud on overflow. Fig-9's observed peak
  band is 3582 against a bound of 8192. A scene keeping more than half its lattice as liquid h tiles would now abort.

Follow-ups noted by implementers, not done:
- Delete the dead `mgCopyPressure` / `mgResidual` / `mgDownsampleSubtract` from `UNIFORM_MIXED_CONTINUATION_ENTRIES`.
- Remove the dead L1+ "residual A" arena fields (~0.7 MB).
- Momentum binds `phase` but never reads it.
- `frame.ts` `cleanupGroups[0]` and the `resolved` param can be dropped.
- Orphan dust pass needs a candidate-count measurement; stage-grid copy gating needs a renderer signal.

