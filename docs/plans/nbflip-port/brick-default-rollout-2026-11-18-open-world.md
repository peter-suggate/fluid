# Open worlds on the brick engine without the moving window (2026-11-18)

This note covers the rollout plan's last gap row, "open world: the moving window | 3 | E, M6"
(brick-default-rollout-2026-11-17.md §1). It affects nbflip-open-ground, -faces and -slope.

## What the scenes need

A released column spreads over open ground. It needs room, not a moving window. The front runs at about
2√(gH) ≈ 12.5 m/s (H = 4 m), so in the scenes' 6 s it reaches about 80 m from the centre. The old engine holds it in
a 64-page window with the box at page 31. On flat ground its page pool is exhausted at frame 295.

## The brick setup (brick-scene.ts brickOpenWorldSetup)

- **Lattice.** The lattice is the old engine's box scene (sceneFluidBoxScene) widened by
  `fluid.brickTiers.openWorldMargin` pages a side on x and z. The default is 4; the open-ground scenes author 10.
- **Walls and ground.**
  - The lattice's sides are the box rule's walls.
  - The floor is the box's, which is the window's.
  - The ground is the box scene's analytic terrain over the widened columns.
  - The far φ is the old box's.
- **REACH latch.** Liquid in the outer x or z brick layer latches FAIL.REACH, as the open top's last layer does
  (setMark, info bit 31). It never meets a side wall the old window does not have.
- **Frame.** origin_m is the box scene's low corner plus sceneFluidBoxShift, so the lattice sits in the scene's own
  frame.
- **Refused explicitly.** Voxels and sources (the box scene refuses them), rigid bodies in an open world, and a
  free-space boundary. A live voxel edit on an open world is also refused.

## Measured (2026-11-18, 360 steps)

| run | result |
|---|---|
| 4 pages | REACH at frame 189 (flat), frame ≈190 (slope) |
| 10 pages, generous pools | all 360 steps, fail 0 |
| | F 6635 / 6203, C 277 / 297, particles 289k / 269k (flat / slope) |
| | median 30 / 37 ms over the growing flood |

The authored pools are 1.5× those peaks.

**Drift, as the engine's own gauge against its budget:**
- Slope: about +1%. The smoke's presentation volume reads +15%, but it counts liquid φ inside the terrain at open
  fraction 1. That is a measurement artifact, not a mass change.
- Flat ground: +3% from step 45, against the old engine's +0.7% at step 60. It reaches +8% at 6 s, when the film is
  about 0.14 cell thick.

## Left for M6

- The presentation is lattice-sized: three r32float 3D textures, about 380 MB at 704×64×704. It should become
  page-sparse.
- A truly unbounded world (free-space boundary, bodies or sources in an open world) still needs the moving window.
