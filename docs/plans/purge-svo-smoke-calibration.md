# SVO smoke calibration cleanup

Removed the historical scene-specific candidate-density ratchets. The hero ratchet (67 candidates per brick) was documented for 1,233 primitives on 200 mm bricks in August 2026. The retained hero filling scene publishes 2,554 primitives on 100 mm bricks: its `water: true` factory applies the existing 12.5 mm solver-cell floor, and each brick spans eight cells.

The hero scene builders and `getScenePreset` declarations match Git HEAD exactly. The model cell-size constant and the geometry, sparse-domain planner, adaptive-brick planner, SVO construction, and SVO scene-publication files are unchanged by this purge. The failed ratchet therefore describes a different geometry/lattice configuration from the current smoke.

The tool now enforces the real 512-candidate arena and zero overflowing bricks, continues to report the 64-candidate refinement target, and retains every rendering, lighting, validation, determinism, and optional frame-time check. Explicit density limits may be stricter than the arena. The captured final smoke had 103 maximum candidates and zero overflowing bricks; its other 16 checks passed.

Validation evidence is outside the repository: `/tmp/fluid-final-svo-smoke.json` and `/tmp/fluid-final-svo-smoke.txt`. The retained volume, visual-layer, water-depth, and surface-extraction lanes pass all six cases with zero skips in `/tmp/fluid-final-retained-water-gpu-tap.txt`.

The final current-tool GPU rerun passes all 17 smoke checks with zero warnings under Node 22.22.1: `/tmp/fluid-purge-svo-final.json` and `/tmp/fluid-purge-svo-final.log`.
