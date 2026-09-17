# Quarter pool: frame-5 coarsening

Reproduced on 2026-09-17 with the live `coarse-first-pool-impact-quarter`
scene document, native 2D `LevelSetVolume`, dt=1/30, 256 pressure iterations,
and relative tolerance 1e-6. No solver policy changes were made.

## Measured sequence

| Frame | Published cells | Decision |
| --- | ---: | --- |
| 0 | 296 | Authored initial topology |
| 1–2 | 352 | Initial support expansion |
| 3–4 | 88 | First certified demotion: ball B8 to B4 |
| 5 | 38 | Second certified demotion: ball B4 to B2 |
| 6 | 48 | Projected support expands before transport |
| 7 | 192 | Projected support raises the grid to B4 before transport |
| 12 | 360 | Thin-feature protection requests B8 near the ball/pool gap |

At frame 5, ball bricks 29, 30, 33 and 34 consume fresh B2 certificates
after the second qualifying topology epoch (frame 4 had proof count one).
Their plan reason is 16 (demotion). Lower-ball velocity activity is about
0.2732 finest spacings; upper-ball activity is about 0.0014. With energy
threshold 8, dt=1/30 and finest spacing 0.2 m, B4 motion demand starts at
1/3 of a finest spacing per step. None of these bricks is marked thin.
The material floor is B2: four finest spacings, or 0.8 m per solver cell.

The surface displacement tolerance is one finest spacing (0.2 m). The proof
tests virtual bilinear restriction of phi; it does not require the blue volume
fractions to reproduce the yellow contour at the same resolution. Actual phi
remains on the independent fine vertex grid. Existing volume/phi disagreement
is explicitly excluded from this restriction certificate.

The frame-5 transport runs on 88 cells; demotion occurs at final publication.
Frame 7 instead starts with 48 cells, has 48 through primary pressure projection,
then 192 at projected-support pressure projection and conservative transport.
Its final resolution receipt is retention, so that receipt alone cannot explain
the promotion. Support sizing uses donor motion/thinness and receiver
compatibility, followed by 2:1 closure. At frame 12, the final planner explicitly
marks bricks 25, 26, 29 and 30 thin and promotes them B4 to B8.

## Comparison with GPU 3D

The active coarse-first paths share velocity-spread/closing-flow sizing,
generation-stamped next-rung proofs, a configurable qualifying-epoch count,
the four-spacing material floor, and immediate finer demand. Neither requires
uniform falling motion to stop before demotion. The legacy eight-epoch rule
does not govern this coarse-first branch.

Differences still needing matched tests:

- 2D certifies vertices and half-grid samples. GPU 3D checks finest cell-centre
  samples and corresponding edge-crossing displacement, with separate support
  availability checks. Their sampled representability guarantees differ.
- 2D surface classification uses direct phi crossings; the GPU census gates
  its surface-axis classification on occupied material. Thin geometry has a
  separate density-independent veto in both.
- Projected receiver preparation and final demotion are separate decisions.
  Retention must be compared at both stages, not only in published receipts.
- GPU 3D's legacy thin floor retains current resolution, but its subsequent
  coarse-first safety floor overrides this with finest resolution. The active
  2D coarse-first path also requests finest resolution. Comments describing
  retention alone do not describe either active coarse-first path.

This comparison is source-based for GPU 3D; no new Dawn run was made. A matched
3D trace is needed before claiming identical frame timing or dynamics.

## Reproduction

Export `{ "scene": sceneDocument(findSceneDefinition(
"coarse-first-pool-impact-quarter")) }` from the live TypeScript catalog to a
JSON file, then run:

```sh
cargo run --manifest-path rust/Cargo.toml -q -p fluid-core --example investigate_2d_levelset -- /tmp/fluid-quarter-pool.json 14
```

The probe now accepts an optional scene JSON file and frame count, and records
brick coordinates/rungs and observer-stage cell counts. The recorded run is
`artifacts/level-set-volume/quarter-pool-coarsening-frame0-14.json`.
