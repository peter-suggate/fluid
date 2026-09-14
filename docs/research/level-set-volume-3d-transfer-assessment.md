# Can level-set-plus-volume carry the 3D adaptive-mass surface? A critical assessment

Date: 2026-09-14. Sources: two Opus audits over `lib/methods/adaptive-mass`, `lib/sparse-world`, `rust/crates/fluid-core/src/{world3d.rs, levelset_volume.rs, levelset_surface.rs}`, the docs deleted in 7f6a2cdc (readable via `git show 7f6a2cdc^:docs/...`), Ando and Batty 2020, Aanjaneya 2017, and the 2D lab results docs.

## Verdict

The idea is sound for the stated problem and unproven as a cost story, and the 2D lab has not yet tested the stated problem at all.

- **Sound:** a volume fraction cannot define a seam-consistent surface, for a dimensional reason. f = V/C is dimensionless and the position it implies is (f − ½)·Δx, so equal fractions on the two sides of a seam imply different positions. A distance is scale-free. Ando and Batty are an existence proof that a cell-centred level set on a graded octree gives a single-valued, C0 surface across T-junctions.
- **Untested:** in the 2D lab φ lives on a full-domain vertex lattice at the finest resolution. Both sides of every seam read one field at one bandwidth. The seam metric now routes through φ but measures the first-order Taylor error of a single shared field, not what a coarse cell fails to store. Nothing in the lab has run a deliberately coarse surface over a fine interior, which is the 3D goal.
- **Unsolved:** φ is not coupled to V. On a gently translating blob at Courant about 1, φ-enclosed area fell 41.7 % over 180 frames while V drifted 4e-9. Sharpening is on paper only. Figure 7 over-capacity rises monotonically with the drain on. Half-pool runs 64 % slower than baseline because a φ that disagrees with V raises pressure iterations 1.6×.
- **Already planned once:** the deleted `docs/HANDOFF_FLUID_SURFACE_REVIEW_2026-09-08.md` prescribed exactly this inside the CM12 resident (one f32 φ per fine support, five-cell band, SL advect, Russo–Smereka redistance, sign-test mass coupling, sharpening retargeted to φ, about twenty launches on top of roughly five hundred). WP0 was done; WP1 to WP5 were never started; its oracles are not on disk.

## What the 3D lane does today

- The app default is the `adaptive-volume` lineage (`lib/methods/index.ts:68`), which retires sharpening. `adaptive-mass` is the 18-stage CM12 lineage.
- There is no level set. The published φ is a density proxy, `(0.5 − ρ)·4·cellWidth` (`webgpu-sparse-cm12-resident.wgsl.ts:2224-2228`), inverted exactly by the renderer and contoured at 0.5.
- The inconsistency Peter names is concrete: `cm12PresentationExactSample` (`:9717-9787`) uses a different reconstruction per rung. Fine bricks take the raw per-cell proxy; coarse bricks take a trilinear over volume-restricted means, or a hard φ = −4h when the page is uniformly wet; a column-height reconstruction can pre-empt both. Three surface definitions by rung, packed to f16, and the grid-imprint Dawn gate exists to catch what that produces.
- Coarsening at the surface is allowed today in the shipped `coarse-first` selector (`:7326-7350`, no surface term); only initialization pins the surface sheet fine. So the blocker is not a rule, it is that the density surface has no rung-invariant meaning.
- Ocean census (B8, deleted doc): 137,520 represented cells, 97.3 % in the 1h and 2h surface-and-air band, 97,920 of them dry. The "84 % of pressure cells are the surface layer" figure in memory has no source and should be treated as unverified. Partitions vary by scene: ocean B16 pressure 32 %, mini32 transport 63 %, long-dam transport 84.5 %.
- Constraints: the resident bind group layout is at the ten-storage-buffer ceiling (`webgpu-sparse-cm12-resident.ts:774-790`); the frame is load-issue-bound; there is no 3D redistancing outside the Losasso lane, which is off limits.
- A 3D CPU lab exists: `World3d` runs a full advance (`world3d.rs:599`), but `world3d.rs:142` rejects every transport experiment because the level-set-volume code is 2D only. `extend_velocity_with_level_set` (`numerics.rs:173`) is already dimension-generic and seeds from φ ≤ 0.

## What "consistent across coarseness" requires

A reconstruction that turns data on cells of differing width into one continuous Φ(x) whose zero set does not depend on which cell reads it. Single-valued gives a closed surface; C0 at the seam stops the zero set terminating on one side and restarting displaced on the other. A step of size δ at a seam face displaces the zero set by δ/sin θ, unbounded as the surface becomes tangent to the seam, which is the common case when seams sit on the surface.

- Trilinear on a graded tree is not C0 at a T-junction. Ando and Batty's MLS reproduces trilinear on regular stencils and is C0 except at the edge of its blend region; exact C0 needs inserted ghost samples. Nothing here is C1, and this renderer reads shading normals and curvature from ∇Φ.
- Even an exactly C0 interpolant stores one scalar per coarse cell where the fine side stores eight. The zero set is continuous but band-limited: a ripple shorter than about two coarse widths leaves the seam as a straight line. Ando and Batty's adaptivity smoothing, their blend toward uniform, and their pinned-rung protocol in Figure 13 are three admissions that a 2:1 seam on a free surface is visible. Both papers refine at the surface; using them to justify coarsening at the surface is using them as evidence for something they do not evidence.

So the goal as stated, one φ whose zero set is rung-independent, is achievable. The goal as implied, coarsen at the surface with no visible penalty, is not a well-posedness question but a detail-loss trade that must be measured.

## Two architectures

**A. φ on the adaptive bricks, MLS interpolant (Ando and Batty).** Single-valued, C0, no gap; a detail step wherever the fine side carried sub-coarse features. Removes the stated blocker. Memory negligible (one f32 per accepted cell). New stages: φ advection on the existing trace (cheap), MLS evaluation (a small normal-equation solve per query over an irregular 8 to 30 sample gather, the wrong shape for a load-issue-bound frame), redistancing (no GPU form here; fast sweeping is eight dependent sweeps over the band), extension re-seeded from φ, and pressure membership and θ from φ, which touches compiled topology row assembly. Ando and Batty's own Table 2 puts level-set advection alone at 2.4× their projection, and surface maintenance at 7×.

**B. φ on a narrow band at finest resolution (Aanjaneya).** No seam in the surface at all; the only published architecture that delivers the requirement literally. Pressure and V stay coarse at the surface. Cost shape: band cells scale with surface area times band width, which is the same order as the 1h and 2h band that is already 97 % of the ocean's represented cells. The band leaves the Poisson matrix, which is the real win, but the structure does not shrink. The contradiction: Aanjaneya widen the band so the surface stays inside it after advection and substep the band because one step is too dissipative; Peter wants Courant 5 to 11 on coarse surface cells, which is 40 to 88 finest cells of travel per frame at rung 8. Band width scales with the Courant number the coarsening exists to buy, and a departure point outside the band reads a clamped plateau. Both papers also keep a full-domain φ; the band is an addition, not a replacement.

## Failure modes 3D hides that 2D does not show

- The 2D corner-traced quad gather does not extend: the 3D footprint is a trilinear hexahedron with non-planar faces that self-intersects under shear, the problem that ended the geometric remap. Fall back to a box and the Courant 5 to 10 diffusion returns.
- Redistancing: 2D has an exact BVH over segments at 2 to 3 ms; 3D has no per-frame BVH over triangles.
- The drain gets weaker with coarsening: release is capped at one open capacity per frame, and a rung-8 cell has 512× the capacity but 64× the face area.
- Semi-Lagrangian φ at large dt: Ando and Batty add FLIP particles and Aanjaneya substep specifically to avoid it. No published method advects a level set at Courant 10 with plain SL and keeps the surface. The 41.7 % loss was measured at Courant about 1.
- Thin sheets: 2D loses a broken curve; 3D loses a hole that closes, a topology change, with V keeping mass φ no longer draws.
- V and φ can disagree by a whole coarse cell: at rung 8 a half-width offset against 10 % fill is a 460-finest-cell disagreement in one cell. The 2D hydrostatic root-cause doc shows the mechanism already at rung 8.

## Three strongest reasons it fails

1. **Reconciliation is the whole problem and 2D has not solved it at the easy end.** The proposed fix, per-region multipliers, is a connected-component label plus segmented reduction plus broadcast every frame over a topology-changing set: a new serial dependent chain on a GPU where exact reductions already serialize.
2. **Both papers spend more resolution at the surface; this project wants less.** A rung-8 surface cell stores 1/512 of the numbers. φ changes where blobbiness comes from, not whether it exists. The 2D "no blobs" evidence was measured at uniform-fine φ on a near-static blob against a broken PLIC feedback loop; none of those conditions holds in the target regime.
3. **The cost lands on the stage that is already the frame**, and the band width needed to survive Courant 5 to 11 grows with the time step the coarsening exists to buy.

## What to carry regardless

The hydrostatic rule from the 2D lab: φ is the single free-surface authority for pressure membership and cut-face θ; V is mass only, with a capped expansion source. Evidence is strong (parasitic velocity 5.4e-6 m/s over 300 frames on a mixed-rung scene) and the failure it fixed, V/C membership and φ geometry choosing different surface heights in neighbouring rows, is exactly what a 3D coarse surface will hit. Separately, `presentationIntegratedColumnHeight` is already a rung-invariant surface construction for near-horizontal surfaces without a level set; it narrows the real problem to steep and non-monotone surfaces.

## What must be proven in 2D first

E1. Put φ on the adaptive cells themselves, one per cell, with an MLS interpolant over the cell-centre neighbour set; delete the exact-distance shortcut that launders the fine field back in; contour the adaptive field. Run half-pool and Figure 7 with surface cells pinned coarse (rung 4 and 8) over a fine interior, the inverse of today. Measure per frame:

1. zero-crossing displacement along each seam face, in finest cells (prediction: near zero with MLS, and not the interesting number);
2. detail loss: contour deviation and contour length against the same scene run uniformly fine (the real question);
3. φ-enclosed area versus ΣV per region (today −41.7 % over 180 frames; gate under 1 %, which requires the reconciliation that does not exist);
4. area loss per frame at Courant 5 to 11 on the coarse surface cells, isolating SL dissipation from the seam question.

Gate before any 3D work: item 3 under 1 %, item 2 accepted by eye, over-capacity bounded and decaying. Then E2: architecture B in 2D, a finest band around the contour with the solver coarse at the surface, measuring band cells against accepted cells and what a departure outside the band does. Only then the 3D CPU lab (`World3d`, extending Baseline), and only then the resident.
