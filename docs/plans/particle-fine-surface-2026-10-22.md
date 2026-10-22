# Fine particle surface: a cosmetic liquid skin meshed from NB-FLIP particles (2026-10-22)

## Why

The Speed view shows detail the level set does not: thin sheets, crown rims,
ripples and the leading edge of the dam-break front. The level set lives at h
and 4h and is smoothed, so it loses that detail. Motion paints energy, not
geometry. This is a new particle display, **Fine**. It meshes a liquid skin
straight from the particles every frame. Where particles cluster densely they
become a surface. Particles that are not enclosed by it are drawn as spray: small
translucent dots. It is presentation only. Physics and the level set are
unchanged.

## Prior art

- Zhu & Bridson 2005, *Animating sand as a fluid*: φ(x) = |x − x̄(x)| − r̄,
  where x̄ is the kernel-weighted mean of nearby particle positions. A flat lattice
  gives a flat surface, not the bumpy look of summed blobs. That matters here
  because Motion already found that the seed lattice strobes as it is advected.
- Yu & Turk 2013, *Reconstructing surfaces of particle-based fluids using
  anisotropic kernels*. Per-particle covariance stretches kernels along sheets.
  It gives the best thin sheets, but costs a neighbour search per particle. It is
  the follow-up if Zhu–Bridson sheets read too thick.
- Akinci et al. 2012 and Wu et al. 2018 (GPU surface reconstruction): build a
  sparse grid only around the surface, then use marching cubes or surface nets.
  This design follows the same structure.
- Screen-space fluid rendering (van der Laan et al. 2009; the narrow-range
  filter of Truong & Yuksel 2018) was rejected. It is not a mesh, it cannot feed
  the water's interface passes, and so it would not inherit Shade, Simple and
  Wire.

## Design

**Grid.** Fine samples at h/2, which is the NB-FLIP seed spacing (8 particles
per cell, radius 0.22 h). Bricks are 8³ fine cells (4h). Bricks live in a fixed
pool. A GPU hash of brick coordinates maps each one to a pool slot. The pool is
rebuilt every frame from the particles, so the grid has no bounds and keeps no
state between frames.

**Per-frame chain.** Every pass is compute with a fixed direct dispatch that
grid-strides over its GPU list, so there are no indirect launches.
1. *claim*: each live particle inserts the keys of the bricks its kernel
   footprint touches. The footprint is dilated one sample toward −xyz, so that
   every sign-changing edge has an owner.
2. *compact*: occupied hash entries take pool slots. The hash stores keys only
   and slots are assigned afterwards, so there is no spinning on a half-written
   value.
3. *count*: each particle's home brick gets `atomicAdd` for its rank.
4. *scan*: one workgroup computes the exclusive prefix of the brick counts.
5. *scatter*: particles go into brick order as (fine position, depth key) plus
   the record index.
6. *mesh*: one workgroup per brick. It gathers its 27 neighbour lists and splats
   W, ΣwΔx and Σw·depth into 10³ samples with workgroup atomics
   (fixed point, so the sum is the same in any order and seams are exact). It
   then forms φ, places one surface-nets vertex per cell over −1…7 and emits
   the quads of the edges it owns. A quad goes out as two non-indexed triangles
   in the water's own `SurfaceVertex` layout. The brick reserves its output with
   one device `atomicAdd`. It also writes φ at each of its home particles.
7. *finish*: writes the indirect draw count, clamped to the pool, and the
   overflow flags.

**Greed.** An isolated particle would mesh as a sphere under Zhu–Bridson. So
the radius is gated by the summed kernel weight W:
r_eff = r · smoothstep(W_min/2, W_min, W). W_min falls as Greed rises. Clusters
below W_min become spray instead of blobs.

**Union with the body.** NB-FLIP keeps particles only in the 4h band, and in
Mixed Uniform only in h tiles. So a lack of particles does not mean air. A
particle-only skin is a shell with a floor, voids between sparse particles, and
a stair-stepped wall where h tiles meet 4h. Every one of those surfaces is a
back face inside the body. It became the back interface, which cut the Simple
murk path to almost nothing: the skin region looked pale. The wall's top was
also a visible step (2026-10-22 screenshot).

Each record carries its depth under the level set (float 7; ballistic records
count as 2 h in the air). The splat sums Σw·depth, so D = Σw·depth / W is the
level set's value at the particles' weighted mean. Carried back to the sample
along n = ∇D/|∇D|, this gives φ_L = D − m·n, with m the mean offset. The field is
φ = min(φ_particles, φ_L), so the floor, the voids and the region under the tile
wall are liquid. The estimate exists only within the particles' support. An edge
that leaves the support from a sample that is liquid by φ_L alone is the edge of
the particle coverage, so it is not emitted. A genuine particle surface at the
support's edge is still kept. On the synthetic slab, side faces went from 908
(a wall from 3.25 to 6.1 cells) to 64 (the rim above 5.6), with no floor
faces. The top skin sits 0.15 h above the level set, inside the default Merge.
∇D is one-sided at the window's border, so two bricks can place a shared vertex
slightly differently where φ_L decides it. That happens only where the skin meets
the level set, which Merge hides.

The skin is drawn *in addition to* the level-set mesh, in the same front, back
and rear interface passes. The nearest front face wins, so the skin replaces the
smooth surface wherever it lies on or above it. Because the skin goes through
the interface passes, Shade (refraction), Simple (murk) and Wire all apply to it
unchanged.

**Merge.** Where the skin hugs the level set it adds only lumps and z-fighting.
Each skin vertex carries the Merge distance (in cells, default 0.5 h) as
−position.w; level-set vertices carry 1. The water's surface raster moves a skin
vertex that far back along its own view ray. Pixel coverage is unchanged, but its
depth grows by exactly that much, because the projection is linear in view depth.
So the level set wins the depth test wherever the skin stands less than Merge in
front of it, and the skin shows only where it protrudes: sheets, crowns and the
front. It costs no extra work. The stored position and normal stay true, so
shading is unchanged. The wireframe draws the whole skin.

**Spray.** In the Fine view the particle overlay reads φ at each particle and
draws only particles outside the skin (φ ≥ 0). Each is drawn exactly as a Motion
sample: the same energy, white tint and unlit footprint, and the same
order-independent sum and composite. Its footprint is the Spray size times
Motion's (2/3 by default) and its opacity is scaled by Spray opacity (1 by
default). Spray behind the water front is hidden in every surface mode.

**Controls** (the Fine row, URL-persisted, live, uniforms only):
Greed, Smooth (kernel radius, 0.5–1.5 h), Thick (surface radius), Merge (0–2 h), Spray size,
Spray opacity.

**Memory.** This is allocated once when Fine is first chosen, never per frame:
32,768 bricks, a 65,536-entry hash, a 2M-vertex pool (64 MiB) and 36 B per
particle record. Workgroup memory is 30,188 B per 512-lane brick workgroup. M1 grants
32 KiB. The pass refuses to build, loudly, if the device grants less storage or
fewer lanes.

## Measured cost

Synthetic band on M1 Max: 131k band particles plus 20k spray, 2,257 bricks,
469k skin vertices. Wall time per rebuild, best of four 10-rebuild submits:

| Step | Rebuild |
|---|---|
| first build | 5.2 ms (splat 2.4, quad emission 2.4) |
| emission reads a per-cell packed vertex and normal, with no dynamic local arrays | 2.95 ms (emission 0.08) |
| splat walks only the sphere's row spans | 2.94 ms |
| the 27 neighbour lists are one flat range (one list kept about 70 of 256 lanes busy) | 2.29 ms |
| mesh workgroup widened to 512 lanes (1024: 2.42 ms) | 1.88 ms |
| union with the depth-record level-set estimate (∇D at every sample) | **2.15 ms** |

Knock-outs at 2.96 ms: removing the splat's atomics left 0.60 ms, so the splat
was about 2.35 ms. Spreading the lanes so that they hit different addresses
made it slower (3.1 ms), so contention was not the cause. The cause was idle
lanes and too little latency hiding: the 30 KB window fits one workgroup per
core. Speed of light for the splat is about 50M workgroup atomics at 218 G/s,
or 0.25 ms, so it is still about 5x off. The chain is skipped while paused
unless a setting changes.

## Follow-ups

- Anisotropic kernels (Yu & Turk) for thinner sheets.
- Laplacian smoothing of the skin (splashsurf's weighted variant).
- Whitewater tint on spray from the Motion energy.
- Splat speed of light. The window's memory caps occupancy at one workgroup
  per core. Candidates: pack W and the offsets into fewer words, or split
  bricks into two half-windows.
