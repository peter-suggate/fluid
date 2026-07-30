# Tetrahedral radiance SVO global illumination

## Decision

Use the existing sparse, apron-padded node-mip page topology for a second,
optional **directional radiance atlas**. Each spatial texel stores outgoing HDR
radiance in four fixed directions pointing to the vertices of a regular
tetrahedron. The four RGB values are four filterable `rgb9e5ufloat` 3D
textures that share the opacity atlas's sampler, directory, direct page table,
generation, and physical-page slot.

This is the useful tetrahedral representation for this renderer. Tetrahedron
*shadow mapping* is a different technique: it projects an omnidirectional
point-light shadow map onto four faces. It neither represents bounced radiance
inside an SVO nor reduces camera-side cone work.

The four tetrahedral samples are an invertible basis for a constant plus a
world-space linear directional field—the same four degrees of freedom as
first-order spherical harmonics. For basis directions `d_i` and stored samples
`s_i`:

```text
mean = 1/4 sum(s_i)
vector = 3/4 sum(s_i d_i)
L(direction) = max(0, mean + vector dot direction)
E(normal) = max(0, pi mean + 2pi/3 vector dot normal)
```

The last expression is the analytic cosine convolution for diffuse irradiance.
Negative reconstructed values are clamped because low-order directional bases
ring around sharp emitters.

## Why this layout

| Representation | Payload / texel | Filtered texture reads | Directional degrees | Decision |
|---|---:|---:|---:|---|
| Isotropic RGB9E5 | 4 B | 1 | 1 | Too much light leaking; cannot preserve surface sidedness |
| Tetrahedral RGB9E5 | **16 B** | **4** | **4** | Selected baseline |
| L0+L1 as 3×RGBA16F | 24 B | 3 | 4 | Valid quality fallback, 50% larger |
| Six-axis RGB9E5 | 24 B | 6 | 6 | More directional detail than diffuse GI needs |
| Four packed RGB9E5 in RGBA32U | 16 B | 8+ | 4 | Manual trilinear filtering is wrong for the latency-bound cone path |

`rgb9e5ufloat` is a filterable WebGPU format, so the selected layout retains
the current atlas's hardware trilinear sampling. It consumes four sampled
texture bindings and no storage-buffer binding. The busiest current reduced
split-lighting entry uses ten sampled textures across its groups; adding four
reaches fourteen and remains within WebGPU's guaranteed per-stage limit of
sixteen. RGB9E5 is non-negative HDR with a shared exponent per RGB triplet,
which matches one directional radiance sample.

At the current 10³ physical page size the payload is 16,000 bytes/page. The
recorded 1,715-page garden working set is about 26.2 MiB; the 8,192-page
capacity ceiling is 125 MiB. A candidate plus visible generation can double
that during publication, so radiance must be optional and radiance publication
should retire its previous generation before unrelated topology work starts.

The executable ABI and CPU reference math live in
`lib/svo-tetrahedral-radiance.ts`. They deliberately do not alter the current
opacity atlas or production shader fingerprint.

## Transport semantics

Store **coverage-premultiplied outgoing radiance**. A base texel holding a
Lambertian surface has `coverage * radiance` in tetrahedral directions on the
surface's outgoing side and zero on the back side. Parent construction averages
the eight children's four samples, exactly as the opacity mean lane is
averaged. At a cone sample:

1. Read mean/max opacity from the existing atlas and compute the step-corrected
   opacity exactly as today.
2. Read four radiance lobes at the same atlas UV and reconstruct radiance toward
   the receiver.
3. Divide by mean coverage (with an epsilon/empty guard), multiply by the
   step-corrected opacity, and composite front-to-back.
4. Stop on the existing transmittance threshold. Missing/stale radiance is
   black, while missing/stale opacity retains today's fail-closed exact
   visibility fallback.

Keeping opacity authoritative prevents bright empty-space interpolation and
lets radiance pages be absent until their lighting revision is complete.

## Do not add GI on top of every current cone

The current prepass can launch four AO cones plus up to eight per-light shadow
cones (with extra area samples). Simply adding four to six diffuse GI cones
would multiply the already dominant work. The transport cutover should instead
replace low-frequency visibility work:

```text
today
  per receiver: AO cones + per-light soft-shadow cones + analytic environment

target
  per dirty SVO surface texel: direct-light injection -> tetra radiance mips
  per receiver: 3-4 wide diffuse cones + at most one high-frequency direct shadow
```

The wide cones provide indirect diffuse light and ambient occlusion together;
there is no separate AO term. Direct-light injection bakes area-light
visibility and penumbra into outgoing surface radiance once per dirty world
texel instead of once per screen receiver. Keep one exact/checkerboard contact
shadow for the dominant sun or nearest important light so voxel resolution
does not blur contact detail. Other diffuse direct light comes from the local
injected surface value; analytic lights remain for specular response.

This makes the new system a redistribution of cone work, not an additive GI
mode. It also attacks the continuous-simulation case that exact static-primary
screen reuse cannot accelerate.

## Staged implementation and gates

1. **Storage and injection.** Allocate the four radiance atlases only when GI is
   enabled. Publish them against `(topology generation, material revision,
   lighting revision)`. Inject authored emission first, then shadowed direct
   diffuse irradiance. Validate the garden's warm lamp and a one-sided
   rectangle emitter before enabling bounce light.
2. **Bottom-up filter.** Build apron-complete parent pages from premultiplied
   children. No parent may publish until all eight children contributing to its
   opacity generation are either present or certified black.
3. **Gather-only A/B.** Add 3-4 cosine-weighted hemisphere cones at the existing
   reduced prepass rate. Compare against a path-traced/offline reference for
   energy, leaking, one-sided emission, and temporal stability. Keep the current
   direct/AO output alongside it for attribution.
4. **Work replacement.** Remove AO cones when GI is valid. Replace diffuse
   soft-shadow cones with injected direct exitance, retaining one dominant
   contact shadow and the exact missing-data fallback. The acceptance gate is
   total cone/mip taps and whole-frame time, not GI-pass time in isolation.
5. **Additional bounces.** Reinject the previous complete radiance generation
   through the same wide-cone gather, dirty-page budgeted over frames. Clamp
   albedo below one and track per-generation energy so feedback cannot amplify.

Every stage must report atlas bytes, resident/dirty pages, injected texels,
cone taps, and fallback pixels. Benchmark garden and hose-tank at both moving
and settled camera phases. The existing reduced-rate relative-luminance gates
still apply; add reference-image diffuse irradiance error and a no-energy-gain
test for a closed, unlit scene.

## Research basis

- Crassin et al.'s [Interactive Indirect Illumination Using Voxel Cone
  Tracing](https://research.nvidia.com/index.php/publication/2011-09_interactive-indirect-illumination-using-voxel-cone-tracing)
  establishes the sparse hierarchical radiance-injection and cone-gather
  architecture, including multiple diffuse/glossy bounces.
- Rauwendaal's [Voxel Based Indirect Illumination using Spherical
  Harmonics](https://rauwendaal.net/publications/PhD%20-%20Voxel%20Based%20Indirect%20Illumination%20using%20Spherical%20Harmonics/)
  specifically evaluates low-order directional functions inside a hierarchical
  voxel representation instead of isotropic radiance.
- The [WebGPU texture-format specification](https://www.w3.org/TR/2026/CRD-webgpu-20260109/)
  defines filterability through the `float` sample type; `rgb9e5ufloat` is the
  packed non-negative HDR format used by this ABI.
