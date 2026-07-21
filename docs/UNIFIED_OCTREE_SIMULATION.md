# Unified octree simulation migration

Status: active migration, updated 2026-07-20. U0-U4 have an authoritative GPU
path for supported body-free scenes. Surface evolution uses `2x2x2` compact
pages by default (with explicit `4x4x4` detail), and the dense bootstrap phi is retired after the first submitted
topology build. On that path there is no persistent finest-cell 3D simulation
field. U5 covers unsupported scene features and capacity
right-sizing, not a second dense authority for the supported path.

The target is one GPU-resident adaptive MAC octree. Pressure, velocity, liquid
surface, solid apertures, diagnostics, and rendering must consume the same leaf
and face topology. Finest surface leaves are part of that topology, not a
second dense simulation tier.

## Non-negotiable invariants

1. A physical face fragment is stored exactly once and has a canonical
   negative-to-positive orientation.
2. A balanced coarse face touches at most four fine fragments. Divergence uses
   `sum(area * normalVelocity)` and the pressure gradient updates those same
   normal-velocity records.
3. Refinement prolongates boundary flux conservatively; coarsening restricts
   by face area. The next pressure solve removes interpolation divergence.
4. Signed-distance samples exist only on finest interface leaves plus their
   swept stencil halo. Deep liquid needs no fine velocity or phi transfer.
5. With overlays disabled, no persistent 3D resource may scale with the
   logical finest-grid box.
6. Rendering reads the simulation topology directly. Dense diagnostic images
   are transient, on-demand products.

The face invariants are live. Invariants 4 and 6 hold for the supported body-free
path: it samples compact pages after bootstrap, destroys the dense phi
publication, omits the compatibility sparse-brick world and full mirror atlas,
and renders both the water surface and grid overlay directly from adaptive
buffers. Owner payload capacity is now derived from pressure/surface bounds.
Invariant 5 now holds for the supported body-free direct-page path. Its
brick-state, topology-tile state, and paired active/retired scheduler streams
are transactional open-addressed key pools bounded by compact surface-producer
and interface-area capacities. Pool saturation rejects the candidate generation
and preserves the last complete publication. The owner lookup is likewise a
capacity-bounded sparse hash, and compact scan/coarse-task scratch follows
pressure-row and active-tile bounds rather than the finest box. Terrain, rigid-body, special
diagnostic and compatibility modes retain dense resources until their
operators have page-native samplers.

The executable topology oracle is `lib/octree-face-fragments.ts`. It defines
canonical ownership, the four-fragment 2:1 bound, restriction, and the signed
leaf-incidence reduction used by divergence. Runtime WGSL follows this
contract.

## Sparse-field architecture decision

The end state is not a dense level set hidden behind a sparse lookup table.
It is a signed sparse hierarchy:

- inactive liquid tiles return a constant negative band value;
- inactive air tiles return a constant positive band value;
- numerical phi samples are active only in a swept, two-sided interface band;
- surface, velocity, meshing, rendering, and diagnostics share one spatial
  index rather than publishing parallel dense or brick worlds.

This follows the established VDB level-set model: positive inactive exterior,
negative inactive interior, and a thin active signed-distance band. NanoVDB
demonstrates that the hierarchy is suitable for GPU simulation and rendering;
fVDB's single GPU index-grid design reinforces sharing topology across many
operators instead of maintaining independent lookup structures.

The current canonical surface representation attaches a `2x2x2` or `4x4x4`
page to each resident finest interface/halo leaf and uses bounded cross-interface
lookup. Inactive regions obtain their sign from the adaptive hierarchy. The
supported body-free path therefore needs neither a persistent dense phi
publication nor a parallel sparse-brick simulation world after bootstrap.

The coarse level-set schedule owns 65 invocation-stable parameter slots: one
cold bootstrap plus the solver's bounded 64 surface substeps in a single
command encoder. Production submits that encoder before `advanceTo` returns
and explicitly retires it before another encoder may use the schedule; the
runtime rejects either a second unretired encoder or a 66th invocation. This
is the supported product contract, not a general concurrent-encoder ring.
Fine-to-coarse restriction, topology finalization, and fine transport still
have one mutable parameter allocation each, so more than one surface substep
in the same unsubmitted encoder remains a known lifetime gap. The bounded
default dam-break path uses one surface substep; lifting that bound requires
the same invocation-stable allocation treatment before it is authoritative.

Admitted power authority now selects the paper's Section 4.3 hybrid PCG
preconditioner. It captures the Cartesian/GFM first-order rows before power L2
publication, applies paired `k=8` second-order smoothing near boundaries and
level transitions, runs the sparse SPD first-order Galerkin V-cycle, and then
applies the matching eight post-smoothing iterations. The localized band is
currently three compact L2 graph rings, an explicitly diagnosed approximation
of the paper's "about 3 voxels" physical band. Aggregate PCG and Chebyshev are
explicit compatibility modes and are not permitted to masquerade as Section
4.3 authority.

Research references:

- OpenVDB overview, especially signed inactive tiles and the active narrow
  band: <https://www.openvdb.org/documentation/doxygen/overview.html>
- Museth, *NanoVDB: A GPU-Friendly and Portable VDB Data Structure for
  Real-Time Rendering and Simulation* (2021):
  <https://research.nvidia.com/labs/prl/publication/nanovdb/>
- Williams et al., *fVDB: A Deep-Learning Framework for Sparse, Large-Scale,
  and High-Performance Spatial Intelligence* (SIGGRAPH 2024):
  <https://research.nvidia.com/labs/prl/publication/williams2024fvdb/>
- Wang, Sun, and Zhu, *Matrix-Free Multigrid with Algebraically Consistent
  Coarsening on Adaptive Octrees* (2026):
  <https://arxiv.org/abs/2604.18886>
- Jaber, Essel, and Sullivan, *A GPU-native adaptive mesh refinement approach
  with application to fluid flow* (2025):
  <https://www.sciencedirect.com/science/article/pii/S0010465525000463>
- Wang et al., *Cirrus: An Adaptive Hybrid Particle/Grid Flow Map Method on
  the GPU* (SIGGRAPH 2025): <https://wang-mengdi.github.io/proj/25-cirrus/>

## Cutover status

### U0 — canonical face contract (complete)

- CPU oracle for complete dyadic leaf tilings.
- Unique physical face publication.
- Four-fragment coarse/fine bound.
- Conservative area restriction and equal/opposite incidence tests.
- Production pressure-capacity planning imports the same bound.

### U1 — deterministic GPU face topology (complete)

- Builds canonical faces from the persistent liquid-leaf frontier with a
  deterministic count/scan/emit sequence.
- Stores bounded signed row-to-face incidence: at most four fragments for each
  of six leaf sides, or 24 entries per pressure row.
- Uses a default capacity of four faces per reserved pressure row and fails
  closed on overflow.
- Preserves velocity across topology changes with exact lookup, coarse-face
  prolongation, or equal-area restriction from four fine faces. Production
  does not retain the optional per-face audit records.
- Keeps on-GPU RHS and projection parity reductions for validation without
  requiring dense CPU readback.

The earlier atomic mirror was a scaffold. It has been replaced by the
deterministic topology above; the face store is now the input to the
authoritative U2/U3 path rather than a permanent dense duplicate.

The 2026-07-19 Dawn/Metal dam-break gate passed for both mirror-only and
face-RHS-authoritative 50-step runs. The authoritative run ended with 50,690
published faces, 15,686 compared liquid rows, zero mismatches, no overflow,
maximum absolute RHS error `3.96e-9`, no WebGPU validation errors, and all
smoke invariants passing. Reproduce the parity gates with
`npm run test:webgpu:octree-face-parity` and
`npm run test:webgpu:octree-face-rhs`.

### U2 — adaptive projection authority (complete for supported scenes)

- Assembles compact pressure RHS/divergence directly from canonical face
  velocities.
- Solves the existing compact leaf pressure system.
- Applies the matched pressure gradient directly to the same canonical face
  records.
- Uses deterministic face incidence ordering and validates face-derived RHS
  and projected velocity against the compatibility path.
- Reapplies the solid normal constraint after projection when adaptive rigid
  faces are present.

Authority is deliberately fail-closed. Terrain and the experimental
hydrostatic split still use the dense-compatible path. Moving rigid bodies are
supported only through the adaptive solid-face module and without terrain.
The compact path is selected by the octree `faceVelocityTransport` option; the
smoke harness exposes it as `FLUID_OCTREE_FACE_TRANSPORT=1`.

### U3 — adaptive transport authority (complete for the same cutover)

- Owns double-buffered scalar normal velocity on canonical faces.
- Advects faces with a bounded, resolution-aware hierarchical sampler.
- Applies acceleration/gravity and domain boundary constraints on the compact
  worklist.
- Reduces maximum component CFL, maximum speed, non-finite count, and processed
  face count into a pooled 16-byte diagnostic readback.
- Transfers velocity conservatively across every deterministic topology
  rebuild.
- Publishes transported values directly to the face records consumed by U2.

When the authority guard succeeds, the host solver no longer allocates or
dispatches the box-sized velocity A/B/C/D, padded transport A/B, flux,
pressure A/B, volume A/B, or sharpening fields. Shared bind-group layouts keep
only format-correct `1x1x1` placeholders. Initial dense VOF upload and retired
dense pipelines are also skipped. Solver texture getters and captures do not
expose the placeholders as simulation fields.

`lib/octree-consumer-sampling.ts` defines a binding-neutral adaptive velocity
and surface sampling contract for downstream consumers. Transport, water mesh
extraction, optical presentation, and the octree grid overlay consume the
adaptive reconstruction directly. Spray and specialized diagnostic consumers
remain to be moved to this shared ABI.

### U4 — adaptive surface authority (complete for supported body-free scenes)

- Converts live compact leaf rows into leaf records and interface/swept-halo
  page candidates on the GPU.
- Allocates leaf-attached `2x2x2` phi pages by default, or explicit `4x4x4`
  detail pages, from a bounded pool with
  lifecycle, free-list, and overflow diagnostics.
- Runs page transport, redistance, and smooth-volume reduction/correction over
  indirect compact worklists.
- Suppresses publication and fails the smoke gate if the page pool overflows.
- Enables pages by default whenever compact face transport is authoritative;
  `FLUID_OCTREE_SURFACE_PAGES=0` is the diagnostic opt-out.
- Uses a 32% resident-page budget, chosen to cover the target ocean scene with
  headroom while keeping memory proportional to compact leaf capacity.

The page arena is authoritative for surface evolution. A dense `r32float` phi
texture is uploaded only to bootstrap the first topology and page population.
Direct paged topology is the default; setting
`FLUID_OCTREE_DIRECT_PAGED_PHI=0` selects the diagnostic compatibility path.
After the bootstrap submission, supported body-free scenes rebind subsequent
work to the compact page arena and destroy the dense texture. Topology resolves
cells through the live-leaf hash, samples resident pages at their configured resolution, and uses
the leaf's affine phi plane outside the fine band.
Unsupported terrain/rigid-body and explicit diagnostic modes retain dense
bindings. The old dense surface scratch, predicted, reversed, two jump-flood
seed arenas and topology snapshot are absent on the supported path; their
format-compatible bindings receive `1x1x1` fallbacks.

### Solid and rigid-body slice (partially complete)

- Classifies canonical adaptive faces against rigid bodies and stores compact
  open fraction, solid normal velocity, dominant owner, and sample mask.
- Applies finite-volume no-penetration constraints before projection and locks
  the solid component again after projection.
- Accumulates paired pressure reactions into a private, fail-closed impulse
  arena and publishes them to the existing rigid exchange.

Partial cut faces currently use an immersed-flux approximation: pressure
matrix coefficients are not yet weighted by aperture. Terrain has not been
ported to canonical faces, and combined terrain/rigid scenes remain on the
compatibility path.

### U5 — remaining unification work

- Replace the solid-cell field in supported solid scenes. The body-free
  scheduler, owner payload, and sparse key/value lookups are now
  pressure/surface/interface-area bounded and fail closed. A producer-published
  refined-brick count would permit a tighter exact scheduler bound.
- Port spray emission, raw-voxel inspection, and remaining specialized
  diagnostics to compact leaf/face/page sources. Water mesh extraction,
  optical rendering, and the grid overlay are already direct adaptive
  consumers.
- Finish aperture-weighted pressure assembly and terrain face apertures.
- Validate or port specialized inflow/outlet behavior on adaptive faces.
- Remove dense compatibility publications from terrain, rigid-body, and
  explicitly requested diagnostic paths as their consumers gain adaptive
  samplers.

## Current memory result

The 2026-07-20 `ocean-seiche` gate at `320x96x80` measured
`305,153,604` allocated bytes on the supported unified path versus
`1,180,390,700` bytes for the dense baseline: a reduction of `875,237,096`
bytes (about `74.15%`). The compact run omits the persistent dense phi, compatibility
sparse-brick world, full mirror atlas, dense frontier map, and dense renderer
publication. Peak startup memory still includes the transient dense phi upload;
steady-state body-free memory does not.

The default `2x2x2` surface-page planner, including adaptive sparse-owner capacity,
reports `206,489,700` compact auxiliary bytes at `320x96x80` and
`826,743,900` bytes at `640x192x160`. Relative to the modeled dense fields,
the net changes are `-267,463,052` and `-2,956,494,196` bytes respectively.
Explicit `4x4x4` pages remain a quality option. Small-scene overhead is
accepted deliberately: the target is large, deep water domains, where removed
box-sized fields dominate.

Compact scan and coarse-task scratch at `640x192x160` is now `439,076` bytes,
down from the `1,286,876`-byte compatibility allocation (`65.9%` removed).
The body-free direct-page scheduler also omits its unused identity and leaf-state
mirrors. At `320x96x80`, its full transactional A/B allocation is now `147,224`
bytes versus `194,504` for the former logical-brick/tile arrays, saving `47,280`
bytes. At the small `60x45x40` dam default the interface-area budget clamps to
the complete 240-brick/8-tile domain, so sparse keys cost `1,984` extra bytes;
this is accepted for strict box-independence and is explicit in allocation
accounting rather than presented as a small-scene memory win.

The final 50-step dam gate passes as exactly 50 encoded steps with 27,555
published faces and 8,272 liquid rows, maximum incidence 6, no capacity
overflow, peak component CFL `0.4066`, zero non-finite values, clean WebGPU
validation, and all smoke invariants passing. Compact-face runs keep the full
32-pass Chebyshev budget until residual adaptation is moved fully onto the GPU;
this removes a former timing-dependent instability without restoring dense
velocity extrapolation work.

The three-step max-leaf-32 ocean gate publishes 93,244 adaptive rows, 310,689
canonical faces, and 93,184 live surface pages. Maximum incidence is 9 of the
bounded 24, with no face, pressure, frontier, surface-page, non-finite, or
WebGPU validation overflow. Owner and face consumers now share the sparse owner
hash; no consumer decodes it as a dense logical-brick table.

No per-step bandwidth factor is claimed yet. Compact transport and direct
adaptive rendering remove the known volume-scaled field passes, but the
required traffic reduction has not been measured end to end. The 3x traffic
gate remains an exit criterion rather than a current result.

Small scenes may use more memory because compact arenas are capacity-planned
and retain fixed metadata. The optimization target is large, deep water
domains, where removed box-sized fields dominate.

## Acceptance gates

These are final migration gates, not claims about the current partial cutover:

- No persistent box-sized 3D allocation in the octree method with diagnostics
  disabled.
- At least 4x lower fluid working-set allocation on the deep-ocean benchmark.
- At least 3x lower estimated field bytes moved per step.
- Less than 15% whole-step growth per ocean-depth doubling.
- Seiche period within 2% and amplitude within 5% of the accepted reference.
- Existing volume, divergence, energy, dam-break, rigid-coupling, and long-run
  stability gates pass.

The default-off hydrostatic split is not a prerequisite. Its current discrete
form failed the seiche parity gate because fine-face forces and the
nonconforming pressure basis did not form a matched operator. U2 establishes
the matched adaptive divergence/gradient pair for absolute pressure; a
well-balanced hydrostatic mode can be evaluated on that representation as a
separate follow-up.
