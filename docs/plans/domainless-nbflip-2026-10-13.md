# Domain-less narrow-band FLIP on 32³ pages

2026-10-13. Review and plan. Implementation started the same day; the section
below records where the implementation departs from the plan as first written.

## Refinements adopted for implementation

Acceptance target: the letters scene (`nbflip-figure-8-letters`, 256×192×128,
193 steps at 1/24 s) at no significant loss of frame time. Baseline before any
change: 31 ms median wall per step, 28.8 ms GPU, about 1.0 M particles on
average, 2.46 GB allocated.

1. **Page table instead of neighbour tables and a hash.** Kernels keep their
   integer cell, tile and vertex coordinate arithmetic. Coordinates live in a
   toroidal window of fixed power-of-two size (target 2048 cells, 64 pages per
   axis), and a `64³` u32 table maps a page coordinate to `slot + 1`, or 0 for
   absent. There is no hashed directory and no per-page neighbour record. Two
   resident pages that alias in the window are fatal. This replaces "Finding a
   page" (design section 3) and the neighbour slots in section 2.
2. **Home page fast path.** Each kernel invocation resolves its home page
   once. A tap that stays in the home page costs shifts and adds; only a tap
   that leaves it reads the page table. This is how the per-tap lookup cost
   measured in September is avoided.
3. **Three kinds of page: resident, AIR and SOLID.** The first plan was to
   author tank walls as solid voxels and delete array-edge wall physics. That
   was measured on letters with the walls authored in a padded lattice (16
   cells each side, two repeats per arm, 193 steps at 1/24 s): 52.8 ms median
   against 31 ms (1.71×), 4.04 GB against 2.46 GB, and a different settled
   state (volume drift +1.0% against −1.0%, residuals about 3× higher). The
   cause is that Dynamic promotes every wet tile touching a solid voxel, and
   its 26 neighbours, to h (about 17,000 extra h tiles), whereas a wall at the
   array edge is owned by every width and stays on 4h owners. So:
   - A page-table entry is a slot, AIR or SOLID. AIR and SOLID pages have no
     storage.
   - AIR reads as ambient. Liquid reaching toward an AIR page requests it.
   - SOLID reads as solid everywhere. A face between a resident tile and a
     SOLID page is exactly today's array-edge wall. The array-edge physics is
     kept and re-keyed from "outside the box" to "the neighbour page is
     SOLID", not deleted.
   - Tank floor, walls and lid are authored one page thick, so the pages
     around a page-aligned box are SOLID and cost nothing. Letters stays at
     192 resident pages. Solids that are not page-aligned (terrain, obstacles)
     remain voxels in resident pages on the existing cut-cell path.
4. **Order of work.** The phases below are regrouped into five increments,
   each of which leaves the solver running with one code path:
   1. *Seam.* Every lattice address, bound test, clamp and dense extent in
      `lib/methods/uniform` goes through a small set of named address
      functions (`uniform-lattice-address.ts`) whose bodies are still box
      arithmetic. Array-edge physics is tagged `UM_EDGE` / `umEdge*`.
      Behaviour and frame time unchanged.
   2. (Dropped as a separate step by the measurement in item 3.)
   3. *Pool.* The address function bodies switch to page-table addressing with
      a static resident set, and the tagged edge physics is re-keyed to the
      neighbour page's kind. The box-shaped algorithms are converted one by
      one: pressure root hierarchy, extension nearest-source pyramid, line
      sweeps and the summed-volume table, the dense solid mask, the particle
      bin order.
   4. *Dynamic.* GPU request, claim, initialise and retire
      (`uniform-page-pool.ts`); solids sampled per page; resident-list
      launches; the renderer reads pages.
   5. *Parity push* on letters.
5. **Done so far.** The three indirect particle launches are fixed direct
   launches and mandatory-seed overflow of the particle pool is fatal. The
   page allocator exists as a standalone module with a Dawn probe and is not
   wired in yet.

### Revision, 2026-10-13 afternoon: correctness first

Priority changed: correctness is the gate for every step, and frame time on
letters is taken up again only once behaviour is confirmed in the app.

- **Item 2 (home page fast path) is withdrawn.** It was built and measured on
  letters over the first 40 steps: 21.25 ms GPU per step with box addressing,
  27.96 ms with a page-table read on every lookup, 31.98 ms with the home
  state and no table read at all, 33.94 ms with the full home path. Private
  per-invocation state costs more than the table read it avoids, and it
  doubles shader build time. Every lookup now reads the page table. The
  remaining 6.7 ms is a performance item (resolve the page once per site or
  per sample, a direct-mapped window mode, halo'd tile words), not a
  correctness one.
- **The pressure root stops at 32h.** The coarsest level is solved with
  16 red-black projected SOR sweeps instead of descending to one cell, so the
  root needs no storage shaped like the whole box below page resolution.
- **Order of work now:**
  1. Seam with direct table reads (done).
  2. Tile and cell keys page-major: `block*512 + local`.
  3. Cell, face and tile textures in block layout.
  4. Vertex fields at stride 33 with a low skirt; low faces in per-block
     planes.
  5. Page table live with AIR and SOLID sentinel blocks, tested with an
     identity seed and a permuted seed with free blocks.
  6. **Live allocation inside a bounding box.** Pages are requested from the
     existing 16³ residency certificate, claimed and written in full by the
     GPU, and retired after 8 idle steps. Solids are read from the dense box
     mask, so a slide ahead of the water needs no page. Fatal on pool
     exhaustion, too many requests, window alias, a certificate reach over
     the cap, a seeded page that was not resident, and a particle arriving in
     a page with no storage. A block costs 4.90 MiB.
  7. Remove the bounding box: the box-shaped algorithm tables (pressure page
     graph, extension pyramid, frame-plan and census tables, wall reach, the
     4h cache, the solid mask) move to pages.
  8. Performance on letters.
- **State at 16:50.** Steps 1 to 5 are in the tree and checked on the device.
  The trough dam break gives 114.2k to 116.6k particles at frame 40 across
  every step, against 114.2k before step 3. With the page table live, a
  permuted seed with free blocks matches the identity seed on the trough
  (114,638 against 114,398) and on letters (971,257 against 985,580 at
  frame 40, drift -0.0102 in both; run-to-run spread on letters is about
  1.3%). Writing NaN into both sentinel blocks every frame leaves no NaN in
  any stored texel. Step 6 has its parts in the tree (request kernel, page
  initialise, FLIP arrival fatal) with every box page still pinned; unpinning
  and the slide runs are in progress. Cost so far on letters: GPU 28.7 ms per
  step before step 4, 33.0 ms after it (a table read per vertex tap).
- **State at 19:30: step 6 is live.** Every narrow-band FLIP scene now runs
  on a pool that commits each step; pages are claimed from the residency
  certificate and retired after 8 idle steps.
  - The pressure root follows the page set: two fixed launches per step
    write its stencils from the page table, and an unstored box tile holds
    an inert row. Root rows stay box-indexed (about 20 KB per unstored box
    page); re-keying the root, the solid record and the multigrid lattice
    above it belongs to step 7. Before this, the drum on a live pool failed
    with "pressure did not converge" in 5 of 7 runs by frame 79; after it,
    4 of 4 pass.
  - The sentinel blocks' support flags are zero. They were set for every
    key at build, so the extension read an unstored page as a supported
    source and filled nearby air with junk the step the first page retired.
  - Device results on this state: trough 114,081 particles at frame 40 with
    4 of 16 pages retired at step 8 and re-claimed by step 20; drum 2 seeded
    pages growing to 75 by step 180, liquid within 0.6% of the full-box run;
    letters 984,281 at frame 40, drift -0.0102. The long slide ran 600 steps
    before the root change (4 to 186 of 192 pages, nothing retired).
  - No scene shows a memory saving yet. The residency closure is
    ceil(travel/4)+5 tiles from every seed tile, about 10 tiles (40 cells)
    around the surface, which covers the whole box of both slides; letters
    seeds 152 of 192 pages because the pages of its nine scheduled drops
    are pinned from t=0. `RESIDENCY_MARGIN_TILES` and the source pin rule
    are the levers.
  - Open: letters measured 55 ms GPU per step on this state against 33 ms
    before it, with every stage slower by a similar factor; not yet
    attributed. Two extension helpers (`restrictBand`, `umFarValue`) resolve
    unstored tiles without a presence test and are safe only through the
    support word.
- **Step 7 design, 20:00.** Route: the 64³ page table stays as the window and
  every box-shaped table is re-keyed to pool blocks or tile keys, so cost
  follows the pool. A hashed directory is ruled out by the 21.25 to 27.96 ms
  measurement above.
  - Step 7 is not where the memory is. Letters holds 727 MB of box-shaped
    GPU storage and 666 MB of it is one rule: capacity for h tiles is every
    box tile, at 6,776 B each. In the tree: capacity follows the pool,
    min(box tiles, 512 × claimable blocks), an exact bound because an h tile
    exists only in a resident page; and scheduled drops are requested one
    step ahead of their start, not pinned from t=0 (letters seeds 96 pages,
    was 152).
  - The capacity rule saves nothing yet. Letters' resident set reaches 192
    of 192 pages because the residency closure reaches 5 to 16 tiles and
    fills the box, while its h tiles peak at 7,227 to 7,547 of a capacity of
    98,304. The saving arrives with a local residency reach (census chain by
    block) or an authored pool smaller than the box. Not yet run on a pool
    smaller than the box.
  - Shared primitives are in the tree and read by nothing yet: a block
    neighbour table (27 words a block) and wall-face list after the block
    records in each topology tail, rebuilt in four fixed launches after every
    page commit (`UniformBlockNeighbours`); kind-aware clamps
    (`um*NearFrom`) that hold a coordinate only where a SOLID page lies
    across the face; block face-plane index families. A module opts in with
    `uniformMixedTopologyNeighboursWGSL`. 25,612 B a topology at 194 blocks;
    one new shader module. Trough dam break unchanged with both patches:
    47,282 / 62,381 / 87,094 / 114,071 particles at frames 10/20/30/40.
  - The seed margin (56 cells) and `RESIDENCY_MARGIN_TILES` (5) stay. The
    first is the certificate's reach written twice; the second is bounded by
    the surface-volume reader chain.
  - Decided by Peter: an AIR neighbour never clamps, so an open-top scene
    keeps liquid that rises above the old top plane rather than deleting it;
    a tank that is not a whole number of 32-cell pages is solid-padded to
    one (below; first decided as a refusal); the
    Requested and Full layout policies and the detail planner are deleted
    with the box (narrow-band FLIP is the only installed method and it fixes
    the Dynamic policy).
  - Order: shared primitives (block neighbour table, kind-aware clamp, block
    face planes, wall-face list); then census, extension, 4h cache, wall
    reach, solid directory, particle clamps and renderer in parallel; then
    the pressure root and multigrid levels by tile key; then delete the box
    symbols and give the lattice an origin.
  - Parallel round, state. Census tables by block: in the tree (per-block
    tile bits and bound cubes replace the prefix tables and the cube
    pyramid in the census and frame plan, 12.24 MiB less on letters and on
    the long slide; trough 47,275 / 62,385 / 87,220 / 114,381, letters
    979,680 particles and drift -0.0110 at step 40, long slide to step 200
    with no fatal). Not done there: the join mask's plane 1 is an arbitrary
    tile set from voxel edits, not regions, and wants a tile-key plane; the
    residency words have about fifteen more consumers than listed (root
    owners, pressure, remap, surface volume) and move as one change. A
    local residency reach is not worth building: the host model of letters
    peaks at 180 of 192 pages with one reach and 166 with a reach per seed
    page, because the 5-tile margin alone rounds a seed to a cube three
    pages a side. Solid directory: in the tree (the dense box mask is one
    image: a 64³ page directory, 4,096 B page slots, the bed-slope lane and
    a cut map by tile key; `umSolidVoxel` answers for every page; about
    1 MiB more scratch a scene; both mask probes 0 mismatches, trough
    47,682 / 64,336 / 88,841 / 116,830 with the root fix below). Particle
    clamps: built, held until the padded tanks below land. Running:
    residency words by block with their fifteen consumers, extension, 4h
    cache, wall reach.
- **Open: tanks that are not whole pages.** Refusing them (decided above) is
  not landable as it stands: 72 of the 111 catalog rows with water are not
  whole pages on every axis, the app default `water-box-dam-break`
  (24×16×16) among them. Padding a tank to whole pages with solid voxels is
  not the lattice wall either: every solid voxel is a cut cell that couples
  the tiles within a cell of it and holds wet neighbours at h (the default
  scene would hold 80 to 96 of its 96 tiles at h), and one padding voxel
  turns on the solid gate (per-particle voxel walk, solid-aware surface
  taps) for the 58 rows that run with it off; the pressure root's wall
  slots and wall release, extension, momentum and redistance also tell a
  box wall from a solid tile. **Decided by Peter (14 Oct): solid-pad scenes
  that are not 32-aligned.** The container (authored extents: liquid
  authoring, world origin, UI, vessel outline) and the lattice (what the
  solver allocates) become two things; the lattice is the container rounded
  up to whole pages on the high sides, the padding solid across closed
  walls and air above an open top, through the existing solid path. A
  padded scene behaves as a voxel bath does (wet tiles beside the padding at
  h, the solid gate on); treating a whole solid tile's face as a wall, so
  padding costs neither, is a later step if the cost matters. **In the
  tree:** `sceneFluidLatticeDimensions`, `solidWorldForFluidLattice` (up to
  three fill slabs past +x, past +z and above a closed top), the solver on
  the lattice and the frame, outline and UI on the container; no solver
  shader changed; the refusal of a non-whole lattice and the deletion of the
  lift branches. All 111 wet catalog rows have a whole lattice; eleven rows
  refused before now construct (`sphere-jet` runs). Measured: a padded
  settled tank (`hydrostatic-power-two-level`, 16³ in 32³) holds 3,072 cells
  for 30 steps with 172 tiles at h (two layers of padding tiles; none
  unpadded); the default scene runs with 236 tiles at h at step 30 against
  66; the hose tank's liquid matches its unpadded run (12,189 against 12,221
  at step 40) with 727 h tiles against 518; the trough is in reference. A
  tank symmetric in a padded axis is not treated symmetrically (lattice wall
  low, voxel wall high), and four rows' high depth symmetry plane is a solid
  wall. The picture of a padded scene has not been looked at.
- **Pressure coarse levels keep enclosed air, in the tree.** A coarse level
  whose cells average a thin air gap away had liquid rows and no
  free-surface row, was singular, and the first cycle raised the residual
  (the padded 16³ tank died at frame 0; a 32³ tank filled to 27 cells
  diverges the same way in the host model). Every coarsening now keeps
  enclosed air and classes a child as air by the liquid test, not by the
  sign of phi. The host model says 8 of 92 scenes change, among them
  `minimal-power-dam-break-64` and `high-resolution-dam-break`.
- **Surface dents at h/4h seams, fixed in the tree (not from paging).** On
  `high-resolution-dam-break` the column's free side faces carried dents one
  tile tall along the edge of the h region. Where an h tile meets a 4h tile
  and both 4h centres are in air, the root has no liquid row for either and
  writes their shared face zero; the h pressure band took that zero as a
  solved Neumann flux, a wall across the liquid, and turned the falling
  liquid sideways along the seam. Present before paging too. Fix
  (`bNeumann` in `uniform-pressure-band.ts`): such a face keeps the h
  face's forced velocity. Measured with `tools/probe-surface-dents.ts`
  (vertices of a side face more than 0.2 cells off their neighbours, steps
  8/10/12): 125 / 566 / 616 before, 0 / 0 / 0 after, the all-h value; trough
  and settled tank unchanged.
- **Raised strips inside the dam break's top edges, fixed in the tree.** A
  tile promoted beside the surface was handed its old 4h faces by
  `oldFaceValue` in `uniform-mixed-remap.ts`, which read them from the h
  texture; on a face between two air cells the h ring's copy of the
  extension is stale (0.000 against −0.652 in the base block, measured at
  the head of step 2), so the new h tile fell too slowly. The load now reads
  the base block (the canonical-load mark `copyFace` already carries). Strip
  height at step 12: +0.98 / +0.88 cells before, −0.38 / −0.16 after, all-h
  reference −0.25 / −0.13 (one run per arm). Which kernel leaves the ring's
  copy stale is not located.
- **Open on the dam break.** (a) Terraces at the column's foot, 4 cells apart
  in y, locked to the tile lattice: the front runs into tiles promoted at
  that step's head, and the remap hands each h face its old 4h face value,
  constant across the face, so neighbouring tile rows advect at speeds 0.7
  to 1.0 cells a step apart. They enter in the advection, not the particle
  reconstruction. In the tree: the remap rebuilds a 4h face handed to h
  faces as a plane tilted toward the faces beside it (`remapSlope`, limited
  to 4/3 of each one-sided difference, zero at walls; flux kept; the dynamic
  census's `SAMPLE_REACH` goes 2.0 to 2.5). Terrace number at steps 5 / 6 /
  8 / 9 / 12: 0.174 / 0.174 / 0.272 / 0.421 / 0.706 before, 0.130 / 0.101 /
  0.139 / 0.276 / 0.693 after, all-h reference 0.062 / 0.077 / 0.115 /
  0.150 / 0.202. Not closed: the extension's far value is blocky on h faces
  (nearest coarse source, in the reference too), and the foot and sheet are
  still 4h under the adaptive profile when the artifacts form. Both in
  progress. (b) A residency fatal,
  "closure of 17 tiles, over the cap of 16", at frames 16 to 23 in most runs
  as the front meets the far wall: 68 cells a step, not a flow speed.
  Predates today's fixes; under investigation. Liquid there dips 4 to 6 % at
  steps 16 to 24 (dt 1/60 only, not swept).
- Built and held: residency words by block (needs pool room for the air
  pages above an open top), extension by block, wall reach by page kind,
  particle clamps; stacked into one patch so an open top is open in every
  consumer at once. On device the stack holds the settled padded tank
  (liquid 3072, 172 h tiles), the drum (8,156 / 16,248 at steps 60 / 120),
  the basin (0.0006 m/s) and letters, and stops the trough at step 1: a
  page claimed above the open top has no pressure-root rows, which the stack
  reports as a root fatal where today that liquid is silently deleted. It
  lands with the root keyed by tile, not before. 4h cache by block: built,
  correct on the trough, waiting on a clean frame-time comparison.
- **Decided by Peter (14 Oct): scenes do not author an h-tile capacity.**
  Capacity stays min(box tiles, 512 × claimable blocks), paid upfront.
- **Gentle slopes, in the tree.** A film on a voxel staircase has a level bed
  under a level surface, so nothing on the grid carries the fall of the ramp
  the treads stand for. Fix: a bed-slope lane (two signed bytes a cell after
  the solid mask, units 1/254, uphill, non-zero only on an open cell standing
  on a solid top face) baked on the GPU from the voxels at build and per
  edited region (`UniformBedSlopeBake`: nearest nosing and eight relaxation
  passes, 108 fixed launches); a force `-g × slope` on the horizontal faces
  of film cells, weighted out as liquid covers the cell above
  (`umBedDrive`); and a particle that lands on a sloped bed keeps its
  velocity along the bed plane. Scenes `nbflip-ramp-*`, probe
  `tools/probe-ramp-residue.ts`. Measured on the ramp channel (7,238 cells
  emitted): residue 698.8 cells without the term at 1/60 s; with it 0 at
  1/60 s and 1.9 to 17.9 at 1/30 s. The lane is box-shaped today and is
  re-keyed with the solid storage.
- **The pressure root on a thin film over a cut 4h tile, fixed in the tree.** With no
  slope term at all the ramp channel dies at frame 438 (1/60 s), "pressure
  rejected a non-improving cycle"; the worst rows are 4h tiles (11,3,2..3),
  the film on the tread whose tile has two open layers, at phi/4h -0.03 and
  pressure -2.4e4. A cut tile is a liquid root row for any film depth (plane
  fit read at the tile centre) but keeps an h face only where a film cell is
  at least half a cell deep (`umWetFace`), so the rows lose their faces and
  their free surface as the film thins. About one ramp run in four at 1/30 s
  with the term hit it. Fix: toward an owner that is not a row, a row's open
  h cells on the side its phi was fit through keep their faces
  (`umWetSide`; the cut slot's second word records the sides), and with
  liquid on neither side of a wall the fit is through the nearer side.
  Measured: the term-off channel runs all 450 steps; with the term, residue
  0 cells at 1/60 s and 5.9, 9.5 and 51.9 at 1/30 s; the basin stays at
  rest (0.0006 m/s); the settled tank's drift at 120 steps goes from
  -0.0025 to -0.0010. The trough dam break carries 2 to 3% more particles
  from frame 20 (64,238 and 64,314 at frame 20 against 62,442; liquid and
  drift unchanged), so its reference numbers move.
- **Splash-phase frame time (cm12-figure-9, GPU ms a step, box addressing to
  paged):** 5.69 to 6.91 with no particles, 39.75 to 68.89 at the splash
  peak. 96% of the excess is in kernels whose only change is the page
  address of each load: a paged vertex load costs 2.5 to 2.7 times a box
  one, a cell load 1.1 to 1.2. The table read itself is 0 to 14% of that;
  about 40% is control flow (stored test, skirt search, h/4h store branch)
  and the rest per-tap arithmetic, so the remedy is to resolve a vertex
  address once a tile workgroup, not once a load. In the tree so far: gather
  loaders that resolve one address for a box of vertices (peak 62.39 to
  58.59 summed over passes) and the coarse retirement test's tile word
  before its heat sample (0.590 to 0.066 ms with no h tile).
- **Scenes that show growth.** `nbflip-water-slide` is a spiral channel in a
  solid drum draining into a central pool (160×96×160, 60 L/s for 8 s);
  `nbflip-water-slide-long` is a serpentine over 512×96×128 where most pages
  never hold liquid. The grid overlay's "pages" layer shows the resident set.
- **Known limitation, not caused by this work:** a closed tank with less than
  16h of air above the liquid fails at the start-up kick with the old and the
  new pressure root alike. A candidate fix in the multigrid setup is recorded
  and not yet verified on the GPU.

## Summary

Today the tank is the fluid's storage, its coordinate system and its boundary
conditions all at once. Making the fluid domain-less means separating those
three things:

1. **Frame.** A world lattice with a fixed origin, one isotropic cell size and
   signed integer cell coordinates, instead of a lattice measured from the
   tank's corner.
2. **Storage.** A fixed pool of 32³-cell pages claimed and released on the
   GPU, instead of arrays shaped like the tank.
3. **Boundaries.** Walls become ordinary solids (the tank shell is already
   authored as voxels), and "outside what is stored" means still air instead
   of a wall.

The recommended route is to treat **the pool as the new lattice**: keep the
solver's tile, owner, particle and pressure machinery as it is, and change
only how a tile is named (pool page and local tile, instead of a tank
coordinate) and how a neighbour across a page face is found (a per-page
neighbour table, instead of coordinate arithmetic). That keeps every per-tile
table and kernel, and it can be reached in steps that each convert the whole
solver, so there is never a second code path.

On sharing with the SVO renderer: share the lattice frame, the unit nesting
and the paged solid world, and nothing else. The fluid should start reading
solids from the renderer's existing signed, hashed `SolidWorld` page image,
which is the one piece of real shared machinery available and deletes code. It
should not share an allocator, a directory or storage with the octree.

Four findings shape the plan:

- **There is no paging to finish.** The "page domain" in the solver is
  uploaded and never read, and the detail layer is one lattice-sized patch
  with every tile reserved. This is a build, though a well-informed one: a GPU
  page transaction was built, Dawn-tested and deleted in September, and its
  measurements still apply.
- **Per-tap directory lookups are ruled out by measurement** (3–4× slower).
  Page placement must be resolved once per kernel workgroup or once per
  sample. With that discipline, 32³ pages measured about equal to dense.
- **Pressure is the hardest part.** The global solve is a multigrid over a
  box-shaped all-4h lattice with wall slots on six planes.
- **"Infinite" will mean unbounded addressing with bounded liquid.** A fixed
  pool is required by the stability rule, so the amount of resident water is
  a budget. Pool exhaustion must be a designed outcome.

## Terms

| Term | Meaning |
|---|---|
| h | Finest cell size. |
| Tile | 4³ cells; one 4h cell. The solver's unit of resolution choice. |
| Owner | One pressure/velocity unknown: an h cell of an h tile, or a whole 4h tile. |
| Page (proposed) | 32³ cells = 8³ tiles. The unit of residency and allocation. |
| Pool | The fixed set of page slots allocated at scene build. |
| Resident | A page that currently holds a slot. |
| Ambient | The state of everything that is not resident: air at rest. |

"Page" already means three different sizes in the code: 16 cells for
residency (`UNIFORM_MIXED_PAGE_TILES = 4`, `uniform-mixed-topology.wgsl.ts:46`),
32 for `UniformPageDomain` (`UNIFORM_GEOMETRIC_PAGE_EDGE`), and 16 for the
pressure atlas (`UNIFORM_PRESSURE_PAGE_EDGE`). The plan ends with one meaning.

Units nest cleanly: one 32³ page is 8³ solver tiles, 4³ renderer bricks and
4³ `SolidWorld` pages (both 8³ cells).

## What the review found

Three sub-reviews covered the root solver, the detail/pressure/particle layers
and the SVO renderer. I read the scene model and recovered the deleted
September design documents from git history.

### The tank is baked in at four levels

**Storage.** Every field is dense over the tank: velocity A/B/D, volume,
surface and gamma textures at `[nx,ny,nz]`, vertex phi at `[n+1]³`, 2D height
and terrain textures, a `(n+2)³` solid bitmask, the pressure arena, and
particle bins at one word per lattice cell and vertex. The h detail layer is
also dense: one patch covering the lattice (`uniform-detail-fields.ts:111`,
`patches:1`) with capacity held at every tile (`hold: true`,
`uniform-detail-policy.ts:158`). The atlas and slot vocabulary in
`lib/core/uniform-detail-abi.ts` is left over from an earlier design.

**Addressing.** A tile's key is
`p.x + UM_T.x*(p.y + UM_T.y*p.z)` (`umTileAt`,
`uniform-mixed-topology.wgsl.ts:127`), and every neighbour is found by key
arithmetic with clamps at the tank faces. There is no neighbour table and no
"absent" state; a tile is h or 4h. Owner indices are a GPU prefix sum in
tile-key order rebuilt each frame (`uniform-mixed-layout-builder.ts:347-352`),
so every relayout runs a remap.

**Shader constants.** Tank dimensions are baked into WGSL on purpose
(`uniform-scene-constants.ts:1-5`: `UM_D`, `UM_TILES`, `NB_CELLS`). The purge
commit `e60dc771` measured tank-independent shaders at 29.3 → 32.3 ms per step
and re-baked them. Device limits therefore define world size:
`max(n)+2 <= maxTextureDimension3D` and
`nx*ny*nz*24 <= maxStorageBufferBindingSize`
(`webgpu-uniform-reference.ts:821-824`).

**Physics.** The array edge is the boundary condition. `valid(p)` means
"storage exists", "inside the wall" and "below the open top" at once. Outside
the array reads as wall for open fraction, air for phi and volume, and
edge-clamp for velocity. Specific mechanisms that are box-shaped:

- Wall slots on six planes of the pressure root, with their own release rule.
- The `closedWalls` six-bit mask and boundary velocity planes on three faces.
- The open top as a ghost row, and as the only place liquid can leave.
- Depth symmetry, which is implemented as a closed wall.
- Particles clamped and reflected at `0` and `UM_D`, and deleted outside.
- The volume target, computed from `fillFraction * ny` and tank cell volume.
- World↔grid mapping, inflow, drops and edits anchored at the tank centre and floor.

### What already works in our favour

- **The relayout loop is GPU-resident.** Census → builder → admit → remap →
  adopt runs in one encoder with no readback. Page allocation can live in it.
- **A residency certificate exists.** Per 16³-cell page there is a flag, an
  ascending resident list and a GPU count, rebuilt each frame with a
  travel-based closure radius (`pageSeed`/`pageMark`/`pageCompact`,
  `uniform-mixed-dynamic.ts:1247-1298`) and audited fatally. Its contract is
  already "absent means certified far air, p = 0". It restricts iteration
  only, and is used unevenly.
- **The h pressure band is a sparse solver.** `UniformPressureBand` has
  capacity rows, a GPU list and count, a slot map, Galerkin 2h and 4h
  aggregates, a flux rim and a fatal overflow. Its slot map is still one word
  per tank tile.
- **Dispatch is mostly compliant.** Launches are fixed direct grid-stride or
  counted direct. The live indirect launches are exactly three, all over
  particles (`uniform-narrow-band-flip.ts:870`, `:956`;
  `uniform-narrow-band-order.ts:151`).
- **Particle runs are tile-local.** Gathers split rows at tile boundaries, so
  no kernel assumes neighbouring tiles are adjacent in memory.
  `spatialRank` already orders tiles by 4×4×4-tile blocks.
- **Stencil kernels run one workgroup per tile** and no kernel uses hardware
  filtering (no `textureSample` under `lib/methods/uniform`), so resolving a
  page once per workgroup is natural and an atlas has no bleed problem.
- **The tank shell is already voxels.** `solidVoxelShellForScene`
  (`lib/core/scene-lattice.ts`) authors the floor at y = −1 and walls at −1
  and n as `SolidWorld` patches in signed finest-voxel coordinates.
- **The renderer already rebinds when the solver's source identity changes**
  (`levelset-consumer-abi.ts:226-228`), and `GPUSolverInstance.fluidDomain`
  (`lib/core/method-contract.ts:195-200`) is a named seam, though no shader
  reads it yet.

### What is nominal or dead

- `UniformPageDomain`: an all-resident rectangular catalogue uploaded and
  never read. `createUniformReferenceComputeShader` ignores its page and
  domain arguments. `planUniformPages` (`uniform-page-layout.ts`) is a
  host-only planner with signed coordinates, stable slots and a six-neighbour
  table that is never uploaded.
- `activeRegion` and the window machinery; "paged pressure"
  (`pagedPressure = false`, `webgpu-uniform-reference.ts:1046`); the native
  CM11a cycle plan (only four setup kernels are live); `reserveFine`
  reallocation (unreachable under the live policy).
- Fluid in the octree: `OctreeSparseBrickWorld.encode` has no callers, and
  `GPUFluidBrickResidency` (`lib/core/webgpu-fluid-brick-residency.ts`),
  `sparse-cm12-frame-plan.ts` and `octree-consumer-sampling.ts` are dead but
  partly still allocated.

### Costs that scale with the tank, not the liquid

About thirty passes run dense over the tile grid every step regardless of how
much liquid there is: builder launches, census, frame plan, adopt copies, the
4h momentum cache, wall planes, particle order prepare and prefix, activity,
the extension pyramid and pressure root setup. The velocity extension coarsens
the whole tank to a single cell
(`webgpu-uniform-velocity-extrapolation.ts:465-472`). Particle capacity is
`min(cells*8, max(2_097_152, cells), limit/48)`
(`uniform-narrow-band-flip.ts:147`).

### Renderer and scene

- The SVO tree holds no fluid. Its only constructor is the renderer sidecar
  with `solverClaim: "none"`
  (`lib/svo/features/construction/webgpu-svo-sparse-bricks.ts:1736`).
- No renderer allocator is GPU-side or able to free. Octree slots, solid pages
  and node-mip pages are allocated in host JS on edits. Octree keys are
  unsigned, the lattice is fixed at build, and edits outside it throw.
- `lib/core/webgpu-solid-world-pages.ts` is the exception worth using: signed
  8³ pages behind a GPU-read hashed directory with a fixed byte budget. The
  solver ignores it and flattens solids to a dense mask.
- The water pipeline (`lib/core/webgpu-water-pipeline.ts`) is the main
  render-side work: one box world mapping, box-semantic field reads, a window
  classifier with 10-bit-per-axis ids, mesh capacity from box face areas,
  caustics over the container footprint, and one indirect compute dispatch.
- Outside the brick world there is only an analytic backdrop
  (`lib/core/backdrop.ts`) that nothing collides with.
- `scene.container` is referenced about 400 times across `lib` and
  `components`. Cell size is `container extent / n`, so it is not exactly
  `finestCellSize_m` and can differ per axis. `lib/core/model.ts:550-751`
  requires liquids, drops and inflow to be inside the container.
- The Rust oracle (`rust/crates/fluid-core/src/narrow_band_flip`) also assumes
  a dense box.

### Prior art recovered from git

`e60dc771^:docs/plans/uniform-sparse-residency-plan.md`,
`6cc2184d^:docs/research/uniform-page-domain-design.md` and the deleted
`uniform-page-generation.ts`, `uniform-page-support.ts` and their Dawn tests.
The results that still bind:

| Result | Consequence here |
|---|---|
| A paged atlas with a catalogue lookup per tap was 3–4× slower (long dam 46 → 119 ms). | The directory is never in a tap path. |
| A 32³ stencil with cached seam neighbours measured −3.2% against dense; 16³ measured +5%. | 32³ is the right page edge, as you proposed. |
| A GPU prepare/publish transaction with a hashed signed directory, stable slots and six-neighbour tables was built and passed its Dawn tests. | Use it as the reference; drop its indirect dispatch and serial planner. |
| A finite phi cap (±16 h) diverged: 8.2% volume-field difference by frame 12 on the garden hose. | Re-test under NB-FLIP, which now rebuilds the level set from particles. See risks. |
| The design note's rules: no array indexed by the bounding box of page coordinates; the edge of allocated pages is not a wall; residency grows from current residents plus explicit source and edit commands. | Adopted unchanged. |

## Design

### 1. Frame

One lattice frame for the solver, the solid world and the renderer:
`{ origin_m, h }` with signed integer cell coordinates. The scene carries it
explicitly; existing scenes derive it from the container's low corner, which
is where `solidVoxels` coordinates are already measured from.

Cell size becomes exactly `finestCellSize_m` on every axis. Container
dimensions snap to the lattice. The container stays in the scene model as an
authoring object (vessel shell, initial fill, vessel outline) and stops being
read by the solver.

### 2. Pages and the pool

A page is 32³ cells with a signed page coordinate. The pool is a fixed number
of slots, `P`, chosen from a memory budget when the scene is built. A page's
slot is stable for its lifetime, so page data never moves.

Each slot has a small record: page coordinate, generation, state, and the
slots of its 26 neighbours (or an "absent" marker). The full 3³ neighbourhood
is needed because vertex and trilinear taps reach edge and corner neighbours.

**The pool is the lattice.** A tile's key becomes `slot * 512 + localTile`.
Every per-tile table keeps its meaning and is sized `P * 512`. h fields live
in 3D textures laid out as a grid of 32³ blocks, one per slot. The constants
baked into WGSL become the page edge, `P`, and the slot grid, all powers of
two, so the divisions in today's addressing turn into shifts. No shader
contains a world extent, which also means pipelines are shared across scenes
of the same budget.

Owner indices can stay a per-frame prefix sum over pool tiles, so the builder,
the remap and the ownership tables carry over. Making owner indices stable is
a possible later simplification and is not needed for this work.

Eventually a page that is entirely 4h should not pay for h storage. To keep
that open, the page record carries an h-store slot that equals the page slot
to begin with. Splitting the pools is then a memory change only.

### 3. Finding a page

There are three ways to reach a page, in order of cost:

1. **Within a page:** local arithmetic.
2. **Across a face, edge or corner:** the page record's neighbour slot, loaded
   once per workgroup. This covers every stencil, MAC face, vertex and
   trilinear tap.
3. **From a coordinate:** a hashed signed directory. This is used only when
   building neighbour tables, when a source, drop or edit injects liquid, and
   by the renderer. Long traces (up to 128 cells) walk neighbour tables page
   by page instead of probing.

Particles carry their page slot, so particle kernels start at case 1.

The directory is small (a few words per bucket, a few thousand buckets) and is
rebuilt from the resident list every generation. Rebuilding avoids deletion
from an open-addressed table, which is the awkward part of a GPU hash.

### 4. Allocation and retirement

All of this runs on the GPU in the existing relayout head.

- **Request.** Each resident page asks for the neighbours that liquid, the
  narrow band or particles could reach this step, using the travel-based
  closure that the 16³ certificate already computes. The particle `refine`
  kernel already requests h tiles from swept boxes; it requests pages the same
  way. Sources, drops and edits add requests through a small host-written
  command list.
- **Claim.** Requests are deduplicated by compare-and-swap in the directory.
  The winner pops a slot from a free stack. An empty stack is fatal.
- **Initialise.** New pages are set to ambient and their solids are sampled
  (section 6). Neighbour tables are updated for the new page and its
  neighbours.
- **Retire.** A page with no liquid, no particles and no claim from any
  neighbour's closure for a fixed number of steps is released. Its slot goes
  to a pending list and returns to the free stack after the next adopt, so no
  reader of the previous generation sees a recycled slot.
- **Certify.** A departure certificate checks that nothing moved into an
  absent page. A violation is fatal. This extends the existing residency
  audit.

Launches over new or resident pages are fixed direct grid-stride launches that
read the count on the GPU.

### 5. Boundaries

| Today | Domain-less |
|---|---|
| Tank walls at the array edge | Authored solid voxels (already present in `SolidWorld`) |
| Open top ghost row | Nothing; air above is ambient, and splashes claim pages |
| Lid | Authored solid |
| Depth symmetry | Authored solid slab (it is already a closed wall) |
| Outside the array | Ambient: phi far positive, velocity 0, volume 0, p = 0 |
| Floor at y = 0 | Analytic `SolidGround`, evaluated when a page is initialised |
| Top plane as the only outflow | Authored sinks (see open decisions) |

This removes the pressure wall slots, `closedWalls`, the boundary velocity
planes, the open-top branches and the particle clamp at `UM_D`. Whether walls
behave the same as thin solids is the first experiment in the plan, because
wall slots have a release rule that solid faces do not.

### 6. Solids

The solver reads solids from the `SolidWorld` page image and the analytic
ground, sampled into per-page solver storage when a page is initialised and
when an edit touches a resident page. This replaces the dense `(n+2)³` mask,
the 2D terrain texture, the tank-centred solid origin and the CPU scan that
drives solid promotion.

An infinite scene needs a collider wherever water can go. The ground function
must therefore be the single source for both the backdrop the renderer draws
and the collider the solver samples.

### 7. Pressure

**Global stage.** Inside a page the 4h lattice is 8³, which coarsens
geometrically to 4³, 2³ and 1 with page-local storage (585 cells per field per
page) and neighbour-table taps. The existing tiled smoother already loads one
4³ tile plus halo per workgroup, so it carries over.

Above the page there is one unknown per resident page. The existing coarsest
launch solves up to 1,000 cells in one workgroup
(`ROOT_SINGLE_MAX_CELLS`, `uniform-mixed-pressure-cycles.ts:41`), so a
page-level system fits it while the resident count is under about a thousand
pages (roughly 33 million cells). If convergence on the target scene fails its
gate, add aggregated levels above the page. Do not build them first.

Ambient pages are Dirichlet p = 0, so separate bodies of liquid are each well
posed and are not coupled.

**h band.** `UniformPressureBand` is already list- and slot-based. Its slot
map changes from one word per tank tile to one per pool tile, and its
cross-tile lookups go through the neighbour table.

### 8. Velocity extension

Replace the whole-tank nearest-source pyramid with a page-local pyramid that
stops at one cell per page, reaching neighbours through the table. NB-FLIP
needs extended velocity within the band and the step's travel, not across the
scene. Beyond the reach, velocity is ambient.

### 9. Particles

- **Position** becomes page-local f32 in `[0, 32)` plus a page slot. This
  removes the non-negative-coordinate assumption and fixes precision:
  absolute f32 cell coordinates are down to about 1/256 of a cell at 2¹⁵ cells
  from the origin, the same order as the 0.01-cell wall offsets.
- **Dead** becomes a reserved slot value instead of `position.x < 0`.
- **Sort key** `64 * umTileAt(t) + local` becomes `64 * poolTile + local`,
  which is page-major order.
- **Bins and links** are sized by the pool.
- **Capacity** is a budget of its own, no longer derived from `cells`.
  Overflow of mandatory seeds becomes fatal; today it is silently counted.
- **Crossing a face** follows the neighbour table. An absent neighbour is the
  departure certificate failing.
- The three indirect launches become fixed direct grid-stride launches. The
  pack kernel is already written as a grid-stride loop.

### 10. Renderer

Sharing, stated once:

| Share | Do not share |
|---|---|
| The lattice frame and unit nesting | Allocators (the renderer's are host-side and never free) |
| The `SolidWorld` page image as the solver's solid source | The octree, its keys or its slots |
| `signedSpatialCoordinateHash` for the page directory | Directory tables |
| Conventions: byte-budgeted reserve, revision-stamped claim, direct-dispatch cursor | Storage |

The fluid stays out of the octree. That was built, caused the render-side
out-of-memory path, and was abandoned.

The solver publishes a page contract in place of `fluidDomain`: the frame, the
page records, the resident list with its GPU count, the directory, and the
field atlas. The water pipeline then:

- maps world position to a page through the directory once per ray segment or
  per surface page, and reads fields page-locally;
- extracts the surface per resident surface page, with mesh capacity sized by
  the pool;
- anchors caustics and the fluid coverage volume to the resident set instead
  of the container footprint.

Other box dependents (grid overlay, scale overlay, particle overlay origin,
raster AO frame, vessel cutaway, editor validation) move to the frame or to
authored scene bounds.

Unbounded scenery is a separate project. This plan needs only that water can
be drawn, and can collide, outside the container.

## Memory

The pool is paid upfront. I have not measured today's bytes per cell, so this
is an estimate to be replaced by a measurement in phase 0. The known parts add
up to roughly 160 B per cell before h detail storage and pressure band rows:
48 for three velocity textures, 12 for volume/surface/gamma, about 4 for
vertex phi, about 20 for particle bins and links, and about 80 for particles
at one per cell.

| Pool | Cells | At 160 B/cell | At 250 B/cell |
|---|---|---|---|
| 64 pages | 2.1 M | 0.34 GB | 0.52 GB |
| 128 pages | 4.2 M | 0.67 GB | 1.05 GB |
| 256 pages | 8.4 M | 1.34 GB | 2.10 GB |

A 256×128×128 tank is 128 pages, so a pool that size costs what that tank
costs today. The difference is that the pages can be anywhere.

This also sets the honest scale. At h = 5 cm a page is 1.6 m on a side, so 256
pages is about 1,050 m³ of resident water and air. A lake is out of reach
until all-4h pages stop paying for h storage (phase 5), and beyond that it
needs coarser levels than 4h, which this plan does not include.

## Plan

Each phase converts the whole solver and deletes what it replaces. Each has
one target scene. Verification follows the standing rules: targeted probes,
two repeats per arm at matched dt, and a 1/30 s and 1/60 s sweep before any
volume claim.

### Phase 0: clear the ground

No paging. Each item stands on its own.

1. **Walls as solids.** Pad the lattice so the authored shell lies inside the
   array, and compare against today's walls on one tank scene. This decides
   whether wall slots, `closedWalls` and the boundary planes can be deleted or
   whether solid faces need the wall release rule first.
2. **Exact isotropic h.** Make cell size `finestCellSize_m` and snap
   containers to the lattice.
3. **Measure bytes per cell** from the existing allocation accounting, to
   price the pool.
4. **Delete the dead paths** listed under "What is nominal or dead". They
   overlap almost exactly with what gets rewritten.
5. **Remove the three indirect launches** and make particle pool overflow
   fatal.

Gate: the padded tank scene matches the unpadded one within the tolerances of
the time-step sweep, at no worse frame time.

### Phase 1: frame and shared solids

Introduce the lattice frame. Move the solver's world↔grid mapping, inflow,
drops and edits onto it. Read solids from the `SolidWorld` page image and
analytic ground. Delete the dense mask, the terrain texture and the CPU
promotion scan. Storage is still one box.

Target scene: an existing terrain scene. Gate: same behaviour, and solid edits
still land.

### Phase 2: page-major addressing in the box

Change `umTileAt` and every dependent table, bin and sort key to
`page * 512 + local`, with pages still in a regular grid and neighbours still
found by arithmetic. Fold the 16³ residency certificate into the 32³ page.
Lattice dimensions round up to multiples of 32; the padding is air.

This is a re-indexing with no change in physics, so it can be compared
directly against phase 1. Gate: frame time within a few percent on the target
scene.

### Phase 3: neighbour tables and ambient

Replace arithmetic neighbours with page records, add the directory, and make
the box edge ambient. The resident set is still fixed at scene build: the
pages that cover the tank plus one ring. Convert the pressure root to
page-local levels with the page-level coarsest solve, and bound the extension
pyramid at the page. Re-key particles. Publish the page contract and move the
water pipeline onto it.

Target scene: the long dam break, because that is where the September paged
atlas lost. Gate: frame time within the prior measurement's range of dense,
pressure residual and cycle counts no worse, and the renderer draws from pages.

This is the largest phase. If it has to be split, split by stage (4h base and
tables, then h fields, then particles, then pressure), each on the fixed
resident set.

### Phase 4: allocation on demand

Add request, claim, initialise, retire, the pending-free list and the
departure certificate. Add the pool budget to the scene and the receipt.
Relax editor validation so liquid, drops and inflow can be placed outside the
container.

Target scene: a new spill scene in which a vessel overflows onto the ground.
Gates:

- Liquid leaves the vessel and runs on the ground.
- No pipeline is compiled, and no buffer or bind group is created, after the
  first frames.
- A deliberately tiny pool produces the fatal diagnostic naming the pool size
  and the requesting page.
- Volume holds across the time-step sweep.

### Phase 5: cost follows the liquid

Convert the dense per-tile passes to resident-list launches. Give all-4h pages
their own cheaper class by attaching h storage on demand.

Target scene: a small puddle in a large pool. Gate: its frame time matches a
tank sized to the puddle.

### What gets deleted along the way

`UniformPageDomain` and the host page planner, `activeRegion` and the window
code, the tank device gates, wall slots and `closedWalls`, the boundary
velocity planes, open-top and depth-symmetry branches, the dense solid mask
and terrain texture, the whole-tank extension pyramid, `fillFraction` as a
volume target, the `fluidDomain` seam, and the dead fluid-in-octree code.

## Design rules

| Rule | How the design meets it |
|---|---|
| No compile during simulation | Shaders bake the page edge and pool size only. Nothing about extent changes at run time. |
| Stability over memory | Pool, directory, particle and pressure storage are created once. Memory is priced above. |
| Fail fast | Pool exhaustion, particle overflow and a departure violation are fatal with a diagnostic. No fallback path. |
| GPU resident | Allocation and retirement run in the relayout encoder. No readback is added. |
| Zero per-frame allocations | The host writes a bounded command list for sources and edits and nothing else. |
| No indirect launches | Page lists are walked by fixed direct grid-stride launches. The three existing indirect launches go in phase 0. |
| Structural over dispatch | Stable slots mean page data is never moved or rebuilt. |
| Fixes add no expensive work | No halo copy passes; neighbours are read through the table. |
| No inter-frame render caching | The renderer reads the published resident list each frame. |

## Risks

1. **Walls as solids may not match.** The wall release rule exists only for
   domain walls. Phase 0 answers this before anything depends on it.
2. **Pressure convergence** with one unknown per page at the coarsest level is
   untested. The fallback is more levels, which is real work.
3. **Neighbour-table cost** was measured on the September solver. The heaviest
   kernels today are different, so phase 3 has to re-measure on the dam.
4. **Far phi.** Ambient phi is a far positive constant, which is what failed
   in September. NB-FLIP already holds the level set at ±6 outside the band
   reach and rebuilds it from particles, so the exposure looks smaller. It
   still needs the time-step sweep.
5. **Late fatals.** Receipts arrive up to two frames late
   (`UNIFORM_MIXED_RECEIPT_RING = 2`). The GPU failure latch must stop
   allocation at the first failed claim so the state is not corrupted in the
   meantime.
6. **The Rust oracle stays a dense box.** Stage-by-stage debugging then needs
   scenes whose resident set is a box with authored walls, which means the
   oracle has to treat walls as solids too.
7. **The renderer's scenery is still finite.** Water will be able to go where
   there is nothing drawn but the backdrop.

## Open decisions

1. **What happens when water spreads without limit.** A fixed pool will
   eventually fill on an open plane. I recommend authored sinks (drain or kill
   volumes) plus the fatal diagnostic, and no automatic eviction.
2. **Pool size.** I recommend two or three fixed budgets chosen at scene
   build, so pipelines are shared across scenes.
3. **Oracle scope.** I recommend leaving it dense and box-shaped, with walls
   as solids.
4. **Phase 0 deletions.** They are safe by the review's reading (no callers),
   but they are your call.
