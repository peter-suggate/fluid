# Uniform Geometric compiled topology and surface performance

6 October 2026. Design analysis for the current dynamic h/4h Uniform Geometric implementation, including the working tree changes. The proposed savings are structural operation reductions, not measured GPU speedups. No GPU performance experiments were used for this analysis.

## Decision

Compile ownership-dependent vertex and face relationships when building a layout, and reuse them across simulation stages. Keep regular interiors arithmetic. Refresh values when their source fields change, while preserving the compiled geometric recipes. Begin with a shared topology contract used by both CPU and GPU builders, surface vertex traversal, phi resolution, surface-volume correction and seam faces. Improve the fine redistance window's preparation by staging coarse corners once.

This is a structural replacement of repeated topology discovery with reusable data. It does not change the h/4h ownership policy, pressure tolerances, characteristic integration, interpolation formulas, sharpening sweeps, conservation rules or work admission.

## Existing reuse and the remaining opportunity

The implementation already amortizes substantial work:

| Existing mechanism | Work already shared |
| --- | --- |
| Frozen 27-bit tile stencil | Neighbor widths |
| Coarse velocity cache | Restriction of fine face patches |
| Hanging velocity taps | Mixed reconstruction shared by surface and momentum |
| Resolved phi | Hanging-vertex reconstruction before subsequent consumers |
| Sharpening geometry cache | Admission flags across eight sweeps |
| Pressure-band rows and neighbor slots | Coefficients and connectivity across iterations |
| Fine redistance workgroup window | Field samples across Newton iterations |

A surface tile is classified from changing fields; a seam tile is classified from ownership. Their lifetimes differ. A mixed 3-by-3-by-3 stencil includes edge and corner proximity, whereas sharpening's exceptional coarse faces require an h neighbor directly across a face. A single universal seam worklist would overprocess some stages.

The default field storage is domain placement, with no atlas directory lookup. Packed atlas and haloed patch implementations are QA alternatives. Production still pays address arithmetic, field-home selection and some ring/store bookkeeping. Optimizing a generic page lookup would target the wrong default path. See `lib/methods/uniform/uniform-detail-fields.ts`.

## Compile vertex ownership and reconstruction

Vertex operations repeatedly decide which owner may write a canonical vertex, whether the vertex hangs, which coarse owner supplies its reconstruction, and which corners and weights to use. These decisions depend on ownership and position, not current phi.

Surface advection and redistance revisit canonical ownership. Surface-volume correction also resolves authorities. `uniform-mixed-phi-resolve.ts` and `uniform-mixed-surface-volume.ts`'s `resolveScale` contain almost identical staged incident-owner selection and interpolation. They should consume one shared reconstruction recipe with different field loaders.

The recipe is:

```
tile-local vertex
  -> stored or reconstructed
  -> relative coarse authority
  -> coarse corner offsets and interpolation weights
```

The second stencil word currently holds minimum width in bits 27-31 and the detail-ring flag in bit 0. Its 26 spare bits can hold seven coarse-incident bits for the negative octant, seven coarse-neighbor bits for the positive octant, and eight bits identifying a coarse tile's owned canonical corners. That is 22 bits without increasing topology allocation. CPU and GPU builders must produce the same encoding during their existing classification.

For an unstored vertex, eligible incident tiles depend only on which local coordinates are zero. There are eight coordinate categories. Intersect the corresponding constant eligibility mask with the compiled coarse-incident mask and select the first eligible coarse tile. All coarse owners are numbered in tile-key order, so the selected tile is the same lowest-index coarse authority as the current walk. Use the original interpolation formulas, positive-weight handling and D4 sum.

A CPU enumeration checked 29,952 mask/local-coordinate cases against the existing candidate-selection rule. This checks the abstract selector only; full boundary handling, builder parity and GPU fields still need validation.

The compiled recipes are stable under compact-owner renumbering because they contain relative tile choices, not owner indices. Fine canonical ownership can use the positive-octant mask with the existing local vertex traversal. Coarse canonical writers can use an eight-bit mask rather than repeating the eight-owner authority search.

## Prepare redistance windows from unique coarse corners

Fine redistance already stages a 14-cubed window: 2,744 floats, or 10,976 bytes of workgroup storage. It also stages the widths of the window's 4-cubed tiles. `umResolvedVertex` fills each coarse portion by independently loading eight coarse corners for each window point. Adjacent points repeatedly fetch the same corners.

The entire window spans at most 5-cubed aligned coarse vertices: 125 floats, or 500 bytes. Stage those once, fill fine portions from the resolved h field, and fill coarse portions by evaluating the same interpolation from workgroup memory. Then perform the existing Newton searches unchanged. Preserve the existing direct certified-fine path.

This reduces source-level texture requests and address evaluation. It is not an eightfold bandwidth claim: the current texture cache can serve repeated requests. Extra synchronization and shared memory may affect occupancy. Field values cannot survive an intervening phi write; only the geometric recipe can be shared across stages.

The window spans tile offsets -1 through +2 on each axis. A cache covering this footprint has a larger dependency region than an immediate-neighbor topology mask.

## Share seam-face recipes

`umFace` derives an owner origin, probes a neighbor, resolves width and compact index, determines subdivision and constructs an anchor. A coarse face beside h has sixteen patches. In sharpening, proposal, limiter and commit repeat face construction across eight sweeps, even though admission geometry is already cached.

A coarse owner with s fine face-neighbors has F = 6 + 15s incident patches. Limit and commit can each traverse these every sweep, in addition to positive-face proposal work. There is meaningful reuse even when ownership changes every frame.

Separate geometric recipes from current storage addresses:

| Shared geometry | Stage-local resolved data |
| --- | --- |
| Side type and boundary status | Current neighbor owner bases |
| One or sixteen patches | Sharpening budget and flux addresses |
| Anchor and local-lane formulas | Extension or pressure-band slots |
| Neighbor tile relationship | Current field bindings |

Tile words can be loaded once for an executing tile or face and their local lanes derived arithmetically. A sixteen-patch face has one neighboring tile relationship. Regular fine interiors retain index offsets 1, 4 and 16. Negative domain planes and RGBA single-writer packing remain explicit. Each stage preserves its summation order.

A table containing only neighboring tile IDs is unhelpful: IDs are already arithmetic. Useful compiled information removes repeated classification and decoding. A six-u32 descriptor per tile costs 0.75 MiB at 128 cubed and 6 MiB at 256 cubed. Six u32 indices per finest cell cost 48 MiB and 384 MiB respectively. Prefer compact tile rules and cooperative staging over an expanded per-cell graph.

Sharpening also repeatedly maps its packed lane to side and patch. The six side bits define only 64 configurations, so this schedule can be derived from a shared finite recipe rather than six-side discovery per lane. Its benefit must include any table loads it introduces.

## Compile fixed hanging-velocity tap recipes

Surface and momentum already share `UniformMixedHangingTaps.unitVelocity`. Coarse seam taps are currently produced through the general mixed sampler. Their locations are fixed lattice sites, so under fixed ownership each can be classified as a canonical face load, coarse-cache interpolation or negative-boundary read. Source relationships and interpolation fractions can be compiled with ownership; values refresh for each extended-velocity epoch.

This optimizes cache production. Consumers already avoid much of this reconstruction, so the scope is narrower than the shared vertex contract. Arbitrary characteristic positions still depend on current velocity and timestep, and cannot be frozen with ownership.

`umVelocitySamplingWeights` enumerates fine neighbor boxes. Some boxes are geometrically redundant: within the current tile a fine face-neighbor can dominate a fine edge or corner neighbor extending in the same direction. An exact reduced candidate mask could shorten repeated weight evaluation. This requires a proof including boundaries; an approximate sampled weight field changes the method.

## Cache lifetimes and dynamic adoption

| Cached information | Invalidated by |
| --- | --- |
| Width relationships and relative vertex authority | Ownership changes within its footprint |
| Face subdivision and canonical writer | Ownership changes within its footprint |
| Compact owner indices and seam slots | Generation renumbering |
| Physical field addresses | Extent, allocation or binding changes |
| Hanging velocity values | Extended velocity or boundary changes |
| Resolved phi | Canonical phi writes |
| Sharpening admission flags | Geometry, solids or policy changes |
| Pressure coefficients and band neighbors | Setup and band membership changes |
| Transport donor recipes | Departure geometry changes |

The GPU builder places all fine owners first and coarse owners afterward, and independently compacts seam slots. One local edit can renumber distant owners and slots. A local changed-tile list therefore cannot invalidate an arbitrary cache of compact indices. Persistent geometry uses stable tile keys, relative offsets and local lanes; current compact addresses are refreshed separately.

The existing radius-one dilated change list covers immediate incident topology, but not the redistance window or radius-three detail ring. Start by compiling the small bitfields inside the existing full classification pass. Incremental maintenance is a separate optimization.

Persistent recipes belong in ownership topology or dedicated persistent storage, not the scratch range overwritten by successive non-pressure stages. Adoption must publish matching topology and recipes, including CPU-built layouts, GPU generations, unchanged generations, capacity growth and failed/deferred admissions.

## Boundaries of the shared representation

The pressure band already bakes neighboring slots and coefficients. Its active membership and numerical coefficients depend on fields, so shared topology can feed setup but cannot replace its prepared operator. `uniform-pressure-band.ts`'s `bCellNear` is an existing example of tile-local arithmetic plus baked cross-tile relationships.

Transport donors depend on current departures. `donorAt` already uses one compact tile word and shifts. Caching donor indices within an advance is possible, but increases metadata in potentially large rows and must outperform the existing compressed representation. It cannot persist with ownership alone.

Compiled topology does not remove Newton iterations, characteristic substeps, cubic interpolation or surface-volume trial evaluation. These retain arithmetic and dependency chains. Source inspection cannot determine the fraction of frame time spent on topology or establish a speedup.

## Implementation and verification

Implement the shared vertex and face contract first, use it across consumers, and stage the unique coarse corners in the redistance window. Prefer a few coherent structural changes to independent micro-tweaks.

CPU reference checks should enumerate incident widths and local positions, test domain faces/edges/corners, compare both ends of every canonical face, and change distant ownership to exercise compact renumbering. GPU validation should compare CPU/GPU builder topology and preserve the numerical lanes for surface resolution, sharpening, volume conservation, symmetry, solids and dynamic churn. Shader compilation must be checked in actual consumers because common source is assembled and rewritten for multiple namespaces and storage modes.

The clean-repository gate remains `npm run check:types`, `npm run test:unit` and `npm run test:dawn`. Dawn runs serially under the WebGPU lease. Do not weaken a lane or raise timing ceilings. Correctness checks are separate from performance experiments; no performance gain is claimed until a controlled comparison includes topology construction, value refresh, all consumers and dynamic adoption.
