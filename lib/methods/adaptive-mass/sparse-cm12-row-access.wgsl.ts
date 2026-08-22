/**
 * The one place that knows where a CM12 row or cell record lives.
 *
 * The resident solver, the face-velocity overlay and every stage lens decode
 * the same packed arena: nine planes of row data based at `arena[7]`, an
 * eight-word cell record based at `arena[6]`, a term list at `arena[8]`. Each
 * consumer used to carry its own copy of that arithmetic, and the tell for a
 * stale copy is a picture that is subtly wrong rather than a build that fails.
 *
 * Emitting the block gives one definition, so a plane that moves moves
 * everywhere at once. What differs between consumers is only *how a word is
 * read*: the resident binds the arena as `array<atomic<u32>>` and goes through
 * `ta`/`taf`, while a lens binds it read-only and indexes it directly. Moving a
 * float read on or off an atomic binding reassociates the surrounding math, so
 * that difference is a parameter here rather than a second copy of the bodies.
 */

/**
 * How a consumer turns an arena word index into a WGSL expression.
 *
 * Both take an *expression* rather than a number because every real call site
 * indexes off another accessor — `rowWord(id,2u)`, `cellBase(id)+7u`.
 */
export interface SparseCM12ArenaReaders {
  /** The u32 at `index`. */
  readonly word: (index: string) => string;
  /** The same word bitcast to f32. */
  readonly float: (index: string) => string;
}

/**
 * The resident's readers.
 *
 * `ta`/`taf` are its `atomicLoad` helpers over the arena binding. Emitting
 * through them reproduces the resident's historical text exactly, which is what
 * keeps adopting this module bit-exact.
 */
export const SPARSE_CM12_ATOMIC_ARENA_READERS: SparseCM12ArenaReaders = Object.freeze({
  word: (index: string) => `ta(${index})`,
  float: (index: string) => `taf(${index})`,
});

/** Readers for a plain `array<u32>` binding — every render-stage consumer. */
export function sparseCM12PlainArenaReaders(name: string): SparseCM12ArenaReaders {
  return Object.freeze({
    word: (index: string) => `${name}[${index}]`,
    float: (index: string) => `bitcast<f32>(${name}[${index}])`,
  });
}

/**
 * Cell-resolution packing constants.
 *
 * The resident declares these itself among its other template constants; a
 * lens has to be handed them. Emitted separately so the resident keeps the copy
 * it has always had rather than acquiring a second one.
 */
export const SPARSE_CM12_CELL_PACKING_WGSL = `const TEMPLATE_CELL_RESOLUTION_BITS:u32=5u;
const TEMPLATE_CELL_RESOLUTION_MASK:u32=31u;`;

/**
 * Cell record addressing.
 *
 * Eight words per cell: `[0..2]` fine-lattice centre, `3` volume, `[4..6]`
 * fine-lattice widths, `7` metadata carrying the brick index above the
 * resolution bits.
 */
export function createSparseCM12CellAccessWGSL(
  readers: SparseCM12ArenaReaders,
): string {
  const { word: w, float: f } = readers;
  return `fn cellBase(id:u32)->u32{return ${w("6u")}+id*8u;}
fn cellMetadata(id:u32)->u32{return ${w("cellBase(id)+7u")};}
fn cellBrick(id:u32)->u32{return cellMetadata(id)>>TEMPLATE_CELL_RESOLUTION_BITS;}
fn cellResolution(id:u32)->u32{return cellMetadata(id)&TEMPLATE_CELL_RESOLUTION_MASK;}
fn cellVolume(id:u32)->f32{return ${f("cellBase(id)+3u")};}
fn cellMinimumWidth(id:u32)->f32{
  let b=cellBase(id);return min(${f("b+4u")},min(${f("b+5u")},${f("b+6u")}));
}
fn cellMinimum(id:u32)->vec3u{
  let b=cellBase(id);let center=vec3f(${f("b")},${f("b+1u")},${f("b+2u")});
  let widths=vec3f(${f("b+4u")},${f("b+5u")},${f("b+6u")});
  return vec3u(center-0.5*widths);
}`;
}

/**
 * Row record addressing, plus the term and incidence lists rows index into.
 *
 * Nine planes, each `rowCapacity` long, based at `arena[7]` with the stride in
 * `arena[3]`: packed terms, packed metadata, static dual weight, static area,
 * distance, exterior phi, then three planes of fine-lattice centre.
 *
 * Nothing here depends on rigid bodies. The open-fraction family that scales
 * these by solid coverage stays with the solver, because a lens draws the
 * static geometry a stage was authored against, not the coupled geometry one
 * particular frame solved.
 */
export function createSparseCM12RowAccessWGSL(
  readers: SparseCM12ArenaReaders,
): string {
  const { word: w, float: f } = readers;
  return `fn rowWord(id:u32,plane:u32)->u32{return ${w("7u")}+plane*${w("3u")}+id;}
fn rowPackedTerms(id:u32)->u32{return ${w("rowWord(id,0u)")};}
fn rowPackedMetadata(id:u32)->u32{return ${w("rowWord(id,1u)")};}
fn rowTermOffset(id:u32)->u32{return rowPackedTerms(id)&0x007fffffu;}
fn rowTermCount(id:u32)->u32{return rowPackedTerms(id)>>23u;}
fn rowAxis(id:u32)->u32{return rowPackedMetadata(id)>>30u;}
fn rowKind(id:u32)->u32{return (rowPackedMetadata(id)>>28u)&3u;}
fn rowRequirementOffset(id:u32)->u32{return rowPackedMetadata(id)&0x0fffffffu;}
fn rowStaticDualWeight(id:u32)->f32{return ${f("rowWord(id,2u)")};}
fn rowStaticArea(id:u32)->f32{return ${f("rowWord(id,3u)")};}
fn rowDistance(id:u32)->f32{return ${f("rowWord(id,4u)")};}
fn rowExteriorPhi(id:u32)->f32{return ${f("rowWord(id,5u)")};}
fn rowCenter(id:u32)->vec3f{return vec3f(${f("rowWord(id,6u)")},${f("rowWord(id,7u)")},${f("rowWord(id,8u)")});}
fn termCell(index:u32)->u32{return ${w(`${w("8u")}+2u*index`)};}
fn termCoefficient(index:u32)->f32{return ${f(`${w("8u")}+2u*index+1u`)};}
fn incidenceBegin(cell:u32)->u32{return ${w(`${w("9u")}+cell`)};}
fn incidenceEnd(cell:u32)->u32{return ${w(`${w("9u")}+cell+1u`)};}
fn incidenceRow(index:u32)->u32{return ${w(`${w("10u")}+2u*index`)};}
fn incidenceTerm(index:u32)->u32{return ${w(`${w("10u")}+2u*index+1u`)};}`;
}

/**
 * CM12's answer to the placement contract the layer templates call.
 *
 * The templates in `stage-lens-layers.wgsl.ts` know that a row glyph sits at a
 * fine-lattice centre and runs along an axis, and nothing about where that came
 * from. This is the only CM12 code they ever reach.
 *
 * A row with zero static area is not drawn. Topology carries such rows — a face
 * fully inside a solid, a face outside the band — but no stage solves one, so
 * drawing it would put a glyph where the stage did nothing.
 *
 * `brick` is optional because a brick's fine-lattice origin is not in the
 * arena: it is a key in the separate topology buffer, and a lens that wants
 * brick frames has to bind that buffer and say where the key table sits. A lens
 * that declares a `brick-frame` program without supplying this fails to compile
 * its shader rather than drawing frames in the wrong places.
 */
export function createSparseCM12LensPlacementWGSL(options: {
  readonly readers: SparseCM12ArenaReaders;
  readonly brick?: {
    /** WGSL expression for the brick key of brick index `brick`. */
    readonly key: (brick: string) => string;
    /** WGSL `vec3u` expression for the brick grid dimensions. */
    readonly dimensions: string;
    /** Fine cells along one brick edge. */
    readonly fineResolution: number;
  };
}): string {
  const f = options.readers.float;
  const placement = `fn cellCentreFine(id:u32)->vec3f{
  let b=cellBase(id);
  return vec3f(${f("b")},${f("b+1u")},${f("b+2u")});
}
fn cellWidthsFine(id:u32)->vec3f{
  let b=cellBase(id);
  return vec3f(${f("b+4u")},${f("b+5u")},${f("b+6u")});
}
fn rowFaceCentreFine(id:u32)->vec3f{return rowCenter(id);}
fn lensRowPlacement(row:u32)->LensRowPlacement{
  var placement:LensRowPlacement;
  placement.valid=rowStaticArea(row)>0.0;
  placement.centreFine=rowFaceCentreFine(row);
  placement.axis=rowAxis(row);
  placement.spanFine=rowDistance(row);
  return placement;
}
fn lensCellPlacement(cell:u32)->LensBoxPlacement{
  var placement:LensBoxPlacement;
  let centre=cellCentreFine(cell);
  let widths=cellWidthsFine(cell);
  placement.valid=cellVolume(cell)>0.0;
  placement.lowerFine=centre-0.5*widths;
  placement.upperFine=centre+0.5*widths;
  return placement;
}`;
  if (!options.brick) return placement;
  const { key, dimensions, fineResolution } = options.brick;
  return `${placement}
fn lensBrickPlacement(brick:u32)->LensBoxPlacement{
  var placement:LensBoxPlacement;
  let dims=${dimensions};
  let plane=dims.x*dims.y;
  let brickKey=${key("brick")};
  let bz=brickKey/plane;
  let remainder=brickKey-bz*plane;
  let by=remainder/dims.x;
  let edge=f32(${fineResolution}u);
  placement.valid=bz<dims.z;
  placement.lowerFine=vec3f(f32(remainder-by*dims.x),f32(by),f32(bz))*edge;
  placement.upperFine=placement.lowerFine+vec3f(edge);
  return placement;
}`;
}
