/**
 * The four layer-kind render templates every stage lens draws through.
 *
 * A lens author writes a *classifier* — "this row is work, that one is reused,
 * here are its three arrow magnitudes" — and nothing else. Geometry, the slice
 * test, the lattice-to-metres mapping, the camera and the palette are the
 * framework's, written once here, because they are the parts that have nothing
 * to do with the stage and everything to do with being legible.
 *
 * ## Why the glyphs are built in the vertex stage
 *
 * A row has no pixel footprint of its own. A fragment program asked to paint
 * "the faces of this cell" would have to re-walk an incidence list per pixel;
 * a vertex program is handed the row index for free and the slice test is one
 * compare. That is the "face arrows cost nothing" result — an arena read in a
 * vertex shader — applied to every row-valued quantity rather than to velocity
 * alone. Rejected instances collapse to a degenerate triangle and cost the
 * rasteriser nothing.
 *
 * ## The three contracts
 *
 * 1. **Placement**, supplied by the *method*: where instance `i` is, in
 *    fine-lattice coordinates. The method owns its arena; nothing here knows
 *    what a row record looks like.
 * 2. **Classification**, supplied by the *lens*: what instance `i` means in
 *    phase `p`. Returns a palette class and up to three values.
 * 3. **Presentation**, supplied here: how a class and a value become pixels.
 *
 * Keeping (1) and (2) apart is what lets seventeen lenses share one set of
 * templates while each still reads its own stage's buffers.
 */
import { STAGE_LENS_UNIFORM_NAME, type LensLayerKind } from "./stage-lens";

/** Bytes of the framework's uniform block: ten `vec4` plus an eight-entry palette. */
export const STAGE_LENS_UNIFORM_BYTES = 288;

/** Palette entries the uniform holds. Mirrors `LENS_PALETTE_SIZE`. */
export const STAGE_LENS_PALETTE_ENTRIES = 8;

/**
 * The class a classifier returns to say "I do not know".
 *
 * Painted magenta, which in this app means *unknown or faulted* and never
 * "dirty" — see the dirty-overlay convention. A header whose magic does not
 * match, a tap that was never encoded, a row outside its own capacity: all of
 * them arrive here rather than as a plausible colour.
 */
export const STAGE_LENS_FAULT_CLASS = 255;

/**
 * Vertices one instance of each layer kind issues.
 *
 * Fixed per kind rather than per lens so the draw call is decided by the
 * template, not by what a classifier happens to return. An instance that draws
 * less than its budget collapses the remainder.
 */
export function stageLensLayerVertexCount(kind: LensLayerKind): number {
  switch (kind) {
    // One segment quad (6) plus three nine-vertex arrows.
    case "row-glyph": return 33;
    // Two triangles over the cell's cross-section on the slice.
    case "cell-tile": return 6;
    // An arrow, or a screen-space quad shaped by the fragment stage.
    case "cell-glyph": return 9;
    // Four edges of the cross-section rectangle, six vertices each.
    case "brick-frame": return 24;
  }
}

const U = STAGE_LENS_UNIFORM_NAME;

/**
 * The uniform block, the camera, the slice and the palette.
 *
 * Deliberately the same nine-`vec4` camera preamble the face-velocity overlay
 * writes, in the same order, so one host routine fills both and a lens and an
 * arrow field cannot disagree about where the camera is. The tenth `vec4` and
 * the palette are the lens's own.
 */
export const stageLensStructsWGSL = /* wgsl */ `
struct StageLensUniforms {
  // xyz camera position, w tan(halfFov)
  cameraPosition:vec4f,
  // xyz camera forward, w aspect
  cameraForward:vec4f,
  // xyz camera right, w global alpha
  cameraRight:vec4f,
  // xyz camera up, w unused
  cameraUp:vec4f,
  // xy viewport pixels, zw unused
  viewport:vec4f,
  // xyz container metres, w finest cell metres
  container:vec4f,
  // xyz fine-lattice domain extent, w the program's fixed reference scale
  domain:vec4f,
  // x slice axis code (0 off, 1 z, 2 x, 3 y, 4 volume), y snapped layer in fine
  // cells, z slab half-thickness in fine cells, w unused
  slice:vec4f,
  // x line half-width px, y arrow head px, z glyph half-size px, w tile inset
  style:vec4f,
  // x phase index, y instance stride, z instance count, w fault bits
  addressing:vec4u,
  palette:array<vec4f,${STAGE_LENS_PALETTE_ENTRIES}>,
}

const LENS_FAULT_CLASS:u32=${STAGE_LENS_FAULT_CLASS}u;
const LENS_FAULT_COLOR:vec3f=vec3f(0.882,0.306,0.800);
const LENS_NEAR_VIEW_M:f32=0.02;
// Below this a projected glyph has no direction left to draw.
const LENS_MINIMUM_SPAN_PIXELS:f32=1.2;

// Marks a cell glyph can take. Anything but an arrow is one screen-space quad
// shaped in the fragment stage, which is cheaper and rounder than a fan.
const LENS_MARK_ARROW:u32=0u;
const LENS_MARK_DOT:u32=1u;
const LENS_MARK_RING:u32=2u;
const LENS_MARK_CROSS:u32=3u;

/** Where an instance sits, in fine-lattice coordinates. Supplied by the method. */
struct LensRowPlacement {
  valid:bool,
  centreFine:vec3f,
  /** 0 = x, 1 = y, 2 = z: the axis the face's normal points along. */
  axis:u32,
  /** Centre-to-centre distance in fine cells — the local cell scale. */
  spanFine:f32,
}

struct LensBoxPlacement {
  valid:bool,
  lowerFine:vec3f,
  upperFine:vec3f,
}

/** What a lens says about one row in one phase. */
struct LensRowGlyph {
  visible:bool,
  /** Palette index of the face segment. */
  swatch:u32,
  /** Up to three magnitudes, in the program's reference-scale units. */
  values:vec3f,
  /** Palette index of each arrow, so a triplet reads pre / delta / post. */
  arrowSwatch:vec3u,
  /** How many values to draw, 0 to 3. */
  arrows:u32,
}

struct LensCellTile {
  visible:bool,
  swatch:u32,
  /** Drawn on the diverging ramp when diverging, else the swatch colour. */
  scalar:f32,
  diverging:bool,
}

struct LensCellGlyph {
  visible:bool,
  swatch:u32,
  /** World-frame direction and magnitude, in the reference-scale units. */
  vector:vec3f,
  mark:u32,
}

struct LensBrickFrame {
  visible:bool,
  swatch:u32,
}

struct LensVertexOut {
  @builtin(position) position:vec4f,
  @location(0) color:vec3f,
  @location(1) alpha:f32,
  /** Quad-local coordinates in [-1,1]; only a mark quad uses them. */
  @location(2) local:vec2f,
  @location(3) @interpolate(flat) mark:u32,
}

`;

/**
 * Presentation helpers, emitted after the binding preamble.
 *
 * Split from the structs because the framework uniform is *declared* by the
 * binding emitter, which needs its type to exist first and its helpers to come
 * after. Nothing here reaches the method contract: every call into a
 * `lensRowPlacement` or a classifier happens inside a layer template, which is
 * emitted last of all.
 */
export const stageLensHelpersWGSL = /* wgsl */ `
fn lensColor(swatch:u32)->vec3f {
  if (swatch>=LENS_FAULT_CLASS) { return LENS_FAULT_COLOR; }
  return ${U}.palette[min(swatch,${STAGE_LENS_PALETTE_ENTRIES}u-1u)].rgb;
}

fn lensClassAlpha(swatch:u32)->f32 {
  if (swatch>=LENS_FAULT_CLASS) { return 1.0; }
  return ${U}.palette[min(swatch,${STAGE_LENS_PALETTE_ENTRIES}u-1u)].a;
}

/**
 * A five-stop diverging ramp for a signed scalar against a fixed scale.
 *
 * Diverging rather than sequential because every scalar a stage publishes that
 * is worth a tile — a residual, a divergence receipt, a pressure — has a
 * meaningful zero, and a sequential ramp would hide it in the middle of a
 * gradient. Cool for negative, near-transparent at zero, warm for positive.
 */
fn lensDivergingColor(value:f32,scale:f32)->vec3f {
  let t=clamp(value/max(scale,1e-9),-1.0,1.0);
  if (t<0.0) { return mix(vec3f(0.94,0.95,0.93),vec3f(0.10,0.45,0.74),-t); }
  return mix(vec3f(0.94,0.95,0.93),vec3f(0.80,0.24,0.16),t);
}

/** 0 = x, 1 = y, 2 = z; 3 when the view is a volume and there is no slice. */
fn lensSliceAxis()->u32 {
  let code=u32(round(${U}.slice.x));
  if (code==1u) { return 2u; }
  if (code==2u) { return 0u; }
  if (code==3u) { return 1u; }
  return 3u;
}

/** Fine-lattice coordinate of the selected cell layer's centre. */
fn lensSlicePlane()->f32 { return ${U}.slice.y+0.5; }

fn lensComponent(v:vec3f,axis:u32)->f32 {
  if (axis==0u) { return v.x; }
  if (axis==1u) { return v.y; }
  return v.z;
}

fn lensUnitAxis(axis:u32)->vec3f {
  if (axis==0u) { return vec3f(1.0,0.0,0.0); }
  if (axis==1u) { return vec3f(0.0,1.0,0.0); }
  return vec3f(0.0,0.0,1.0);
}

/**
 * The in-plane axis of a face: the one that is neither the face's own normal
 * nor the slice normal. A face segment is drawn along it.
 */
fn lensInPlaneAxis(faceAxis:u32,sliceAxis:u32)->u32 {
  if (sliceAxis>2u) { return select(0u,1u,faceAxis==0u); }
  return 3u-faceAxis-sliceAxis;
}

/**
 * Fine-lattice coordinates are corner-origin and continuous: 0 is the domain
 * minimum on every axis, and the domain extent is its maximum. The container is
 * centred on x and z and grounded on y, which is the renderer's convention.
 */
fn lensWorld(fine:vec3f)->vec3f {
  let half=0.5*${U}.container.xyz;
  return fine/max(${U}.domain.xyz,vec3f(1.0))*${U}.container.xyz
    +vec3f(-half.x,0.0,-half.z);
}

fn lensViewSpace(point:vec3f)->vec3f {
  let relative=point-${U}.cameraPosition.xyz;
  return vec3f(dot(relative,${U}.cameraRight.xyz),
    dot(relative,${U}.cameraUp.xyz),dot(relative,${U}.cameraForward.xyz));
}

fn lensNdc(view:vec3f)->vec2f {
  let tangent=max(${U}.cameraPosition.w,1e-4);
  let depth=max(view.z,LENS_NEAR_VIEW_M);
  return vec2f(view.x/(depth*tangent*max(${U}.cameraForward.w,1e-4)),
    view.y/(depth*tangent));
}

/** Screen pixels of a world point, and whether it is in front of the camera. */
struct LensPixel { ok:bool, pixel:vec2f }

fn lensProject(world:vec3f)->LensPixel {
  let view=lensViewSpace(world);
  if (view.z<LENS_NEAR_VIEW_M) { return LensPixel(false,vec2f(0.0)); }
  let viewport=max(${U}.viewport.xy,vec2f(1.0));
  return LensPixel(true,lensNdc(view)*0.5*viewport);
}

fn lensClip(pixel:vec2f)->vec4f {
  let viewport=max(${U}.viewport.xy,vec2f(1.0));
  return vec4f(pixel/(0.5*viewport),0.0,1.0);
}

/** One vertex of a thick screen-space line between two pixels. */
fn lensSegmentPixel(a:vec2f,b:vec2f,halfWidth:f32,vertex:u32)->vec2f {
  let delta=b-a;
  let span=max(length(delta),1e-4);
  let forward=delta/span;
  let side=vec2f(-forward.y,forward.x);
  var quad=array<vec2f,6>(vec2f(0.0,-1.0),vec2f(1.0,-1.0),vec2f(1.0,1.0),
    vec2f(0.0,-1.0),vec2f(1.0,1.0),vec2f(0.0,1.0));
  let corner=quad[min(vertex,5u)];
  return mix(a,b,corner.x)+side*halfWidth*corner.y;
}

/** One vertex of a nine-vertex arrow: six for the shaft, three for the head. */
fn lensArrowPixel(tail:vec2f,head:vec2f,halfWidth:f32,headLength:f32,vertex:u32)->vec2f {
  let delta=head-tail;
  let span=max(length(delta),1e-4);
  let forward=delta/span;
  let side=vec2f(-forward.y,forward.x);
  let headPixels=min(headLength,0.55*span);
  let shaftHalf=min(halfWidth,0.4*max(headPixels,1e-4));
  let neck=head-forward*headPixels;
  if (vertex<6u) { return lensSegmentPixel(tail,neck,shaftHalf,vertex); }
  if (vertex==6u) { return head; }
  if (vertex==7u) { return neck+side*2.6*shaftHalf; }
  return neck-side*2.6*shaftHalf;
}

/** A screen-space quad of half-size radius around a pixel, in [-1,1] local. */
fn lensMarkQuad(centre:vec2f,radius:f32,vertex:u32)->vec2f {
  var quad=array<vec2f,6>(vec2f(-1.0,-1.0),vec2f(1.0,-1.0),vec2f(1.0,1.0),
    vec2f(-1.0,-1.0),vec2f(1.0,1.0),vec2f(-1.0,1.0));
  return quad[min(vertex,5u)];
}

/**
 * The two axes of the slice plane, and the box's cross-section corners on it.
 *
 * A cell is a box; what it looks like on the slice is the rectangle its two
 * in-plane extents cut, held at the plane coordinate. Projecting the four
 * corners independently rather than projecting a centre and expanding keeps
 * the tile correct under perspective, which matters at the near edge of a
 * 128³ domain seen from inside it.
 */
fn lensBoxCornerFine(box:LensBoxPlacement,sliceAxis:u32,corner:u32)->vec3f {
  var point=box.lowerFine;
  var u=0u;
  var v=1u;
  if (sliceAxis==0u) { u=1u; v=2u; }
  else if (sliceAxis==1u) { u=0u; v=2u; }
  else { u=0u; v=1u; }
  // Corners in ring order, so consecutive pairs are edges.
  let takeUpperU=(corner==1u||corner==2u);
  let takeUpperV=(corner==2u||corner==3u);
  var out=point;
  if (u==0u) { out.x=select(box.lowerFine.x,box.upperFine.x,takeUpperU); }
  else if (u==1u) { out.y=select(box.lowerFine.y,box.upperFine.y,takeUpperU); }
  else { out.z=select(box.lowerFine.z,box.upperFine.z,takeUpperU); }
  if (v==0u) { out.x=select(box.lowerFine.x,box.upperFine.x,takeUpperV); }
  else if (v==1u) { out.y=select(box.lowerFine.y,box.upperFine.y,takeUpperV); }
  else { out.z=select(box.lowerFine.z,box.upperFine.z,takeUpperV); }
  if (sliceAxis<3u) {
    let plane=lensSlicePlane();
    if (sliceAxis==0u) { out.x=plane; }
    else if (sliceAxis==1u) { out.y=plane; }
    else { out.z=plane; }
  }
  return out;
}

/** Whether a box straddles the selected layer. Volume views accept everything. */
fn lensBoxOnSlice(box:LensBoxPlacement,sliceAxis:u32)->bool {
  if (sliceAxis>2u) { return true; }
  let plane=lensSlicePlane();
  let low=lensComponent(box.lowerFine,sliceAxis);
  let high=lensComponent(box.upperFine,sliceAxis);
  return low<=plane&&plane<high;
}

/**
 * Whether a face lies in the selected layer.
 *
 * Tested against the *cell's* half-span rather than a fixed tolerance, because
 * a coarse face and a fine face on either side of a 2:1 seam belong to layers
 * of different thickness and a constant would either drop one or double-draw
 * the other.
 */
fn lensRowOnSlice(row:LensRowPlacement,sliceAxis:u32)->bool {
  if (sliceAxis>2u) { return true; }
  let plane=lensSlicePlane();
  let centre=lensComponent(row.centreFine,sliceAxis);
  return abs(centre-plane)<=max(0.5*row.spanFine,${U}.slice.z);
}

/** A rejected instance: one degenerate triangle the rasteriser drops. */
fn lensRejected()->LensVertexOut {
  return LensVertexOut(vec4f(0.0,0.0,0.0,1.0),vec3f(0.0),0.0,vec2f(0.0),0u);
}

@fragment fn fragmentMain(input:LensVertexOut)->@location(0) vec4f {
  if (input.alpha<=0.0) { discard; }
  if (input.mark==LENS_MARK_DOT) {
    if (length(input.local)>1.0) { discard; }
  } else if (input.mark==LENS_MARK_RING) {
    let radius=length(input.local);
    if (radius>1.0||radius<0.58) { discard; }
  } else if (input.mark==LENS_MARK_CROSS) {
    let arm=min(abs(input.local.x-input.local.y),abs(input.local.x+input.local.y));
    if (arm>0.34||length(input.local)>1.0) { discard; }
  }
  return vec4f(input.color,input.alpha);
}
`;

/**
 * The row-glyph template: a face segment, and up to three arrows along the
 * face's own axis.
 *
 * The triplet is the whole reason this kind exists. "Pre minus delta equals
 * post" is a claim about one face at one moment, and the only honest way to
 * draw it is three arrows from the same origin, offset across the face so they
 * do not overlap. The offsets are in *fine cells* rather than pixels so the
 * triplet stays inside its own cell at every zoom, which is what keeps a dense
 * field readable as a field rather than as overlapping streaks.
 *
 * A face whose normal is the slice normal has nothing in-plane to draw; it gets
 * a dot at its centre, so "there are faces here pointing through the screen" is
 * still visible rather than silently absent.
 */
export const stageLensRowGlyphWGSL = /* wgsl */ `
@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex:u32,
  @builtin(instance_index) instance:u32,
)->LensVertexOut {
  let row=instance*max(${U}.addressing.y,1u);
  if (row>=${U}.addressing.z) { return lensRejected(); }
  let placement=lensRowPlacement(row);
  if (!placement.valid) { return lensRejected(); }
  let sliceAxis=lensSliceAxis();
  if (!lensRowOnSlice(placement,sliceAxis)) { return lensRejected(); }

  let glyph=lensRow(row,${U}.addressing.x);
  if (!glyph.visible) { return lensRejected(); }

  let centre=lensWorld(placement.centreFine);
  let normal=lensUnitAxis(placement.axis);
  let inPlane=lensInPlaneAxis(placement.axis,sliceAxis);
  let alpha=clamp(${U}.cameraRight.w,0.0,1.0);

  // A face pointing through the slice: no in-plane extent, so a dot at its
  // centre rather than a segment collapsed to nothing.
  if (sliceAxis<3u&&placement.axis==sliceAxis) {
    if (vertexIndex>=6u) { return lensRejected(); }
    let projected=lensProject(centre);
    if (!projected.ok) { return lensRejected(); }
    let local=lensMarkQuad(projected.pixel,${U}.style.z,vertexIndex);
    var output:LensVertexOut;
    output.position=lensClip(projected.pixel+local*0.6*${U}.style.z);
    output.color=lensColor(glyph.swatch);
    output.alpha=alpha*lensClassAlpha(glyph.swatch);
    output.local=local;
    output.mark=LENS_MARK_DOT;
    return output;
  }

  // The face segment: the face's own extent, seen edge-on in the slice.
  if (vertexIndex<6u) {
    let half=0.5*placement.spanFine*lensUnitAxis(inPlane);
    let a=lensProject(lensWorld(placement.centreFine-half));
    let b=lensProject(lensWorld(placement.centreFine+half));
    if (!a.ok||!b.ok) { return lensRejected(); }
    var output:LensVertexOut;
    output.position=lensClip(
      lensSegmentPixel(a.pixel,b.pixel,${U}.style.x,vertexIndex));
    output.color=lensColor(glyph.swatch);
    output.alpha=alpha*lensClassAlpha(glyph.swatch);
    output.local=vec2f(0.0);
    output.mark=LENS_MARK_ARROW;
    return output;
  }

  let arrow=(vertexIndex-6u)/9u;
  if (arrow>=glyph.arrows) { return lensRejected(); }
  var value=glyph.values.x;
  var swatch=glyph.arrowSwatch.x;
  if (arrow==1u) { value=glyph.values.y; swatch=glyph.arrowSwatch.y; }
  else if (arrow==2u) { value=glyph.values.z; swatch=glyph.arrowSwatch.z; }

  // Length saturates at half the local cell, for the same reason the face
  // arrows do: an arrow that overruns its own cell reads as a streak into the
  // neighbour it never touched.
  let unit=clamp(value/max(${U}.domain.w,1e-9),-1.0,1.0);
  let lengthFine=unit*0.5*placement.spanFine;
  // The three arrows are offset across the face so a triplet reads as three
  // measurements of one quantity rather than as three faces.
  let lane=(f32(arrow)-1.0)*0.22*placement.spanFine;
  let base=placement.centreFine+lane*lensUnitAxis(inPlane);
  let tail=lensProject(lensWorld(base));
  let head=lensProject(lensWorld(base+lengthFine*normal));
  if (!tail.ok||!head.ok) { return lensRejected(); }
  if (length(head.pixel-tail.pixel)<LENS_MINIMUM_SPAN_PIXELS) { return lensRejected(); }

  var output:LensVertexOut;
  output.position=lensClip(lensArrowPixel(tail.pixel,head.pixel,
    ${U}.style.x,${U}.style.y,(vertexIndex-6u)%9u));
  output.color=lensColor(swatch);
  output.alpha=alpha*lensClassAlpha(swatch);
  output.local=vec2f(0.0);
  output.mark=LENS_MARK_ARROW;
  return output;
}
`;

/**
 * The cell-tile template: one filled quad over the cell's cross-section.
 *
 * The inset is not decoration. Tiles that meet exactly form an unbroken sheet
 * in which a 2:1 seam is invisible, and the seam is usually the thing being
 * looked at; a fraction of a cell of gap makes the adaptive structure legible
 * without drawing a single grid line.
 */
export const stageLensCellTileWGSL = /* wgsl */ `
@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex:u32,
  @builtin(instance_index) instance:u32,
)->LensVertexOut {
  let cell=instance*max(${U}.addressing.y,1u);
  if (cell>=${U}.addressing.z) { return lensRejected(); }
  let box=lensCellPlacement(cell);
  if (!box.valid) { return lensRejected(); }
  let sliceAxis=lensSliceAxis();
  if (!lensBoxOnSlice(box,sliceAxis)) { return lensRejected(); }

  let tile=lensCell(cell,${U}.addressing.x);
  if (!tile.visible) { return lensRejected(); }

  let inset=clamp(${U}.style.w,0.0,0.4);
  let centreFine=0.5*(box.lowerFine+box.upperFine);
  var indices=array<u32,6>(0u,1u,2u,0u,2u,3u);
  let cornerIndex=indices[min(vertexIndex,5u)];
  let corner=lensBoxCornerFine(box,sliceAxis,cornerIndex);
  let shrunk=mix(corner,centreFine,inset);
  let projected=lensProject(lensWorld(shrunk));
  if (!projected.ok) { return lensRejected(); }

  var color=lensColor(tile.swatch);
  if (tile.diverging) { color=lensDivergingColor(tile.scalar,${U}.domain.w); }
  var output:LensVertexOut;
  output.position=lensClip(projected.pixel);
  output.color=color;
  output.alpha=clamp(${U}.cameraRight.w,0.0,1.0)*lensClassAlpha(tile.swatch);
  output.local=vec2f(0.0);
  output.mark=LENS_MARK_ARROW;
  return output;
}
`;

/**
 * The cell-glyph template: an arrow from the cell centre, or a mark on it.
 *
 * An arrow here is a *collocated* vector — a claim about a quantity the stage
 * reconstructed at the cell centre, not about a face. That is why it is a
 * separate kind from the row glyph rather than a mode of it: drawing a
 * reconstruction and a measurement in the same shape would make them look like
 * the same kind of fact.
 */
export const stageLensCellGlyphWGSL = /* wgsl */ `
@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex:u32,
  @builtin(instance_index) instance:u32,
)->LensVertexOut {
  let cell=instance*max(${U}.addressing.y,1u);
  if (cell>=${U}.addressing.z) { return lensRejected(); }
  let box=lensCellPlacement(cell);
  if (!box.valid) { return lensRejected(); }
  let sliceAxis=lensSliceAxis();
  if (!lensBoxOnSlice(box,sliceAxis)) { return lensRejected(); }

  let glyph=lensCellGlyph(cell,${U}.addressing.x);
  if (!glyph.visible) { return lensRejected(); }

  var centreFine=0.5*(box.lowerFine+box.upperFine);
  if (sliceAxis<3u) {
    let plane=lensSlicePlane();
    if (sliceAxis==0u) { centreFine.x=plane; }
    else if (sliceAxis==1u) { centreFine.y=plane; }
    else { centreFine.z=plane; }
  }
  let spanFine=max(lensComponent(box.upperFine-box.lowerFine,
    select(0u,sliceAxis,sliceAxis<3u)),1e-4);
  let alpha=clamp(${U}.cameraRight.w,0.0,1.0)*lensClassAlpha(glyph.swatch);
  let projected=lensProject(lensWorld(centreFine));
  if (!projected.ok) { return lensRejected(); }

  if (glyph.mark!=LENS_MARK_ARROW) {
    if (vertexIndex>=6u) { return lensRejected(); }
    let local=lensMarkQuad(projected.pixel,${U}.style.z,vertexIndex);
    var output:LensVertexOut;
    output.position=lensClip(projected.pixel+local*${U}.style.z);
    output.color=lensColor(glyph.swatch);
    output.alpha=alpha;
    output.local=local;
    output.mark=glyph.mark;
    return output;
  }

  // A collocated vector is drawn in the slice plane: the out-of-plane component
  // has no direction on screen, and projecting it would draw a length the
  // reader would read as in-plane speed.
  var planar=glyph.vector;
  if (sliceAxis==0u) { planar.x=0.0; }
  else if (sliceAxis==1u) { planar.y=0.0; }
  else if (sliceAxis==2u) { planar.z=0.0; }
  let magnitude=length(planar);
  if (magnitude<=1e-9) { return lensRejected(); }
  let unit=clamp(magnitude/max(${U}.domain.w,1e-9),0.0,1.0);
  let offsetFine=planar/magnitude*unit*0.5*spanFine;
  let tail=lensProject(lensWorld(centreFine-0.5*offsetFine));
  let head=lensProject(lensWorld(centreFine+0.5*offsetFine));
  if (!tail.ok||!head.ok) { return lensRejected(); }
  if (length(head.pixel-tail.pixel)<LENS_MINIMUM_SPAN_PIXELS) { return lensRejected(); }

  var output:LensVertexOut;
  output.position=lensClip(lensArrowPixel(tail.pixel,head.pixel,
    ${U}.style.x,${U}.style.y,min(vertexIndex,8u)));
  output.color=lensColor(glyph.swatch);
  output.alpha=alpha;
  output.local=vec2f(0.0);
  output.mark=LENS_MARK_ARROW;
  return output;
}
`;

/**
 * The brick-frame template: the outline of a resident brick on the slice.
 *
 * Outline rather than fill, always, because a brick frame is drawn *over* the
 * cells it contains and a filled one would hide the picture it is annotating.
 */
export const stageLensBrickFrameWGSL = /* wgsl */ `
@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex:u32,
  @builtin(instance_index) instance:u32,
)->LensVertexOut {
  let brick=instance*max(${U}.addressing.y,1u);
  if (brick>=${U}.addressing.z) { return lensRejected(); }
  let box=lensBrickPlacement(brick);
  if (!box.valid) { return lensRejected(); }
  let sliceAxis=lensSliceAxis();
  if (!lensBoxOnSlice(box,sliceAxis)) { return lensRejected(); }

  let frame=lensBrick(brick,${U}.addressing.x);
  if (!frame.visible) { return lensRejected(); }

  let edge=vertexIndex/6u;
  let a=lensProject(lensWorld(lensBoxCornerFine(box,sliceAxis,edge)));
  let b=lensProject(lensWorld(lensBoxCornerFine(box,sliceAxis,(edge+1u)%4u)));
  if (!a.ok||!b.ok) { return lensRejected(); }

  var output:LensVertexOut;
  output.position=lensClip(
    lensSegmentPixel(a.pixel,b.pixel,${U}.style.x,vertexIndex%6u));
  output.color=lensColor(frame.swatch);
  output.alpha=clamp(${U}.cameraRight.w,0.0,1.0)*lensClassAlpha(frame.swatch);
  output.local=vec2f(0.0);
  output.mark=LENS_MARK_ARROW;
  return output;
}
`;

/** The template one layer kind draws through. */
/** Structs then helpers, for a test that wants the whole framework prelude. */
export const stageLensPreludeWGSL = `${stageLensStructsWGSL}\n${stageLensHelpersWGSL}`;

export function stageLensLayerWGSL(kind: LensLayerKind): string {
  switch (kind) {
    case "row-glyph": return stageLensRowGlyphWGSL;
    case "cell-tile": return stageLensCellTileWGSL;
    case "cell-glyph": return stageLensCellGlyphWGSL;
    case "brick-frame": return stageLensBrickFrameWGSL;
  }
}
