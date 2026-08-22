/**
 * Classifiers for the velocity-projection lens.
 *
 * Each function answers one question about one instance in one phase, and
 * nothing else: no camera, no slice, no colour. The framework owns all three,
 * which is why these read as a table of solver facts rather than as a shader.
 *
 * The palette indices are named here and their colours are named beside the
 * phases in `sparse-cm12-face-projection.lens.ts`. Keeping the two apart is
 * deliberate — a colour is a design decision and a class is a solver fact, and
 * the legend is built from the same pairing the shader uses, so a class can
 * never be drawn in a colour no legend row explains.
 */
import {
  SPARSE_CM12_FACE_PROJECTION_CAUSE,
  SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER,
} from "./sparse-cm12-face-projection-authority";

/** Scrubber positions, left to right. Shared by the shader and the phase list. */
export const SPARSE_CM12_PROJECTION_PHASE = Object.freeze({
  plan: 0, execute: 1, collocate: 2, before: 3, after: 4, roots: 5, accept: 6,
} as const);

/** Palette slots. Eight is the uniform's capacity and all eight are spoken for. */
export const SPARSE_CM12_PROJECTION_CLASS = Object.freeze({
  /** A row the stage did not touch. Present so the plan reads as a fraction. */
  idle: 0,
  /** Rooted as a candidate this generation. */
  planned: 1,
  /** In the accepted work set and executed this frame. */
  work: 2,
  /** Accepted, but its execution stamp is older: the authority reused it. */
  reused: 3,
  /** Rooted by closure — the seam rows. Lavender, never magenta. */
  seam: 4,
  /** Rooted by a pressure-bit change, which is the ordinary reason. */
  pressure: 5,
  /** Rooted by prepared-face or boundary bits. */
  boundary: 6,
  /** The change the stage made. Only the delta arrow wears it. */
  delta: 7,
} as const);

/** The class constants, shared by both row programs. */
const SPARSE_CM12_PROJECTION_CLASS_WGSL = `const PROJ_IDLE:u32=${SPARSE_CM12_PROJECTION_CLASS.idle}u;
const PROJ_PLANNED:u32=${SPARSE_CM12_PROJECTION_CLASS.planned}u;
const PROJ_WORK:u32=${SPARSE_CM12_PROJECTION_CLASS.work}u;
const PROJ_REUSED:u32=${SPARSE_CM12_PROJECTION_CLASS.reused}u;
const PROJ_SEAM:u32=${SPARSE_CM12_PROJECTION_CLASS.seam}u;
const PROJ_PRESSURE:u32=${SPARSE_CM12_PROJECTION_CLASS.pressure}u;
const PROJ_BOUNDARY:u32=${SPARSE_CM12_PROJECTION_CLASS.boundary}u;
const PROJ_DELTA:u32=${SPARSE_CM12_PROJECTION_CLASS.delta}u;`;

const H = SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER;
const CAUSE = SPARSE_CM12_FACE_PROJECTION_CAUSE;
const P = SPARSE_CM12_PROJECTION_PHASE;
const C = SPARSE_CM12_PROJECTION_CLASS;

/**
 * Row classifier.
 *
 * `beforeFaces` is the tap: the face bank as it stood before the execute
 * dispatch. The delta phase is the only place the two are subtracted, and it is
 * marked `derived` in the phase list so it draws dashed — a difference of two
 * measurements is not itself a measurement.
 */
export const sparseCM12ProjectionRowsWGSL = /* wgsl */ `
${SPARSE_CM12_PROJECTION_CLASS_WGSL}

fn projectionCandidateGeneration()->u32{
  return cm12ProjectionHeader(${H.candidateGeneration}u);
}
fn projectionAcceptedGeneration()->u32{
  return cm12ProjectionHeader(${H.acceptedGeneration}u);
}
fn projectionRooted(row:u32)->bool{
  let generation=arena[addressing.projection0.z+row];
  return generation!=0u&&generation==projectionCandidateGeneration();
}

/**
 * Which cause a row wears when it has several.
 *
 * Ordered by how much it explains rather than by bit position: a row rooted by
 * both a pressure change and a closure write is a seam row, because the closure
 * is why it is in the set at all and the pressure change is true of thousands
 * of its neighbours.
 */
fn projectionCauseClass(cause:u32)->u32{
  if((cause&${CAUSE.closure}u)!=0u){return PROJ_SEAM;}
  if((cause&${CAUSE.preparedFaceBits}u)!=0u){return PROJ_BOUNDARY;}
  if((cause&${CAUSE.boundary}u)!=0u){return PROJ_BOUNDARY;}
  if((cause&${CAUSE.pressureBits}u)!=0u){return PROJ_PRESSURE;}
  if(cause!=0u){return PROJ_PLANNED;}
  return PROJ_IDLE;
}

fn lensRow(row:u32,phase:u32)->LensRowGlyph{
  var glyph:LensRowGlyph;
  glyph.visible=true;
  glyph.swatch=PROJ_IDLE;
  glyph.values=vec3f(0.0);
  glyph.arrowSwatch=vec3u(PROJ_IDLE);
  glyph.arrows=0u;
  if(phase==${P.plan}u){
    glyph.swatch=select(PROJ_IDLE,PROJ_PLANNED,projectionRooted(row));
    return glyph;
  }
  if(phase==${P.execute}u){
    if(!cm12ProjectionAccepted(row)){return glyph;}
    glyph.swatch=select(PROJ_REUSED,PROJ_WORK,
      cm12ProjectionExecutionGeneration(row)==projectionAcceptedGeneration());
    return glyph;
  }
  if(phase==${P.roots}u){
    glyph.swatch=select(PROJ_IDLE,
      projectionCauseClass(arena[cm12ProjectionCauseWord(row)]),projectionRooted(row));
    return glyph;
  }
  glyph.visible=false;
  return glyph;
}`;

/**
 * The accept phase: what the execute dispatch actually changed.
 *
 * Both banks are taps rather than one tap and the live buffer, because the D4
 * horizontal pass runs later in the same stage and the frame-end faces are
 * therefore not the faces the projection wrote. Subtracting the live bank from
 * the captured one would attribute D4's work to the projection, which is the
 * kind of quietly-wrong attribution a lens exists to prevent.
 */
export const sparseCM12ProjectionDeltaWGSL = /* wgsl */ `
${SPARSE_CM12_PROJECTION_CLASS_WGSL}

fn lensRow(row:u32,phase:u32)->LensRowGlyph{
  var glyph:LensRowGlyph;
  glyph.visible=false;
  glyph.swatch=PROJ_IDLE;
  glyph.values=vec3f(0.0);
  glyph.arrowSwatch=vec3u(PROJ_IDLE);
  glyph.arrows=0u;
  if(phase!=${P.accept}u){return glyph;}
  let bank=cm12AcceptedFaceBank();
  let before=beforeFaces[cm12FaceIndex(row,bank)];
  let after=afterFaces[cm12FaceIndex(row,bank)];
  let delta=after-before;
  let changed=abs(delta)>1e-7;
  glyph.visible=true;
  // A row the projection left alone is drawn as its segment only. Three arrows
  // on an unchanged row would read as work that happened.
  glyph.swatch=select(PROJ_IDLE,PROJ_WORK,changed);
  glyph.values=vec3f(before,delta,after);
  glyph.arrowSwatch=vec3u(PROJ_IDLE,PROJ_DELTA,PROJ_WORK);
  glyph.arrows=select(0u,3u,changed);
  return glyph;
}`;

/**
 * Cell classifier for the two scalar phases.
 *
 * `b before` and `div after` share one program and therefore one reference
 * scale, which is the entire point of putting them next to each other: the
 * projection's job is to turn the first into the second, and a ramp that
 * renormalised per phase would make a converged solve and a diverged one look
 * the same.
 */
export const sparseCM12ProjectionCellsWGSL = /* wgsl */ `
fn lensCell(cell:u32,phase:u32)->LensCellTile{
  var tile:LensCellTile;
  tile.visible=false;
  tile.swatch=${C.idle}u;
  tile.scalar=0.0;
  tile.diverging=true;
  if(phase==${P.before}u){
    tile.visible=true;
    tile.scalar=rhs[cell];
    return tile;
  }
  if(phase==${P.after}u){
    tile.visible=true;
    tile.scalar=divergence[cell];
    return tile;
  }
  return tile;
}`;

/**
 * Cell glyphs for the collocation phase.
 *
 * `collocateAndDiagnose` is where the face field becomes a cell field, and the
 * only place in the frame where both representations of the same velocity
 * exist. Drawing the cell-centred result beside the face rows of the phase on
 * either side is what makes the collocation legible as a step rather than as an
 * implementation detail.
 */
export const sparseCM12ProjectionCellGlyphsWGSL = /* wgsl */ `
fn lensCellGlyph(cell:u32,phase:u32)->LensCellGlyph{
  var glyph:LensCellGlyph;
  glyph.visible=false;
  glyph.swatch=${C.work}u;
  glyph.vector=vec3f(0.0);
  glyph.mark=LENS_MARK_ARROW;
  if(phase!=${P.collocate}u){return glyph;}
  let base=cm12CellVelocityBase()+4u*cell;
  glyph.visible=true;
  glyph.vector=vec3f(state[base],state[base+1u],state[base+2u]);
  return glyph;
}`;
