/**
 * What a Sparse CM12 stage offers its lens.
 *
 * The framework in `lib/core/stage-lens.ts` is method-agnostic on purpose: it
 * knows a row glyph sits somewhere and is some colour, and nothing about arenas
 * or banks or generations. This is the other half — the CM12 vocabulary a lens
 * is written in, in one place so that eighteen lenses cannot each invent their
 * own name for the divergence field.
 *
 * Three things live here:
 *
 * - **Publications.** The catalogue of buffers and ranges a lens may bind. A
 *   lens picks a subset; picking one that does not exist is a `tsc` error.
 * - **Addressing.** A small uniform carrying every base offset a classifier
 *   needs, written once per frame from the resident's own layout. Offsets are
 *   *runtime* facts — they move with the scene's cell and row counts — so they
 *   cannot be constants in the WGSL, and a lens that hard-coded one would be
 *   right for exactly one scene.
 * - **The prelude.** The accessor block, shared with the resident through
 *   `sparse-cm12-row-access.wgsl`, plus CM12's answer to the layer templates'
 *   placement contract.
 *
 * ## Parity is read on the GPU
 *
 * The face banks swap every step, and which one holds the accepted velocities
 * is a device-side fact. The host does not compute it and does not pass it: the
 * addressing block carries the *word index* of the frame-control parity word,
 * and the shader reads it. A host that guessed the parity would be right about
 * half the time and would look plausible either way, which is the worst
 * possible failure mode for an inspection view.
 */
import type { StagePublication, StagePublications } from "../../core/stage-lens";
import {
  createSparseCM12CellAccessWGSL,
  createSparseCM12LensPlacementWGSL,
  createSparseCM12RowAccessWGSL,
  sparseCM12PlainArenaReaders,
  SPARSE_CM12_CELL_PACKING_WGSL,
  type SparseCM12ArenaReaders,
} from "./sparse-cm12-row-access.wgsl";

/** Words in the addressing uniform. Eight `vec4u`. */
export const SPARSE_CM12_ADDRESSING_WORDS = 32;
export const SPARSE_CM12_ADDRESSING_BYTES = 4 * SPARSE_CM12_ADDRESSING_WORDS;

/**
 * The addressing block, in the order the host writes it.
 *
 * Grouped four to a line because a uniform's `vec4u` alignment makes any other
 * grouping a lie about the layout. The names are the stage vocabulary; the
 * comments say which frame each number is in, because CM12 has three of them
 * (float indices into `state`, word indices into `arena`, and fine-lattice
 * cells) and mixing two is the classic wrong picture.
 */
export const SPARSE_CM12_ADDRESSING_WGSL = `struct SparseCM12Addressing {
  // x rows, y cells, z bricks, w arena word holding the accepted face parity
  counts:vec4u,
  // x live face bank base in state floats, y bank stride in floats, z cell
  // field stride in floats, w unused
  faces:vec4u,
  // Float bases in state: x pressure, y rhs, z divergence, w diagonal
  scalars:vec4u,
  // Float bases in state: x liquid, y theta, z residual, w applied
  auxiliary:vec4u,
  // Arena word bases for the face-projection authority: x stage header,
  // y accepted active bits, z candidate generation, w candidate cause
  projection0:vec4u,
  // x candidate depth, y execution generation, z accepted pressure bits,
  // w leaves in the authority's tree
  projection1:vec4u,
  // Arena word bases for velocity extension: x root cause, y root stamp,
  // z accepted depth, w unused
  extension:vec4u,
  // x cell velocity bank A, y bank B, both in state floats; z the arena word
  // holding the accepted scalar parity, w unused. Scalar parity is a different
  // word from face parity and flips on a different schedule, so a classifier
  // that reached for counts.w here would be wrong on exactly the frames a
  // bank swap matters.
  velocity:vec4u,
}`;

/**
 * Every buffer or range a CM12 lens may bind, by key.
 *
 * The scalar fields are carved out as their own ranges rather than left as
 * offsets into `state` for one reason: a tap has to copy something, and copying
 * the whole state arena to see one field's before-and-after would be tens of
 * megabytes per seam. A range is also self-indexing — `divergence[cell]` in a
 * snapshot means the same as in the live buffer — so a classifier reads the
 * same way whether it is looking at now or at then.
 */
export const SPARSE_CM12_STAGE_PUBLICATIONS = Object.freeze({
  /** The whole topology arena: row and cell records, and every authority bank. */
  arena: { kind: "read-only-storage", wgslType: "array<u32>", label: "Topology arena" },
  /** Base offsets for this frame's layout. Always bind this. */
  addressing: { kind: "uniform", wgslType: "SparseCM12Addressing",
    wgslStruct: SPARSE_CM12_ADDRESSING_WGSL, label: "Field offsets" },
  /** Both face banks, back to back. Index `bank*addressing.faces.y + row`. */
  faces: { kind: "read-only-storage", wgslType: "array<f32>", label: "Face velocities" },
  pressure: { kind: "read-only-storage", wgslType: "array<f32>", label: "Pressure" },
  rhs: { kind: "read-only-storage", wgslType: "array<f32>", label: "Right-hand side" },
  divergence: { kind: "read-only-storage", wgslType: "array<f32>", label: "Divergence" },
  liquid: { kind: "read-only-storage", wgslType: "array<f32>", label: "Liquid fraction" },
  /** The full float arena, for a field with no carved range of its own. */
  state: { kind: "read-only-storage", wgslType: "array<f32>", label: "State arena" },
} as const satisfies StagePublications);

export type SparseCM12PublicationKey = keyof typeof SPARSE_CM12_STAGE_PUBLICATIONS;

/**
 * The subset of the catalogue a lens binds.
 *
 * A helper rather than a hand-written object literal so a lens cannot declare a
 * publication with the wrong WGSL type — the type comes from the catalogue, and
 * only the *choice* of keys comes from the lens.
 */
export function sparseCM12Publications<const Keys extends readonly SparseCM12PublicationKey[]>(
  ...keys: Keys
): { readonly [Key in Keys[number]]: (typeof SPARSE_CM12_STAGE_PUBLICATIONS)[Key] } {
  const picked: Record<string, StagePublication> = {};
  for (const key of keys) picked[key] = SPARSE_CM12_STAGE_PUBLICATIONS[key];
  return Object.freeze(picked) as never;
}

/** Everything the host has to know to fill the addressing block. */
export interface SparseCM12AddressingSpec {
  readonly rowCount: number;
  readonly cellCount: number;
  readonly brickCount: number;
  /** Word index in the arena of the frame-control face parity word. */
  readonly faceParityWord: number;
  /** Float base of bank A. Bank B is one stride further on. */
  readonly faceBase: number;
  /** Floats between the two face banks, which is `align4(rowCount)`. */
  readonly faceBankStride: number;
  /** Floats between two consecutive cell fields, which is `align4(cellCount)`. */
  readonly cellFieldStride: number;
  readonly pressure: number;
  readonly rhs: number;
  readonly divergence: number;
  readonly diagonal: number;
  readonly liquid: number;
  readonly theta: number;
  readonly residual: number;
  readonly applied: number;
  readonly projectionHeaderWords: number;
  readonly projectionActiveBitsWords: number;
  readonly projectionCandidateGenerationWords: number;
  readonly projectionCandidateCauseWords: number;
  readonly projectionCandidateDepthWords: number;
  readonly projectionExecutionGenerationWords: number;
  readonly projectionAcceptedPressureBitsWords: number;
  readonly projectionLeafCount: number;
  readonly extensionRootCauseWords: number;
  readonly extensionRootStampWords: number;
  readonly extensionAcceptedDepthWords: number;
  readonly cellVelocityA: number;
  readonly cellVelocityB: number;
  /** Word index in the arena of the frame-control scalar parity word. */
  readonly scalarParityWord: number;
}

/** Fill the addressing block. The only place its word order is written. */
export function writeSparseCM12Addressing(
  target: Uint32Array,
  spec: SparseCM12AddressingSpec,
): void {
  if (target.length < SPARSE_CM12_ADDRESSING_WORDS) {
    throw new RangeError("Sparse CM12 addressing target is smaller than the block");
  }
  target.set([spec.rowCount, spec.cellCount, spec.brickCount, spec.faceParityWord], 0);
  target.set([spec.faceBase, spec.faceBankStride, spec.cellFieldStride, 0], 4);
  target.set([spec.pressure, spec.rhs, spec.divergence, spec.diagonal], 8);
  target.set([spec.liquid, spec.theta, spec.residual, spec.applied], 12);
  target.set([spec.projectionHeaderWords, spec.projectionActiveBitsWords,
    spec.projectionCandidateGenerationWords, spec.projectionCandidateCauseWords], 16);
  target.set([spec.projectionCandidateDepthWords, spec.projectionExecutionGenerationWords,
    spec.projectionAcceptedPressureBitsWords, spec.projectionLeafCount], 20);
  target.set([spec.extensionRootCauseWords, spec.extensionRootStampWords,
    spec.extensionAcceptedDepthWords, 0], 24);
  target.set([spec.cellVelocityA, spec.cellVelocityB, spec.scalarParityWord, 0], 28);
}

/**
 * The CM12 half of a lens shader: constants, accessors, placement, and the
 * handful of authority reads every projection-family classifier wants.
 *
 * Emitted per program rather than shared, because the accessors index whatever
 * the program named its arena binding. `arena` and `addressing` are the two
 * publications a lens cannot do without; a program that omits either will not
 * compile, which is the intended failure.
 */
export function sparseCM12LensPreludeWGSL(options: {
  /** WGSL identifier of the `arena` binding. */
  readonly arena?: string;
  /** WGSL identifier of the `addressing` binding. */
  readonly addressing?: string;
  /** Fine cells along one brick edge, when the lens draws brick frames. */
  readonly brickFineResolution?: number;
} = {}): string {
  const arena = options.arena ?? "arena";
  const addressing = options.addressing ?? "addressing";
  const readers: SparseCM12ArenaReaders = sparseCM12PlainArenaReaders(arena);
  return [
    SPARSE_CM12_CELL_PACKING_WGSL,
    createSparseCM12CellAccessWGSL(readers),
    createSparseCM12RowAccessWGSL(readers),
    createSparseCM12LensPlacementWGSL({ readers }),
    `/**
 * The bank the finished step accepted, read from frame control rather than
 * passed in: the host does not know it, and a host that guessed would be
 * plausible and wrong half the time.
 */
fn cm12AcceptedFaceBank()->u32{
  let word=${addressing}.counts.w;
  if(word==0xffffffffu){return 0u;}
  return ${arena}[word]&1u;
}
fn cm12FaceIndex(row:u32,bank:u32)->u32{
  return bank*${addressing}.faces.y+row;
}
/** Word holding the candidate cause bitmask for a row, in the arena. */
fn cm12ProjectionCauseWord(row:u32)->u32{
  return ${addressing}.projection0.w+row;
}
fn cm12ProjectionCandidateGeneration(row:u32)->u32{
  return ${arena}[${addressing}.projection0.z+row];
}
fn cm12ProjectionExecutionGeneration(row:u32)->u32{
  return ${arena}[${addressing}.projection1.y+row];
}
/** Whether a row is in the accepted work set, from the authority's bitset. */
fn cm12ProjectionAccepted(row:u32)->bool{
  let word=${arena}[${addressing}.projection0.y+(row>>5u)];
  return (word&(1u<<(row&31u)))!=0u;
}
fn cm12ProjectionHeader(word:u32)->u32{
  return ${arena}[${addressing}.projection0.x+word];
}
fn cm12AcceptedScalarBank()->u32{
  let word=${addressing}.velocity.z;
  if(word==0xffffffffu){return 0u;}
  return ${arena}[word]&1u;
}
/** Float base of the accepted collocated cell velocity, four floats per cell. */
fn cm12CellVelocityBase()->u32{
  return select(${addressing}.velocity.x,${addressing}.velocity.y,
    cm12AcceptedScalarBank()==1u);
}`,
  ].join("\n");
}
