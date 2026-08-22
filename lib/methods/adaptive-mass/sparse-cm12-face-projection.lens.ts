/**
 * The velocity-projection lens.
 *
 * Stage twelve of the Sparse CM12 advance subtracts the pressure gradient from
 * every face and then collocates the result onto cells. It is the stage the
 * "CM12 Frame Anatomy" page draws by hand, and the one worth drawing first:
 * everything upstream is preparation for it and everything downstream is a
 * consequence of it, so a wrong picture anywhere in the frame usually shows up
 * here as a row that should have moved and did not.
 *
 * Seven phases, left to right, each a viewpoint on the same executed stage:
 *
 * | | phase | what it answers |
 * |---|---|---|
 * | 0 | plan | which rows the authority rooted this generation |
 * | 1 | execute | which of those it ran, and which it reused |
 * | 2 | collocate | the cell-centred field the face field became |
 * | 3 | b before | the right-hand side the solve was given |
 * | 4 | div after | the divergence that survived it |
 * | 5 | roots | *why* each rooted row was rooted |
 * | 6 | accept | what the execute dispatch actually changed |
 *
 * Phases 3 and 4 share one program and therefore one fixed reference scale.
 * That is the whole reason they are adjacent: the projection's job is to turn
 * the first into the second, and a ramp that renormalised between them would
 * make a converged solve and a diverged one look identical.
 *
 * Seam rows are lavender. The reference page draws them magenta, but magenta is
 * this codebase's fault-and-unknown colour everywhere else, and a seam is a
 * perfectly ordinary thing for a row to be.
 */
import { stageLens } from "../../core/stage-lens";
import {
  SPARSE_CM12_FACE_PROJECTION_HEADER_WORDS,
  SPARSE_CM12_FACE_PROJECTION_MAGIC,
  SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER,
} from "./sparse-cm12-face-projection-authority";
import {
  sparseCM12LensPreludeWGSL,
  sparseCM12Publications,
} from "./sparse-cm12-stage-contract";
import {
  SPARSE_CM12_PROJECTION_CLASS,
  sparseCM12ProjectionCellGlyphsWGSL,
  sparseCM12ProjectionCellsWGSL,
  sparseCM12ProjectionDeltaWGSL,
  sparseCM12ProjectionRowsWGSL,
} from "./sparse-cm12-face-projection.lens.wgsl";

/**
 * Header words, offset past the authority's own header.
 *
 * The readback covers both headers so the magic at word zero can be checked:
 * the stage header alone is thirty-two plausible integers, and a base offset
 * that moved would read as a stage in a strange state rather than as the wrong
 * region entirely. Derived from the frozen table rather than retyped, so a word
 * that moves moves here too.
 */
const HEADER_WORDS = Object.freeze(Object.fromEntries(
  Object.entries(SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER).map(
    ([name, offset]) => [name, offset + SPARSE_CM12_FACE_PROJECTION_HEADER_WORDS]),
)) as { readonly [Word in keyof typeof SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER]: number };

/**
 * Class colours.
 *
 * Indexed by `SPARSE_CM12_PROJECTION_CLASS`, so the array's order is the
 * shader's vocabulary and the legend below is built from the same pairing.
 */
const PALETTE: readonly string[] = Object.freeze([
  "#5a647266", // idle — present so the plan reads as a fraction of the whole
  "#e8a13a",   // planned
  "#5cc46a",   // work
  "#4a8fd6",   // reused
  "#a78bfa",   // seam
  "#3fb8cf",   // pressure
  "#e8763a",   // boundary
  "#e0473a",   // delta
]);

const prelude = sparseCM12LensPreludeWGSL();

/**
 * Full scale for a scalar tile, in the units the solver stores.
 *
 * `rhs` and `divergence` are both cell-volume-weighted velocity divergences, so
 * one number serves both. Chosen so a converged frame's divergence field is
 * nearly flat and an unconverged one is not: an inspection scale that puts
 * everything mid-ramp shows nothing.
 */
const SCALAR_FULL_SCALE = 1e-3;

export const SPARSE_CM12_FACE_PROJECTION_LENS = stageLens({
  stage: "velocity-projection",
  id: "stage-lens/velocity-projection",
  label: "Velocity projection",
  description:
    "Stage twelve: the pressure gradient comes off every face and the result is collocated onto cells. The scrubber walks the stage's own account of itself — what it planned, what it ran, what it reused, and what it changed — drawn from the authority's per-row banks rather than reconstructed from the finished frame.",
  publications: sparseCM12Publications(
    "arena", "addressing", "faces", "rhs", "divergence", "state"),
  taps: {
    /**
     * The face banks as the execute dispatch found them, and the right-hand
     * side the solve was given.
     *
     * `rhs` is a tap rather than a live read because `candidate-transfer`, four
     * stages downstream, zeroes it on every cell whose topology changed this
     * frame. Read live, the b field would be correct everywhere except exactly
     * where the frame was interesting.
     */
    beforeExecute: ["faces", "rhs"],
    /**
     * And as it left them, before the D4 horizontal pass later in the same
     * stage touches them again.
     */
    afterExecute: ["faces"],
    /**
     * The divergence `collocateAndDiagnose` just wrote — what survived the
     * projection. A tap for the same reason `rhs` is one.
     */
    afterCollocate: ["divergence"],
  },
  header: {
    resource: "arena",
    words: HEADER_WORDS,
    wordCount: 2 * SPARSE_CM12_FACE_PROJECTION_HEADER_WORDS,
    magic: { word: 0, value: SPARSE_CM12_FACE_PROJECTION_MAGIC },
  },
  programs: {
    rows: {
      kind: "row-glyph",
      label: "Rows",
      bindings: [
        { name: "arena", kind: "read-only-storage", type: "array<u32>", resource: "arena" },
        { name: "addressing", kind: "uniform", type: "SparseCM12Addressing", resource: "addressing" },
      ],
      palette: PALETTE,
      wgsl: () => `${prelude}\n${sparseCM12ProjectionRowsWGSL}`,
    },
    delta: {
      kind: "row-glyph",
      label: "Change",
      bindings: [
        { name: "arena", kind: "read-only-storage", type: "array<u32>", resource: "arena" },
        { name: "addressing", kind: "uniform", type: "SparseCM12Addressing", resource: "addressing" },
        { name: "beforeFaces", kind: "read-only-storage", type: "array<f32>",
          resource: "tap:beforeExecute:faces" },
        { name: "afterFaces", kind: "read-only-storage", type: "array<f32>",
          resource: "tap:afterExecute:faces" },
      ],
      palette: PALETTE,
      wgsl: () => `${prelude}\n${sparseCM12ProjectionDeltaWGSL}`,
    },
    cells: {
      kind: "cell-tile",
      label: "Scalars",
      bindings: [
        { name: "arena", kind: "read-only-storage", type: "array<u32>", resource: "arena" },
        { name: "addressing", kind: "uniform", type: "SparseCM12Addressing", resource: "addressing" },
        { name: "rhs", kind: "read-only-storage", type: "array<f32>",
          resource: "tap:beforeExecute:rhs" },
        { name: "divergence", kind: "read-only-storage", type: "array<f32>",
          resource: "tap:afterCollocate:divergence" },
      ],
      palette: PALETTE,
      scale: SCALAR_FULL_SCALE,
      wgsl: () => `${prelude}\n${sparseCM12ProjectionCellsWGSL}`,
    },
    cellGlyphs: {
      kind: "cell-glyph",
      label: "Collocated",
      bindings: [
        { name: "arena", kind: "read-only-storage", type: "array<u32>", resource: "arena" },
        { name: "addressing", kind: "uniform", type: "SparseCM12Addressing", resource: "addressing" },
        { name: "state", kind: "read-only-storage", type: "array<f32>", resource: "state" },
      ],
      palette: PALETTE,
      wgsl: () => `${prelude}\n${sparseCM12ProjectionCellGlyphsWGSL}`,
    },
  },
  phases: [
    {
      id: "plan", label: "Plan",
      layers: [{ program: "rows", evidence: "live" }],
      counters: [
        { word: "dirtyLeafCount", lit: true },
        { word: "activeLeafCount" },
        { word: "previousActiveLeafCount" },
        { word: "candidateGeneration" },
      ],
    },
    {
      id: "execute", label: "Execute",
      layers: [{ program: "rows", evidence: "live" }],
      counters: [
        { word: "workCount", lit: true },
        { word: "executedCount", lit: true },
        { word: "reusedCount" },
      ],
    },
    {
      id: "collocate", label: "Collocate",
      layers: [{ program: "cellGlyphs", evidence: "live" }],
      counters: [
        { word: "directWriteCount", lit: true },
        { word: "closureWriteCount" },
      ],
    },
    {
      id: "before", label: "b before",
      layers: [{ program: "cells", evidence: "captured" }],
      counters: [
        { word: "expectedProducerReceipts" },
        { word: "coveredProducerReceipts" },
      ],
    },
    {
      id: "after", label: "div after",
      layers: [{ program: "cells", evidence: "captured" }],
      counters: [
        { word: "verifiedLeafCount", lit: true },
        { word: "activeLeafCount" },
      ],
    },
    {
      id: "roots", label: "Roots",
      layers: [{ program: "rows", evidence: "live" }],
      counters: [
        { word: "causeMask", lit: true },
        { word: "candidateGeneration" },
        { word: "topologyGeneration" },
        { word: "pcmGeneration" },
      ],
    },
    {
      id: "accept", label: "Accept",
      // A difference of two captures is not itself a capture.
      layers: [{ program: "delta", evidence: "derived" }],
      counters: [
        { word: "acceptedGeneration", lit: true },
        { word: "fault" },
        { word: "firstFaultRow" },
        { word: "phase" },
      ],
    },
  ],
  legend: [
    { swatch: PALETTE[SPARSE_CM12_PROJECTION_CLASS.planned]!, mark: "line", label: "rooted this generation" },
    { swatch: PALETTE[SPARSE_CM12_PROJECTION_CLASS.work]!, mark: "line", label: "executed" },
    { swatch: PALETTE[SPARSE_CM12_PROJECTION_CLASS.reused]!, mark: "line", label: "accepted, reused" },
    { swatch: PALETTE[SPARSE_CM12_PROJECTION_CLASS.pressure]!, mark: "line", label: "rooted by pressure" },
    { swatch: PALETTE[SPARSE_CM12_PROJECTION_CLASS.boundary]!, mark: "line", label: "rooted by a face or boundary bit" },
    { swatch: PALETTE[SPARSE_CM12_PROJECTION_CLASS.seam]!, mark: "line", label: "rooted by closure — a seam row" },
    { swatch: PALETTE[SPARSE_CM12_PROJECTION_CLASS.delta]!, mark: "arrow", label: "the change this stage made" },
    { swatch: "linear-gradient(90deg,#2650bd,#f2f2f2,#e0473a)", mark: "box", label: "−1e−3 → +1e−3" },
  ],
});
