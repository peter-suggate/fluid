/**
 * The advance as four readings, derived rather than restated.
 *
 * The loop the whole method is — represent, solve the motion, transport, adapt
 * and publish — is not the same partition as the diagram's bands: geometric
 * transport is in the transport band beside velocity extension and face
 * preparation, but it is its own reading of the loop because it runs *after*
 * the projection. So the reading a stage belongs to is declared per stage, in
 * `ADVANCE_STAGE_DECLARATIONS[stage].slice.loopStep`, and the ranges the strip is laid
 * out from are worked out here from the encode order.
 *
 * What this replaces is a hand-written table of numeric stage indices. A stage
 * inserted into the encoder shifted every index after it, silently, and the
 * strip went on drawing a step over the wrong seven stages until somebody
 * noticed the caption did not match the picture.
 */
import { ADVANCE_STAGE_ORDER, type AdvanceStageId } from "./advance-work";
import {
  ADVANCE_LOOP_STEP_NAMES,
  type AdvanceLoopStepNumber, type AdvanceSliceDeclaration,
} from "./definition";
import { ADVANCE_STAGE_DECLARATIONS } from "./advance-stages";

/** The slice declaration displayed for one stage. */
export function advanceStageSlice(stage: AdvanceStageId): AdvanceSliceDeclaration {
  const declared = ADVANCE_STAGE_DECLARATIONS[stage].slice;
  if (!declared) {
    throw new Error(`${stage} is encoded but declares no advance slice`);
  }
  return declared;
}

/** One reading of the loop, over the range of stages that encode it. */
export interface AdvanceLoopStep {
  readonly n: AdvanceLoopStepNumber;
  readonly name: string;
  /**
   * The stages this step covers, 1-based and inclusive into
   * `ADVANCE_STAGE_ORDER`. Both are 0 for step 1, which encodes nothing: it is
   * the state the advance starts from.
   */
  readonly from: number;
  readonly to: number;
}

function deriveLoopSteps(): readonly AdvanceLoopStep[] {
  const spans = new Map<AdvanceLoopStepNumber, { from: number; to: number }>();
  ADVANCE_STAGE_ORDER.forEach((stage, index) => {
    const step = advanceStageSlice(stage).loopStep;
    const span = spans.get(step);
    if (!span) spans.set(step, { from: index + 1, to: index + 1 });
    else span.to = index + 1;
  });
  const steps: AdvanceLoopStep[] = [
    { n: 1, name: ADVANCE_LOOP_STEP_NAMES[1], from: 0, to: 0 },
  ];
  for (const n of [2, 3, 4] as const) {
    const span = spans.get(n);
    if (!span) continue;
    steps.push({ n, name: ADVANCE_LOOP_STEP_NAMES[n], from: span.from, to: span.to });
  }
  return Object.freeze(steps);
}

/** The four readings and the stage ranges they cover. */
export const ADVANCE_LOOP_STEPS: readonly AdvanceLoopStep[] = deriveLoopSteps();

/** The reading a stage falls under, for a caption or a highlight. */
export function advanceStageLoopStep(stage: AdvanceStageId): AdvanceLoopStep {
  const n = advanceStageSlice(stage).loopStep;
  const step = ADVANCE_LOOP_STEPS.find(entry => entry.n === n);
  if (!step) throw new Error(`${stage} declares loop step ${n}, which is not derived`);
  return step;
}
