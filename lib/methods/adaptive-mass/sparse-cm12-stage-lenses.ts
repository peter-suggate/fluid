/**
 * The lens half of the Sparse CM12 stage registry, as the resident consumes it.
 *
 * A lens is filed on its stage in `sparse-cm12-stages.ts`, beside that stage's
 * diagram node and trace phases; this module only reads the roster back out in
 * the two shapes the resident and the renderer want — stage → lens | null, and
 * the flat list of lenses that exist — and owns the typed tap sinks a stage
 * body captures through. Nothing here decides which lens a stage has.
 *
 * Kept as its own module, rather than folded into the registry, because the
 * resident imports it at runtime and the registry is a leaf that must stay
 * free of the resident: the dependency runs resident → here → registry, and
 * back to the resident only through types.
 */
import type { StageTapSink } from "../../core/stage-lens";
import type { AnyStageLens } from "../../core/stage-lens";
import type { SparseCM12StageLensSource } from "./sparse-cm12-stage-lens-source";
import { SPARSE_CM12_STAGES, type SparseCM12StageTapName } from "./sparse-cm12-stages";
import type { SparseCM12ResidentStageId } from "./webgpu-sparse-cm12-resident";

export type { SparseCM12StageTapName } from "./sparse-cm12-stages";

/**
 * Every Sparse CM12 stage, and the lens on it — read from the registry, where
 * "this stage has nothing worth drawing yet" is written down as `null` rather
 * than left as an absence.
 */
export const SPARSE_CM12_STAGE_LENSES: {
  readonly [Stage in SparseCM12ResidentStageId]: (typeof SPARSE_CM12_STAGES)[Stage]["lens"];
} = Object.freeze(Object.fromEntries(
  Object.entries(SPARSE_CM12_STAGES).map(([stage, entry]) => [stage, entry.lens]),
) as { [Stage in SparseCM12ResidentStageId]: (typeof SPARSE_CM12_STAGES)[Stage]["lens"] });

/** The lenses that exist, in stage order. What the renderer is handed. */
export const SPARSE_CM12_LENSES: readonly AnyStageLens[] = Object.freeze(
  Object.values(SPARSE_CM12_STAGES)
    .map((entry): AnyStageLens | null => entry.lens)
    .filter((lens): lens is AnyStageLens => lens !== null));

/**
 * The typed tap sinks for one frame's encoder.
 *
 * Built once per advance beside the encoder's other closures. The encoder's
 * `stage()` helper hands each body `for(itsOwnId)`, so a stage body reads
 * `lens.capture("beforeExecute")` and cannot name a tap its lens does not
 * declare, nor reach another stage's lens at all.
 *
 * `closePass` is threaded through because a copy cannot happen inside an open
 * compute pass. It runs only while a lens is armed: an unarmed frame keeps the
 * pass structure it has always had, so arming a lens is the only thing that
 * changes how the advance is encoded.
 */
export interface SparseCM12StageTaps {
  for<Stage extends SparseCM12ResidentStageId>(
    stage: Stage,
  ): StageTapSink<SparseCM12StageTapName<Stage>>;
}

const INERT: StageTapSink<string> = Object.freeze({ capture: () => {} });

export function sparseCM12StageTaps(
  source: SparseCM12StageLensSource | undefined,
  encoder: GPUCommandEncoder,
  closePass: () => void,
): SparseCM12StageTaps {
  return {
    for<Stage extends SparseCM12ResidentStageId>(stage: Stage) {
      const lens: AnyStageLens | null = SPARSE_CM12_STAGE_LENSES[stage];
      if (!source || !lens || source.armed !== lens.id) {
        return INERT as StageTapSink<SparseCM12StageTapName<Stage>>;
      }
      return {
        capture: (tap: string) => {
          closePass();
          source.capture(encoder, lens.id, tap);
        },
      } as StageTapSink<SparseCM12StageTapName<Stage>>;
    },
  };
}
