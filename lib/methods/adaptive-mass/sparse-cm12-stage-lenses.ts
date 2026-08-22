/**
 * Every Sparse CM12 stage, and the lens on it.
 *
 * Exhaustive over `SparseCM12ResidentStageId` on purpose: adding a stage to the
 * advance and forgetting to decide about its lens is a `tsc` error, and the
 * decision "this stage has nothing worth drawing yet" is written down as
 * `null` rather than left as an absence. Seventeen of the eighteen say that
 * today. The one that does not is the stage the frame turns on.
 */
import type { StageTapSink } from "../../core/stage-lens";
import type { AnyStageLens, StageLens } from "../../core/stage-lens";
import { SPARSE_CM12_FACE_PROJECTION_LENS } from "./sparse-cm12-face-projection.lens";
import type { SparseCM12StageLensSource } from "./sparse-cm12-stage-lens-source";
import type { SparseCM12ResidentStageId } from "./webgpu-sparse-cm12-resident";

export const SPARSE_CM12_STAGE_LENSES = Object.freeze({
  "transport-velocity-extension": null,
  "face-preparation": null,
  "conservative-transport": null,
  "tracer-advection": null,
  "gamma-diffusion": null,
  "surface-sharpening": null,
  "symmetry-authority": null,
  "body-forces": null,
  "pressure-topology": null,
  "pressure-rhs": null,
  "pressure-solve": null,
  "velocity-projection": SPARSE_CM12_FACE_PROJECTION_LENS,
  "projection-diagnostics": null,
  "activity-measurement": null,
  "resolution-planning": null,
  "candidate-transfer": null,
  "brick-retirement": null,
  "presentation-publication": null,
} as const satisfies Readonly<Record<SparseCM12ResidentStageId, AnyStageLens | null>>);

/** The lenses that exist, in stage order. What the renderer is handed. */
export const SPARSE_CM12_LENSES: readonly AnyStageLens[] = Object.freeze(
  Object.values(SPARSE_CM12_STAGE_LENSES).filter((lens) => lens !== null));

/** The tap names a stage's lens declares, or `never` when it has no lens. */
export type SparseCM12StageTapName<Stage extends SparseCM12ResidentStageId> =
  (typeof SPARSE_CM12_STAGE_LENSES)[Stage] extends
    StageLens<infer _Id, infer _Publications, infer Taps, infer _Header, infer _Programs>
    ? keyof Taps & string
    : never;

/**
 * The typed tap sinks for one frame's encoder.
 *
 * Built once per advance beside the encoder's other closures, so a stage body
 * reads `taps.for("velocity-projection").capture("beforeExecute")` and cannot
 * name a tap that stage does not declare, nor a stage that is not its own.
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
