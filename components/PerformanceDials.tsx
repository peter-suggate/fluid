"use client";

import {
  OCTREE_RUNTIME_DIALS,
  octreeDialledIterationCap,
  octreeDialledRelativeTolerance,
  resolveOctreeRuntimeDials,
  type OctreeRuntimeDialSpec,
} from "@/lib/octree-runtime-dials";
import { simulation } from "@/lib/simulation/controller";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useMethodStore, resolvedMethodValues } from "@/lib/stores/method-store";
import { useSceneStore } from "@/lib/stores/scene-store";

/**
 * The live accuracy-for-frame-time strip.
 *
 * These are the five coarse-band Losasso knobs with the largest measured
 * bearing on the physics lane, ordered by leverage (see
 * `lib/octree-runtime-dials.ts` for the capture that picked them). They sit at
 * the top of the observatory because the point is to move one and watch the
 * lane's measured time respond in the same session: every dial is declared
 * `update: "runtime"`, so a drag reaches the attached solver as a queue write
 * instead of resetting the simulation to t=0.
 *
 * Each carries two captions — what it controls, and what the cheap direction
 * costs — instead of a share-of-frame figure. The figure was true of one
 * capture on one scene and read as a promise; the cost line is true wherever
 * you point the dial, and it is what makes a dialled-down frame recognisable
 * as a setting rather than a bug.
 *
 * Each dial shows the value it will actually take — clamped to the compiled
 * envelope, snapped to the step the GPU can express — rather than the raw
 * slider position, because "6 sweeps" that the ping-pong smoother rounds to 6
 * and "48 iterations" that the compiled envelope clamps to 33 are different
 * claims, and only the effective one explains the measurement.
 */

/** Value the panel shows for a dial that is still at construction's choice. */
function effectiveLabel(
  spec: OctreeRuntimeDialSpec,
  value: number,
  resolved: { tolerance: number; iterationCap: number },
): { text: string; auto: boolean } {
  const auto = spec.auto !== undefined && value === spec.auto;
  switch (spec.key) {
    case "pressureToleranceDecades":
      return { text: resolved.tolerance.toExponential(1), auto };
    case "pressureIterationCap":
      return { text: `${resolved.iterationCap}`, auto };
    default:
      return { text: `${value}`, auto };
  }
}

export function PerformanceDials() {
  const methodState = useMethodStore();
  const methodId = methodState.methodId;
  const overrides = methodState.overrides[methodId] ?? {};
  const scene = useSceneStore((state) => state.scene);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const values = resolvedMethodValues(methodState);
  const dials = resolveOctreeRuntimeDials(values);
  // The strip describes Losasso coarse-dynamics machinery. On any other
  // backend the controls would be inert, and an inert slider reads as a broken
  // one, so the strip states why instead of pretending.
  const losasso = methodId === "octree" && values.coarseBackend !== "power2017";
  // What the solver was BUILT with bounds what a dial can ask for; the panel
  // must show the clamp rather than the request. The envelope is telemetry, so
  // fall back to the paper ceiling until the first stats poll lands.
  const builtEnvelope = gpuInfo?.quadtreePressureIterationBudget ?? 40;
  const resolved = {
    tolerance: octreeDialledRelativeTolerance(
      scene.numerics.pressureRelativeTolerance, dials),
    iterationCap: octreeDialledIterationCap(builtEnvelope, dials),
  };
  const executed = gpuInfo?.quadtreePressureIterationsUsed;
  const anyOverridden = OCTREE_RUNTIME_DIALS.some((spec) => spec.key in overrides);
  return (
    <section className="performance-dials" data-testid="performance-dials" aria-label="Live accuracy and frame-time dials">
      <header>
        <div>
          <span>LOSASSO COARSE BAND · LIVE</span>
          <h3>Accuracy for frame time</h3>
        </div>
        <div className="performance-dials-status">
          {executed !== undefined && <span title="Iterations the residual gate actually executed on the last polled advance">
            {executed} / {resolved.iterationCap} PCG
          </span>}
          {anyOverridden && <button
            type="button"
            onClick={() => {
              for (const spec of OCTREE_RUNTIME_DIALS) {
                if (spec.key in overrides) simulation.resetMethodParam(methodId, spec.key);
              }
            }}
          >RESTORE ALL</button>}
        </div>
      </header>
      {!losasso && <p className="performance-dials-inert">
        These dials drive Losasso coarse-dynamics machinery — the resident MGPCG,
        its first-order V-cycle, the Section 5 axis-face extension, and the
        candidate topology epoch. Select the octree method with the Losasso
        backend to make them live.
      </p>}
      <div className="performance-dial-grid">
        {OCTREE_RUNTIME_DIALS.map((spec) => {
          const value = dials[spec.key];
          const overridden = spec.key in overrides;
          const { text, auto } = effectiveLabel(spec, value, resolved);
          return (
            <label key={spec.key} className="performance-dial" title={spec.hint} data-dial={spec.key} data-modified={overridden}>
              <span className="performance-dial-heading">
                <span>
                  {spec.short}
                  {overridden && <button
                    type="button"
                    className="reset-chip"
                    title="Restore the construction-time value"
                    onClick={(event) => { event.preventDefault(); simulation.resetMethodParam(methodId, spec.key); }}
                  >↺</button>}
                </span>
                <output data-auto={auto}>{auto ? "AUTO" : text}</output>
              </span>
              <input
                type="range"
                disabled={!losasso}
                min={spec.min}
                max={spec.max}
                step={spec.step}
                value={value}
                aria-label={spec.label}
                onChange={(event) =>
                  simulation.setMethodParam(methodId, spec.key, Number(event.currentTarget.value))}
              />
              <small className="performance-dial-does">{spec.does}</small>
              <small className="performance-dial-cost">{spec.cost}</small>
            </label>
          );
        })}
      </div>
    </section>
  );
}
