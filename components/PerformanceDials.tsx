"use client";

import type { RuntimeDialReadouts, RuntimeDialSpec } from "../lib/core/method-contract";
import { getMethod } from "@/lib/core/method-registry";
import { simulation } from "../lib/core/simulation/controller";
import { useDiagnosticsStore } from "../lib/core/stores/diagnostics-store";
import { useMethodStore, resolvedMethodValues } from "../lib/core/stores/method-store";
import { useSceneStore } from "../lib/core/stores/scene-store";

/**
 * The live accuracy-for-frame-time strip.
 *
 * Which dials exist, in what order, and what each slider position actually
 * resolves to are all the running method's answers — the panel renders the
 * declaration and knows nothing about any individual knob. It used to import
 * one method's dial table and three of its V-cycle derivations directly, which
 * made the observatory's most prominent strip a consumer of Losasso coarse-band
 * internals and gave every other method nine greyed-out sliders for machinery
 * it does not run. The capture that picked and ordered the octree lane's dials
 * is in `lib/methods/octree-shared/octree-runtime-dials.ts`, beside them.
 *
 * The strip sits at the top of the observatory because the point is to move one
 * and watch the lane's measured time respond in the same session: every dial is
 * declared `update: "runtime"`, so a drag reaches the attached solver as a queue
 * write instead of resetting the simulation to t=0.
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
  spec: RuntimeDialSpec,
  value: number,
  readouts: RuntimeDialReadouts,
): { text: string; auto: boolean } {
  return {
    text: readouts.effective[spec.key] ?? `${value}`,
    auto: spec.auto !== undefined && value === spec.auto,
  };
}

export function PerformanceDials() {
  const methodState = useMethodStore();
  const methodId = methodState.methodId;
  const overrides = methodState.overrides[methodId] ?? {};
  const scene = useSceneStore((state) => state.scene);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const values = resolvedMethodValues(methodState);
  const dials = getMethod(methodId).dials;
  // A method with no live knobs gets no strip. The alternative is another
  // method's sliders, disabled, which reads as a broken control rather than an
  // absent one — and it was only ever the octree lane's strip anyway.
  if (!dials) return null;
  const readouts = dials.readouts(values, scene, gpuInfo ?? undefined);
  const anyOverridden = dials.specs.some((spec) => spec.key in overrides);
  return (
    <section className="performance-dials" data-testid="performance-dials" aria-label="Live accuracy and frame-time dials">
      <header>
        <div>
          <span>{dials.label}</span>
          <h3>Accuracy for frame time</h3>
        </div>
        <div className="performance-dials-status">
          {readouts.executed && <span title={readouts.executed.hint}>
            {readouts.executed.text}
          </span>}
          {anyOverridden && <button
            type="button"
            onClick={() => {
              for (const spec of dials.specs) {
                if (spec.key in overrides) simulation.resetMethodParam(methodId, spec.key);
              }
            }}
          >RESTORE ALL</button>}
        </div>
      </header>
      <div className="performance-dial-grid">
        {dials.specs.map((spec) => {
          const value = readouts.positions[spec.key] ?? spec.default;
          const overridden = spec.key in overrides;
          const { text, auto } = effectiveLabel(spec, value, readouts);
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
