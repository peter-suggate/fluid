"use client";

import { useState } from "react";
import { useEditorHost } from "../lib/core/session/host-context";
import type { SceneDescription } from "../lib/core/model";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice-dimensions";
import { patchCommitter } from "./EntityOptions";
import { RailRange } from "./rail";

/** Water's surface tension at 20 °C, N/m. */
export const WATER_SURFACE_TENSION_N_M = 0.072;

/** The slider's ladder (N/m): off, then water's value times powers of two from 1/4 to 32. */
export const SURFACE_TENSION_STEPS = [0, ...[-2, -1, 0, 1, 2, 3, 4, 5].map(k => WATER_SURFACE_TENSION_N_M * 2 ** k)] as const;

/**
 * The explicit capillary step limit (Brackbill et al. 1992; Ando & Batty 2020 §8.3): the shortest capillary wave a
 * grid of cell size h holds is stable while dt <= sqrt(ρ h³ / (2π σ)). Infinity without tension.
 */
export function capillaryStepLimit_s(sigma_N_m: number, density_kg_m3: number, h_m: number): number {
  return sigma_N_m > 0 ? Math.sqrt(density_kg_m3 * h_m ** 3 / (2 * Math.PI * sigma_N_m)) : Infinity;
}

/** The scene's finest cell (the fine detail's, when on), m. */
export function sceneFinestCell_m(scene: SceneDescription): number {
  const cells = sceneLatticeDimensions(scene);
  return scene.container.width_m / cells[0]! / (scene.fluid.automaticDetail?.factor ?? 1);
}

const nearest = (sigma: number) => SURFACE_TENSION_STEPS.reduce((best, s, i) => Math.abs(s - sigma) < Math.abs(SURFACE_TENSION_STEPS[best]! - sigma) ? i : best, 0);
const format = (sigma: number) => sigma === 0 ? "Off" : `${sigma < 0.1 ? sigma.toFixed(3) : sigma < 1 ? sigma.toFixed(2) : sigma.toFixed(1)} N/m`;

/** One re-seeding scene edit of `fluid.surfaceTension_N_m` (the engines compile σ into their kernels). */
export function useSurfaceTension(scene: SceneDescription) {
  const commit = patchCommitter(useEditorHost<SceneDescription, Partial<SceneDescription>>());
  const sigma = scene.fluid.surfaceTension_N_m;
  return {
    sigma,
    setSigma(next: number) {
      if (!(next >= 0) || next === sigma) return;
      commit(`Surface tension ${format(next)}`, { fluid: { ...scene.fluid, surfaceTension_N_m: next } }, { reseed: true });
    },
  };
}

/**
 * The Tune card's surface tension: a slider over SURFACE_TENSION_STEPS, committing when the drag ends (each change
 * rebuilds the solve). The readout flags a value past the explicit capillary step limit at the finest cell, where the
 * shortest capillary waves grow instead of oscillating; the value is the user's to choose and is not clamped.
 */
export function SurfaceTensionTune({ scene }: { scene: SceneDescription }) {
  const tension = useSurfaceTension(scene);
  const [dragging, setDragging] = useState<number | undefined>(undefined);
  const index = dragging ?? nearest(tension.sigma);
  const shown = dragging === undefined ? tension.sigma : SURFACE_TENSION_STEPS[index]!;
  const h = sceneFinestCell_m(scene), dt = scene.numerics.fixedDt_s;
  const limit = capillaryStepLimit_s(shown, scene.fluid.density_kg_m3, h);
  const over = dt > limit;
  return <RailRange label="Surface tension" value={index} min={0} max={SURFACE_TENSION_STEPS.length - 1} step={1}
    display={`${format(shown)}${over ? " · unstable" : ""}`}
    hint={`Capillary force from the surface's curvature (water ${WATER_SURFACE_TENSION_N_M} N/m): rounds rims and pulls thin sheets together. Explicit, so stable while the step is under √(ρh³/2πσ) at the finest cell (${(h * 100).toPrecision(3)} cm): ${Number.isFinite(limit) ? `${(limit * 1000).toPrecision(3)} ms` : "no limit"} against the scene's ${(dt * 1000).toPrecision(3)} ms. Restarts the solver.`}
    testId="scene-surface-tension"
    onInput={value => setDragging(value)}
    onChange={value => { setDragging(undefined); tension.setSigma(SURFACE_TENSION_STEPS[value]!); }}
    onGestureEnd={() => setDragging(undefined)} />;
}
