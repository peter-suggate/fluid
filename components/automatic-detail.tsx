"use client";

import { useState } from "react";
import { useEditorHost } from "../lib/core/session/host-context";
import type { SceneDescription } from "../lib/core/model";
import { DETAIL_DEFAULT_BUDGET } from "../lib/methods/narrow-band/brick-detail-regions";
import { patchCommitter } from "./EntityOptions";
import { RailChoice, RailRange } from "./rail";

/** The automatic detail factors on offer: none, h / 2 and h / 4. */
export const AUTOMATIC_DETAIL_CHOICES = [
  { factor: undefined, label: "Off", hint: "Only the boxes drawn as Detail tiers run the fine engine." },
  { factor: 2, label: "Detail ×2", hint: "The fastest-deforming surface (a splash's crown, a breaking sheet) runs again at half the cell size, within the budget." },
  { factor: 4, label: "Detail ×4", hint: "The fastest-deforming surface runs again at a quarter of the cell size, within the budget: each tile costs eight times ×2's." },
] as const;

/** The budgets on offer (parent 4h tiles a step), powers of two; the default is DETAIL_DEFAULT_BUDGET. */
export const AUTOMATIC_DETAIL_BUDGETS = [64, 128, 256, 512, 1024] as const;

const BUDGET_HINT = "At most this many of the parent's 4h tiles a step run the fine engine, the hottest (fastest-deforming surface) first: more detail, more cost. Restarts the solver.";

/**
 * The scene's automatic detail (`fluid.automaticDetail`, brick-detail-regions.ts) and its one writer: a scene edit that
 * commits (and undoes) as one and re-seeds the solve. The Make toolstrip's menu and the Tune card both write through it.
 */
export function useAutomaticDetail(scene: SceneDescription) {
  const commit = patchCommitter(useEditorHost<SceneDescription, Partial<SceneDescription>>());
  const factor = scene.fluid.automaticDetail?.factor, budget = scene.fluid.automaticDetail?.budget ?? DETAIL_DEFAULT_BUDGET;
  const write = (label: string, next: { readonly factor: 2 | 4; readonly budget: number } | undefined) => {
    const { automaticDetail: _previous, ...fluid } = scene.fluid;
    const value = next ? { factor: next.factor, ...(next.budget !== DETAIL_DEFAULT_BUDGET ? { budget: next.budget } : {}) } : undefined;
    commit(`Automatic detail ${label}`, { fluid: value ? { ...fluid, automaticDetail: value } : fluid }, { reseed: true });
  };
  return {
    factor, budget,
    setFactor(next: 2 | 4 | undefined) {
      if (next === factor) return;
      const label = AUTOMATIC_DETAIL_CHOICES.find(choice => choice.factor === next)?.label ?? "Off";
      write(label, next ? { factor: next, budget } : undefined);
    },
    setBudget(next: number) {
      if (!factor || next === budget) return;
      write(`budget ${next}`, { factor, budget: next });
    },
  };
}

/**
 * The Tune card's fine detail: the factor (Off, ×2, ×4) and, while on, the budget as a slider over its powers of two.
 * Each change rebuilds the solve, so the slider commits when the drag ends, not on every step of it.
 */
export function AutomaticDetailTune({ scene }: { scene: SceneDescription }) {
  const detail = useAutomaticDetail(scene);
  const [dragging, setDragging] = useState<number | undefined>(undefined);
  const exponent = (n: number) => Math.round(Math.log2(n));
  const shown = dragging ?? exponent(detail.budget);
  const lo = exponent(AUTOMATIC_DETAIL_BUDGETS[0]), hi = exponent(AUTOMATIC_DETAIL_BUDGETS[AUTOMATIC_DETAIL_BUDGETS.length - 1]);
  return <>
    <RailChoice label="Fine detail" value={String(detail.factor ?? "off")}
      options={AUTOMATIC_DETAIL_CHOICES.map(choice => ({ value: String(choice.factor ?? "off"), label: choice.factor ? `×${choice.factor}` : "Off", hint: choice.hint }))}
      hint="A fine engine at h / 2 or h / 4 on the fastest-deforming surface (a splash's crown, a breaking sheet), chosen every step within the budget. Restarts the solver."
      testId="scene-detail-fine-factor"
      onChange={value => detail.setFactor(value === "off" ? undefined : Number(value) as 2 | 4)} />
    {detail.factor ? <RailRange label="Fine budget" value={shown} min={lo} max={hi} step={1} display={`${2 ** shown} tiles`}
      hint={BUDGET_HINT} testId="scene-detail-fine-budget"
      onInput={value => setDragging(value)}
      onChange={value => { setDragging(undefined); detail.setBudget(2 ** value); }}
      onGestureEnd={() => setDragging(undefined)}
      modified={detail.budget !== DETAIL_DEFAULT_BUDGET} onReset={() => detail.setBudget(DETAIL_DEFAULT_BUDGET)} /> : null}
  </>;
}
