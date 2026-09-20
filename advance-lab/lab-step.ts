import { numberQuery, queryRecord } from "../lib/framework/persistence";
import { UNIFORM_PAPER_DT_S } from "../lib/methods/uniform/uniform-paper";

/** Retiming changes the next advance, never the scene seed. */
export const STEP_SIZES = [
  { dt: 1 / 15, label: "1/15 s" },
  { dt: UNIFORM_PAPER_DT_S, label: "1/30 s" },
  { dt: 1 / 60, label: "1/60 s" },
  { dt: 1 / 120, label: "1/120 s" },
] as const;
export const labStepQuery = queryRecord({
  dt: numberQuery("dt", UNIFORM_PAPER_DT_S, 1 / 120, 1 / 15),
});
