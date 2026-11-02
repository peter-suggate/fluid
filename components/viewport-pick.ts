import type { Run } from "../lib/core/run-scope";

/**
 * What a GPU pick's answer does when it lands.
 *
 * The readback belongs to the press that started it. If that press was
 * superseded by a newer one, cancelled, or already became something else, the
 * answer is stale and does nothing — it used to be taken by whichever press
 * held the same pointer id by then. Otherwise a body under a held pointer opens
 * a throw and under a released one is a click that selects it (a released
 * pointer cannot be dragged); with no body a released pointer was a click on
 * whatever the analytic pick found, and a held one becomes the orbit that
 * decides the same thing when it comes up.
 */
export type PickLanding = "stale" | "throw" | "select-body" | "select" | "orbit";

export function pickLanding(press: { readonly action: string; readonly pick?: Run; readonly released?: boolean } | null,
  pick: Run, body: boolean): PickLanding {
  if (press?.action !== "pick" || press.pick !== pick) return "stale";
  if (body) return press.released ? "select-body" : "throw";
  return press.released ? "select" : "orbit";
}
