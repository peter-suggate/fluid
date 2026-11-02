import type { CommandResult } from "../../interaction/contracts";
import type { RunScope } from "../../core/run-scope";

/**
 * What lifting a press on the interaction layer does. A click on a draft
 * places it and a dragged base lands where it was dragged; a drag on a draft
 * was an orbit, and a click on a base only selected it.
 *
 * A cancelled press — a system gesture, a lost capture — does nothing at all:
 * the browser took the pointer away, so the press was abandoned, not
 * completed. Ending it as a release placed the draft, or moved the entity to
 * wherever the pointer last was.
 */
export function pressRelease(press: { readonly kind: "placement" | "move" | "look" | "select"; readonly dragged: boolean },
  cancelled: boolean): "place" | "move" | "nothing" {
  if (cancelled) return "nothing";
  if (press.kind === "placement" && !press.dragged) return "place";
  if (press.kind === "move" && press.dragged) return "move";
  return "nothing";
}

/**
 * Send a row's request as the latest of `requests`, and answer with its
 * refusal only if nothing has superseded it since: a newer request, a change
 * to the record it edits, or the row closing (each of which cancels the scope).
 * A stale refusal used to put its reason over a value accepted after it.
 */
export async function latestRefusal(requests: RunScope, send: () => Promise<CommandResult>): Promise<string | undefined> {
  const request = requests.start();
  const result = await send();
  return !result.accepted && request.current() ? result.reason : undefined;
}
