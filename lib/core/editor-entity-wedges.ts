import type { EditorAction, EditorActionEffect } from "./editor-action";
import type { EditorEntity } from "./editor-entity";

/**
 * The two verbs every editable thing offers, composed once.
 *
 * `entityActionsAt` built these inline, which was correct while the studio was
 * the only host: the ring for a selected thing is "what this thing offers,
 * then Edit and Delete", and repeating that pair in seven entity declarations
 * would guarantee one of them quietly lost its delete. The lab then wrote the
 * same pair a third time, by hand, in `advance-lab/slice-actions.ts` — with its
 * own ids (`slice-region-select`, `slice-region-remove`), its own labels
 * ("Select", "Remove") and its own tones. Two pages disagreeing about what the
 * word for "delete this" is, on the same capability, is exactly the drift this
 * package exists to stop.
 *
 * So they are here, taking the entity and — for the delete — the effect that
 * performs it. The *effect* is the only part that is genuinely the host's: the
 * studio writes a new `SceneDescription`, the lab sends a whole-list command to
 * a running Rust world, and neither can express the other's. Everything a
 * reader can see about the wedge — where it sits in the ring, what it is
 * called, what it is toned, what it promises — is one declaration.
 *
 * `Delete` is last and toned `danger` wherever it appears, so the one
 * irreversible wedge is always in the same place whatever else the ring holds.
 */

/** Raise this thing's own controls, and its handles. */
export function entitySelectWedge<Patch, Doc>(
  entity: Pick<EditorEntity<Patch, Doc>, "selection" | "label" | "tone">,
): EditorAction {
  return {
    id: "select",
    label: "Edit",
    icon: "edit",
    tone: entity.tone,
    hint: `Select ${entity.label} and open its controls`,
    effect: { kind: "select", selection: entity.selection, openControls: true },
  };
}

/**
 * End this thing.
 *
 * The effect is the caller's because performing a delete is the one part of it
 * that is not shared; the wedge around the effect is not. An entity with no
 * `remove` has no delete wedge at all — see `entityActionsAt` — rather than a
 * disabled one, because a thing that cannot be removed is not a thing whose
 * removal a reader should be learning the direction of.
 */
export function entityDeleteWedge<Patch, Doc>(
  entity: Pick<EditorEntity<Patch, Doc>, "label">,
  effect: EditorActionEffect,
): EditorAction {
  return {
    id: "delete",
    label: "Delete",
    icon: "delete",
    tone: "danger",
    hint: `Remove ${entity.label} from the scene`,
    effect,
  };
}
