import type { EditorAction, EditorActionEffect, EditorActionIcon, EditorActionTone } from "./editor-action";
import type { SceneDescription } from "./model";

/**
 * The scene document's own verbs, declared once.
 *
 * Both surfaces that offer them — the ring's Scene wedge and the container
 * strip's document rows — compose from this list, so a verb added here appears
 * on both and the two can never disagree. The declaration carries its own
 * prominence: `priority` says whether a surface should stand the verb up as a
 * row of its own or fold it behind the document's one chevron. Saving,
 * exporting and starting fresh are reached rarely; the strip owes them a route,
 * not a rank.
 *
 * `dryOnly` marks the one verb that is a state transition rather than a file
 * operation: adding water to a document that runs none. On a wet scene it is
 * no verb at all — not a disabled one — so it is omitted, the same judgement
 * the instrument wedge makes about a dry scene's pipeline.
 */
export interface SceneDocumentVerb {
  readonly id: string;
  readonly label: string;
  readonly hint: string;
  readonly icon: EditorActionIcon;
  readonly tone: EditorActionTone;
  readonly effect: EditorActionEffect;
  readonly priority: "high" | "low";
  readonly dryOnly?: boolean;
}

export const SCENE_DOCUMENT_VERBS: readonly SceneDocumentVerb[] = [
  {
    id: "choose-scene",
    label: "Open…",
    icon: "scene",
    tone: "prop",
    hint: "Choose the scene this pane runs, without leaving the studio",
    effect: { kind: "choose-scene" },
    priority: "low",
  },
  {
    id: "scene-new",
    label: "New",
    icon: "scene-new",
    tone: "prop",
    hint: "Start a fresh document. Water is added later, deliberately",
    effect: { kind: "scene-document", op: "new" },
    priority: "low",
  },
  {
    id: "scene-save",
    label: "Save",
    icon: "scene-save",
    tone: "prop",
    hint: "Save to this browser's library under the document's own name",
    effect: { kind: "scene-document", op: "save" },
    priority: "low",
  },
  {
    id: "scene-export",
    label: "Export",
    icon: "scene-export",
    tone: "prop",
    hint: "Download the document as scene JSON",
    effect: { kind: "scene-document", op: "export" },
    priority: "low",
  },
  {
    id: "scene-import",
    label: "Import…",
    icon: "scene-import",
    tone: "prop",
    hint: "Open a scene JSON file from this machine",
    effect: { kind: "scene-document", op: "import" },
    priority: "low",
  },
  {
    id: "scene-enable-water",
    label: "Add water",
    icon: "water-ball",
    tone: "fluid",
    hint: "Hand the document to the fluid solver, starting from its authored setup",
    effect: { kind: "scene-document", op: "enable-water" },
    priority: "high",
    dryOnly: true,
  },
];

/** The verbs that apply to this document, in declaration order. */
export function sceneDocumentVerbs(scene: SceneDescription): readonly SceneDocumentVerb[] {
  const dry = scene.systems?.fluid === false;
  return SCENE_DOCUMENT_VERBS.filter((verb) => !verb.dryOnly || dry);
}

/** The same verbs as ring wedges, for the Scene wedge's child ring. */
export function sceneDocumentActions(scene: SceneDescription): readonly EditorAction[] {
  return sceneDocumentVerbs(scene).map((verb) => ({
    id: verb.id,
    label: verb.label,
    icon: verb.icon,
    tone: verb.tone,
    hint: verb.hint,
    effect: verb.effect,
  }));
}
