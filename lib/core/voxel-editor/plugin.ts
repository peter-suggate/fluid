import { normalizeControlNumber, type ControlMetadata } from "../../framework/controls";
import type { EditorEntityTone, EditorRay } from "../editor-entity";
import type { EditorHighlight } from "../editor-target";
import type { LiveFluidEdit } from "../live-fluid-edit";
import type { SceneDescription } from "../model";
import type { SolidWorldVoxelPatch } from "../solid-world";

export type ToolValues = Readonly<Record<string, number>>;
export interface ToolControl extends ControlMetadata {
  readonly id: string;
  readonly kind?: "number" | "toggle";
  /** Contextual hosts keep primary controls visible and disclose advanced ones. */
  readonly presentation?: "primary" | "advanced";
  readonly label: string;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  readonly initial: number;
}
export interface ToolContext {
  readonly scene: SceneDescription;
  readonly ray: EditorRay;
  readonly values: ToolValues;
  /** The reader held the "do the opposite" modifier at the press: a build carves, a carve builds. */
  readonly invert?: boolean;
}
export type ToolAction = { readonly kind: "fluid"; readonly edit: LiveFluidEdit };
export interface ToolUpdate {
  /** A transient action is previewed during the gesture and executed once on release. */
  readonly action?: ToolAction;
  /** Complete proposal for this stroke, relative to its immutable base. */
  readonly patches: readonly SolidWorldVoxelPatch[];
  readonly highlight: EditorHighlight;
  readonly tone?: EditorEntityTone;
  readonly caption: string;
}
export interface ToolGesture {
  update(ray: EditorRay): ToolUpdate | undefined;
  /**
   * Asked once, when the pointer is released. True means the gesture has a
   * further phase that follows the bare pointer (push/pull's extrusion after
   * its footprint); the host keeps the stroke open and the next press commits.
   */
  advance?(): boolean;
}
/**
 * A capability owner's declaration and behavior, composed in a static catalog
 * like resource plugins and framework features. The voxel shelf is this domain's
 * UI slot; group/order declare insertion, icon/controls declare representation.
 * Hosts own interaction and publication lifecycle, and never dispatch on tool id.
 */
export interface VoxelToolPlugin {
  readonly id: string;
  readonly version: 1;
  /**
   * What release does. `"author"` (the default) appends the previewed patches
   * to the scene document once; `"release"` executes the previewed transient
   * action and authors nothing. Neither touches the document during the drag.
   */
  readonly execution?: "author" | "release";
  readonly ui: {
    readonly label: string;
    readonly hint: string;
    /** A consequence users should see while this tool is armed. */
    readonly notice?: string;
    readonly group: string;
    readonly order: number;
    /** SVG path, so adding an icon does not require a second registry. */
    readonly icon: string;
    readonly controls: readonly ToolControl[];
  };
  /** Context-aware initial values; explicit user values always take precedence. */
  defaults?(scene: SceneDescription, values: ToolValues): ToolValues;
  unavailable(context: { scene: SceneDescription; methodId: string }): string | undefined;
  begin(context: ToolContext): ToolGesture | undefined;
}
export function toolValues(plugin: VoxelToolPlugin, values: ToolValues = {}, scene?: SceneDescription): ToolValues {
  const normalize = (fallback: ToolValues) => Object.fromEntries(plugin.ui.controls.map((control) => {
    const initial = normalizeControlNumber(fallback[control.id], control.initial, control, true);
    return [control.id, normalizeControlNumber(values[control.id], initial, control, true)];
  }));
  const initial = normalize({});
  return scene && plugin.defaults ? normalize(plugin.defaults(scene, initial)) : initial;
}
export function createVoxelToolRegistry(plugins: readonly VoxelToolPlugin[]) {
  const ids = new Set<string>();
  for (const plugin of plugins) {
    if (!plugin.id || ids.has(plugin.id)) throw new Error(`Duplicate or empty voxel tool id: ${plugin.id}`);
    ids.add(plugin.id);
    if ((plugin.execution !== undefined && plugin.execution !== "author" && plugin.execution !== "release")
      || plugin.version !== 1 || !plugin.ui.label.trim() || !plugin.ui.group.trim()
      || !plugin.ui.icon.trim() || !Number.isFinite(plugin.ui.order)
      || (plugin.defaults !== undefined && typeof plugin.defaults !== "function")
      || typeof plugin.begin !== "function" || typeof plugin.unavailable !== "function") {
      throw new Error(`Invalid voxel tool definition: ${plugin.id}`);
    }
    const controls = new Set<string>();
    for (const control of plugin.ui.controls) {
      if (!control.id.trim() || controls.has(control.id) || !control.label.trim()
        || ![control.min, control.max, control.step, control.initial].every(Number.isFinite)
        || !(control.step > 0) || control.min > control.max
        || control.initial < control.min || control.initial > control.max
        || (control.kind !== undefined && control.kind !== "number" && control.kind !== "toggle")
        || (control.presentation !== undefined && control.presentation !== "primary" && control.presentation !== "advanced")
        || (control.kind === "toggle" && (control.min !== 0 || control.max !== 1 || control.step !== 1
          || (control.initial !== 0 && control.initial !== 1)))) {
        throw new Error(`Invalid control ${plugin.id}.${control.id}`);
      }
      controls.add(control.id);
    }
  }
  const ordered = Object.freeze([...plugins].sort((a, b) => a.ui.order - b.ui.order));
  return Object.freeze({ tools: ordered, get: (id: string | undefined) => ordered.find((tool) => tool.id === id) });
}
