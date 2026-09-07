import type { EditorRay } from "../editor-entity";
import type { EditorHighlight } from "../editor-target";
import type { SceneDescription } from "../model";
import type { SolidWorldVoxelPatch } from "../solid-world";

export type ToolValues = Readonly<Record<string, number>>;
export interface ToolControl {
  readonly id: string;
  readonly kind?: "number" | "toggle";
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
}
export interface ToolUpdate {
  /** Complete proposal for this stroke, relative to its immutable base. */
  readonly patches: readonly SolidWorldVoxelPatch[];
  readonly highlight: EditorHighlight;
  readonly caption: string;
}
export interface ToolGesture {
  update(ray: EditorRay): ToolUpdate | undefined;
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
  readonly ui: {
    readonly label: string;
    readonly hint: string;
    readonly group: string;
    readonly order: number;
    /** SVG path, so adding an icon does not require a second registry. */
    readonly icon: string;
    readonly controls: readonly ToolControl[];
  };
  unavailable(context: { scene: SceneDescription; methodId: string }): string | undefined;
  begin(context: ToolContext): ToolGesture | undefined;
}
export function toolValues(plugin: VoxelToolPlugin, values: ToolValues = {}): ToolValues {
  return Object.fromEntries(plugin.ui.controls.map((control) => {
    const candidate = values[control.id];
    const value = candidate !== undefined && Number.isFinite(candidate) ? candidate : control.initial;
    const clamped = Math.max(control.min, Math.min(control.max, value));
    // Match native number-input stepping, which is relative to its minimum.
    const snapped = control.min + Math.round((clamped - control.min) / control.step) * control.step;
    return [control.id, Math.max(control.min, Math.min(control.max, snapped))];
  }));
}
export function createVoxelToolRegistry(plugins: readonly VoxelToolPlugin[]) {
  const ids = new Set<string>();
  for (const plugin of plugins) {
    if (!plugin.id || ids.has(plugin.id)) throw new Error(`Duplicate or empty voxel tool id: ${plugin.id}`);
    ids.add(plugin.id);
    if (plugin.version !== 1 || !plugin.ui.label.trim() || !plugin.ui.group.trim()
      || !plugin.ui.icon.trim() || !Number.isFinite(plugin.ui.order)
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
