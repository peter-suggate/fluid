import { normalizeControlNumber, type ControlMetadata } from "../../framework/controls";
import type { EditorEntityTone, EditorRay } from "../editor-entity";
import type { EditorHighlight } from "../editor-target";
import type { LiveFluidEdit } from "../live-fluid-edit";
import type { SceneDescription } from "../model";
import type { SolidWorldVoxelPatch } from "../solid-world";
import type { VoxelClipboard } from "./verb";
import type { VoxelReader, VoxelRegion } from "./voxel-access";
import { voxelReaderForScene } from "./voxel-reader";

export type ToolValues = Readonly<Record<string, number>>;
/**
 * Every stored control value, by owner: a tool's id, a verb's owner id, and
 * `SHARED_TOOL_VALUES_ID` for the controls the whole editor shares.
 */
export type StoredToolValues = Readonly<Record<string, ToolValues>>;
/** The owner under which shared controls are stored. No tool may take it as its id. */
export const SHARED_TOOL_VALUES_ID = "shared";
interface ToolControlBase {
  readonly id: string;
  readonly label: string;
  /** One or two characters for a host with no room for the label: "W", "D". */
  readonly short?: string;
  /**
   * Where the value lives. `"tool"` (the default) keeps one per tool, so a
   * sphere's width is not a brush's. `"shared"` keeps one for the editor: every
   * tool and verb that lists the control reads and writes the same value, which
   * is what a material or a mirror is — a property of the hand, not of the tool
   * it happens to be holding.
   */
  readonly scope?: "tool" | "shared";
  /**
   * Where a host with a rail draws it. `"line"` (the default) is the armed
   * tool's own line. `"rail"` is the rail itself, once, whichever tool is
   * armed: a control every brush shares and the hand changes between strokes
   * — the operation — should not move when the tool does.
   */
  readonly surface?: "line" | "rail";
  /**
   * Drawn only while this holds for the tool's resolved values. The value is
   * kept either way: a material is not forgotten because erasing has no use for it.
   */
  readonly when?: (values: ToolValues) => boolean;
}
export interface ToolNumberControl extends ToolControlBase, ControlMetadata {
  readonly kind?: "number" | "toggle";
  /**
   * The number the brackets and the modified wheel step while this tool is
   * armed: `"primary"` on the bare pair, `"secondary"` with Shift. A tool
   * declaring only one of them answers both, so the keys are never dead.
   */
  readonly adjust?: "primary" | "secondary";
  /** Single unmodified key that flips this toggle while the tool is armed. */
  readonly shortcut?: string;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  readonly initial: number;
}
export interface ToolChoiceOption {
  /** What is stored. Values stay numbers so one record holds every control. */
  readonly value: number;
  readonly label: string;
  /** One sentence saying what choosing it means, for a host with a tip to put it in. */
  readonly hint?: string;
  /** A 24-box SVG path drawn instead of the label. */
  readonly icon?: string;
  /** A CSS colour drawn instead of the label: the option *is* a colour. */
  readonly swatch?: string;
  /** The palette token the option is lit in, and the ghost of a stroke made with it is drawn in. */
  readonly tone?: EditorEntityTone;
  /** Single unmodified key that chooses this option while the tool is armed. */
  readonly shortcut?: string;
}
/**
 * One of a closed set. The set may be empty — a palette nobody has filled —
 * in which case the control resolves to `initial` and a host draws nothing.
 */
export interface ToolChoiceControl extends ToolControlBase {
  readonly kind: "choice";
  readonly options: readonly ToolChoiceOption[];
  readonly initial: number;
}
export type ToolControl = ToolNumberControl | ToolChoiceControl;
export interface ToolContext {
  readonly scene: SceneDescription;
  readonly ray: EditorRay;
  /**
   * Every control the tool lists, resolved — shared ones included. What a
   * gesture needs beyond its own numbers is read from here by the accessors in
   * `./controls` (`toolMaterialId`, `toolPatchOperation`, `toolMirrorAxes`) and
   * from the scene by `toolReader`, so a context is three plain facts and a
   * scripted caller can build one without a host.
   */
  readonly values: ToolValues;
  /** The reader held the "do the opposite" modifier at the press: attaching erases, erasing attaches. */
  readonly invert?: boolean;
  /** The voxels last copied off a selection, for the tool that stamps them. */
  readonly clipboard?: VoxelClipboard;
  /**
   * The host is drawing the ghost under a bare pointer, and nothing this
   * gesture proposes will be committed. A tool whose answer is a search — the
   * connected piece under the pointer — may show where it would start instead
   * of running it on every move.
   */
  readonly hover?: boolean;
}
export type ToolAction = { readonly kind: "fluid"; readonly edit: LiveFluidEdit };
export interface ToolUpdate {
  /** A transient action is previewed during the gesture and executed once on release. */
  readonly action?: ToolAction;
  /** Complete proposal for this stroke, relative to its immutable base. */
  readonly patches: readonly SolidWorldVoxelPatch[];
  /**
   * The cells this gesture selects. Like the patches it is a proposal until
   * release, when the host makes it the standing voxel selection — after the
   * patches are accepted, when the gesture proposes both. A selection is not
   * document data: it authors nothing and records no history.
   */
  readonly selection?: VoxelRegion;
  /** What `selection` does to the one already standing. Absent replaces it. */
  readonly selectionMode?: "replace" | "add" | "subtract";
  readonly highlight: EditorHighlight;
  readonly tone?: EditorEntityTone;
  readonly caption: string;
}
export interface ToolGesture {
  /**
   * False when the press met nothing and the gesture stands on the ground plane
   * instead. An interactive host hands such a press to the camera — the room
   * orbits, the scene edits — while a scripted one may still build from it.
   */
  readonly surface?: boolean;
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
 * like resource plugins and framework features. The tool rail is this domain's
 * UI slot; group/order declare insertion, icon/controls declare representation
 * and the keys that drive them.
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
    /**
     * What the "do the opposite" modifier does to this tool, in a few words
     * for a host's legend. Absent where the tool gives it nothing to do.
     */
    readonly opposite?: string;
    readonly group: string;
    readonly order: number;
    /** SVG path, so adding an icon does not require a second registry. */
    readonly icon: string;
    readonly controls: readonly ToolControl[];
  };
  /** Context-aware initial values; explicit user values always take precedence. */
  defaults?(scene: SceneDescription, values: ToolValues): ToolValues;
  unavailable(context: ToolAvailabilityContext): string | undefined;
  begin(context: ToolContext): ToolGesture | undefined;
}
/** What a tool is asked about before it is offered: the document, the solver, and what the hand holds. */
export interface ToolAvailabilityContext {
  readonly scene: SceneDescription;
  readonly methodId: string;
  readonly clipboard?: VoxelClipboard;
}
/**
 * Resolve a list of controls against flat values: each one clamped to its own
 * range or snapped to its own options, and `initial` wherever a value is
 * missing or no longer valid. `fallback` stands between the two — a plugin's
 * context-aware defaults.
 */
export function controlValues(controls: readonly ToolControl[], values: ToolValues = {},
  fallback: ToolValues = {}): ToolValues {
  const resolve = (control: ToolControl, value: number | undefined, otherwise: number): number => {
    if (control.kind !== "choice") return normalizeControlNumber(value, otherwise, control, true);
    return control.options.some((option) => option.value === value) ? value! : otherwise;
  };
  return Object.fromEntries(controls.map((control) =>
    [control.id, resolve(control, values[control.id], resolve(control, fallback[control.id], control.initial))]));
}
/** The single resolver of a tool's values. `values` is flat: see `storedToolValues` for a host's store. */
export function toolValues(plugin: VoxelToolPlugin, values: ToolValues = {}, scene?: SceneDescription): ToolValues {
  const initial = controlValues(plugin.ui.controls, values);
  return scene && plugin.defaults
    ? controlValues(plugin.ui.controls, values, plugin.defaults(scene, initial)) : initial;
}
/** The owner a control's value is stored under: the editor for a shared control, otherwise `ownerId`. */
export const controlValueOwner = (ownerId: string, control: ToolControl): string =>
  control.scope === "shared" ? SHARED_TOOL_VALUES_ID : ownerId;
/**
 * What a host has stored for these controls, flattened for the resolver: the
 * owner's own values, and the editor's for every shared control listed. The
 * store is never read any other way, so "shared" means the same everywhere.
 */
export function storedControlValues(ownerId: string, controls: readonly ToolControl[],
  stored: StoredToolValues): ToolValues {
  const values: Record<string, number> = {};
  for (const control of controls) {
    const value = stored[controlValueOwner(ownerId, control)]?.[control.id];
    if (value !== undefined) values[control.id] = value;
  }
  return values;
}
export const storedToolValues = (plugin: VoxelToolPlugin, stored: StoredToolValues): ToolValues =>
  storedControlValues(plugin.id, plugin.ui.controls, stored);
/**
 * The solids of the scene a gesture began on.
 *
 * Asked for, never handed in: building a reader is work a tool that only
 * projects a ray should not pay, and the reader module memoizes per scene, so
 * asking twice — or from a hover ghost and then the press — costs nothing.
 */
export const toolReader = (context: Pick<ToolContext, "scene">): VoxelReader => voxelReaderForScene(context.scene);

/**
 * Reject a malformed control list. Shared by the tool and the verb registries:
 * both hang the same controls on the same rail line. `shared` collects the
 * shared controls already seen across the registry.
 */
export function validateControls(owner: string, list: readonly ToolControl[],
  shared: Map<string, ToolControl>): void {
  const controls = new Set<string>();
  const keys = new Set<string>();
  const adjusts = new Set<string>();
  for (const control of list) {
    const invalid = () => new Error(`Invalid control ${owner}.${control.id}`);
    // A key or a bracket pair names one control; a second claimant would be silently unreachable.
    const claim = (key: string | undefined) => {
      if (key === undefined) return;
      if (!/^[a-z]$/.test(key) || keys.has(key)) throw invalid();
      keys.add(key);
    };
    if (!control.id.trim() || controls.has(control.id) || !control.label.trim()
      || (control.short !== undefined && !control.short.trim())
      || (control.scope !== undefined && control.scope !== "tool" && control.scope !== "shared")
      || (control.surface !== undefined && control.surface !== "line" && control.surface !== "rail")
      // The rail stands for every tool at once, so what it draws cannot be one
      // tool's own value; and it draws a switch, which is a choice.
      || (control.surface === "rail" && (control.scope !== "shared" || control.kind !== "choice"))
      || (control.when !== undefined && typeof control.when !== "function")
      || !Number.isFinite(control.initial)) {
      throw invalid();
    }
    // One stored value cannot be read under two declarations: a shared id is one object.
    if (control.scope === "shared") {
      const known = shared.get(control.id);
      if (known && known !== control) throw new Error(`Shared control ${control.id} is declared twice (${owner}).`);
      shared.set(control.id, control);
    }
    if (control.kind === "choice") {
      const values = control.options.map((option) => option.value);
      if (!Array.isArray(control.options) || new Set(values).size !== values.length || !values.every(Number.isFinite)
        || control.options.some((option) => !option.label.trim())
        || (values.length > 0 && !values.includes(control.initial))) {
        throw invalid();
      }
      for (const option of control.options) claim(option.shortcut);
    } else {
      if ((control.shortcut !== undefined && control.kind !== "toggle")
        || (control.adjust !== undefined && (control.kind === "toggle" || adjusts.has(control.adjust)
          || (control.adjust !== "primary" && control.adjust !== "secondary")))
        || ![control.min, control.max, control.step].every(Number.isFinite)
        || !(control.step > 0) || control.min > control.max
        || control.initial < control.min || control.initial > control.max
        || (control.kind !== undefined && control.kind !== "number" && control.kind !== "toggle")
        || (control.kind === "toggle" && (control.min !== 0 || control.max !== 1 || control.step !== 1
          || (control.initial !== 0 && control.initial !== 1)))) {
        throw invalid();
      }
      claim(control.shortcut);
      if (control.adjust) adjusts.add(control.adjust);
    }
    controls.add(control.id);
  }
}
export function createVoxelToolRegistry(plugins: readonly VoxelToolPlugin[]) {
  const ids = new Set<string>();
  const shared = new Map<string, ToolControl>();
  for (const plugin of plugins) {
    // A tool's id is also the owner its values are stored under, beside the editor's and the verbs'.
    if (!plugin.id || ids.has(plugin.id) || plugin.id === SHARED_TOOL_VALUES_ID || plugin.id.includes(":")) {
      throw new Error(`Duplicate, empty or reserved voxel tool id: ${plugin.id}`);
    }
    ids.add(plugin.id);
    if ((plugin.execution !== undefined && plugin.execution !== "author" && plugin.execution !== "release")
      || plugin.version !== 1 || !plugin.ui.label.trim() || !plugin.ui.group.trim()
      || !plugin.ui.icon.trim() || !Number.isFinite(plugin.ui.order)
      || (plugin.defaults !== undefined && typeof plugin.defaults !== "function")
      || typeof plugin.begin !== "function" || typeof plugin.unavailable !== "function") {
      throw new Error(`Invalid voxel tool definition: ${plugin.id}`);
    }
    validateControls(plugin.id, plugin.ui.controls, shared);
  }
  const ordered = Object.freeze([...plugins].sort((a, b) => a.ui.order - b.ui.order));
  return Object.freeze({ tools: ordered, get: (id: string | undefined) => ordered.find((tool) => tool.id === id) });
}

/** The control the brackets step: the bare pair's, or Shift's — whichever the tool declares when it has only one. */
export function adjustableControl(plugin: VoxelToolPlugin, secondary: boolean): ToolNumberControl | undefined {
  const find = (adjust: ToolNumberControl["adjust"]) => plugin.ui.controls.find((control): control is ToolNumberControl =>
    control.kind !== "choice" && control.adjust === adjust);
  return secondary ? find("secondary") ?? find("primary") : find("primary") ?? find("secondary");
}
/**
 * The control a single key drives while its owner is up, and the value the key
 * gives it: a toggle flips, a choice takes the option that declared the key.
 */
export function controlForKey(controls: readonly ToolControl[], key: string, values: ToolValues):
  { readonly control: ToolControl; readonly value: number } | undefined {
  for (const control of controls) {
    if (control.kind === "choice") {
      const option = control.options.find((candidate) => candidate.shortcut === key);
      if (option) return { control, value: option.value };
    } else if (control.shortcut === key) return { control, value: values[control.id] === 1 ? 0 : 1 };
  }
  return undefined;
}
