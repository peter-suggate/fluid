import type { EditorActionTone } from "../editor-action";
import { solidVoxelClearPreview } from "../editor-solid-voxel";
import type { SceneDescription } from "../model";
import { sceneLatticeDimensions } from "../scene-lattice";
import type { VoxelBlock } from "../solid-patch/kind";
import type { SolidWorldCoordinate, SolidWorldVoxelPatch } from "../solid-world";
import { materialControl, toolMaterialId } from "./controls";
import { controlValues, SHARED_TOOL_VALUES_ID, storedControlValues, validateControls, type StoredToolValues,
  type ToolControl, type ToolValues } from "./plugin";
import type { VoxelReader, VoxelRegion } from "./voxel-access";
import { voxelReaderForScene } from "./voxel-reader";

/**
 * What a selection can be told to do, as plugins.
 *
 * The other half of the tool contract in `./plugin`. A tool turns a *gesture*
 * into patches; a verb turns a *selection* into patches — the region is
 * already standing, so there is nothing to drag and nothing to preview, and
 * the whole of a verb is one pure function from what is selected to what
 * should change. Hosts own everything else: where a verb is offered (the
 * selection's ring, the rail's selection line, a key), and how its result is
 * committed, which is exactly how a stroke's is (`./verb-runtime`).
 *
 * So a verb never touches a store, never bakes and never names a host, and
 * adding one is a file in `./verbs` and a line in its registry.
 */

/** Voxels lifted off the scene by a copy, and where they stood. Never document data. */
export interface VoxelClipboard {
  readonly block: VoxelBlock;
  /** The lattice cell the block's first cell was copied from. */
  readonly minimum: SolidWorldCoordinate;
}

/** What a verb is pointed at: enough to say whether it applies, before any option is read. */
export interface VoxelRegionVerbTarget {
  readonly scene: SceneDescription;
  /** The running solver's id; the verbs that edit solids are gated on it exactly as the tools are. */
  readonly methodId: string;
  readonly region: VoxelRegion;
  /** The solids of `scene`. Lazy: touching it is what builds the reader. */
  readonly reader: VoxelReader;
  /** How many cells of the region are solid. Lazy: counted on first read. */
  readonly solidCount: number;
  readonly clipboard?: VoxelClipboard;
}

/** A lattice axis and a way along it. */
export interface VoxelVerbAxis {
  readonly axis: 0 | 1 | 2;
  readonly sign: -1 | 1;
}

/**
 * How the reader is looking at the lattice, as far as a verb may know: the
 * horizontal lattice axis nearest screen-right, and the one nearest the way
 * the camera faces. What an arrow key means is a direction on the screen, and
 * this is what turns it into cells.
 */
export interface VoxelVerbView {
  readonly right: VoxelVerbAxis;
  readonly forward: VoxelVerbAxis;
}

/** The view of a camera at azimuth zero: looking down −Z with +X to its right. A scripted caller's default. */
export const DEFAULT_VOXEL_VERB_VIEW: VoxelVerbView = Object.freeze({
  right: Object.freeze({ axis: 0, sign: 1 }), forward: Object.freeze({ axis: 2, sign: -1 }),
});

/**
 * The view of a camera orbited to `azimuth_rad` (`CameraState`'s convention:
 * the eye stands at +Z of its target at zero and swings toward +X).
 *
 * Screen-right takes whichever of X and Z it lies nearer, X on the exact
 * diagonal, and forward takes the other, so the two are never the same axis.
 */
export function voxelVerbView(azimuth_rad: number): VoxelVerbView {
  const sin = Math.sin(azimuth_rad), cos = Math.cos(azimuth_rad);
  const sign = (value: number): -1 | 1 => value < 0 ? -1 : 1;
  // Right is (cos, −sin) and forward is (−sin, −cos) in (x, z).
  return Math.abs(cos) >= Math.abs(sin)
    ? { right: { axis: 0, sign: sign(cos) }, forward: { axis: 2, sign: sign(-cos) } }
    : { right: { axis: 2, sign: sign(-sin) }, forward: { axis: 0, sign: sign(-sin) } };
}

/** A target, and the options the verb runs with. */
export interface VoxelRegionVerbContext extends VoxelRegionVerbTarget {
  /** The verb's own controls, resolved, and over them whatever the chosen variant fixes. */
  readonly values: ToolValues;
  /** The material to write: the variant's when it names one, otherwise the editor's chosen one. */
  readonly materialId: number;
  /** Which lattice axes the screen's own directions lie along. */
  readonly view: VoxelVerbView;
}

export interface VoxelRegionVerbResult {
  /** Appended to the document as one edit. Empty for a verb that only selects or copies. */
  readonly patches: readonly SolidWorldVoxelPatch[];
  /** The selection afterwards: a new region, `null` for none, absent to leave it standing. */
  readonly region?: VoxelRegion | null;
  /** Replaces the clipboard when present. */
  readonly clipboard?: VoxelClipboard;
  /**
   * A voxel tool to arm once everything else has landed, by its registry id.
   * This is how a paste hands the clipboard to the pointer. The selection is
   * left to the store, exactly as when the rail arms the tool.
   */
  readonly armTool?: string;
  /** The notice, and the undo entry's label: "Cleared 12 solid voxels". Empty for a verb with nothing to announce. */
  readonly caption: string;
}

/**
 * One way of running a verb: an axis to flip about, a direction to nudge in, a
 * material to paint with.
 *
 * A variant fixes values the verb reads from `context.values`, over its
 * controls. It is how one verb is three wedges on the ring and six keys on the
 * keyboard without being three verbs, and without a host knowing what an axis is.
 */
export interface VoxelVerbVariant {
  /** Unique within its verb. */
  readonly id: string;
  /** Short: it is drawn after the verb's own label ("Flip" then "X"). */
  readonly label: string;
  readonly hint?: string;
  /** SVG path in a 24 box, drawn instead of the label where there is room for a mark only. */
  readonly icon?: string;
  /** A CSS colour, when the variant *is* one. */
  readonly swatch?: string;
  /** Key chords that run the verb this way. Same grammar and the same rules as the verb's own. */
  readonly keys?: readonly string[];
  /** What the verb reads this way. `material` also sets `context.materialId`. */
  readonly values: ToolValues;
  /**
   * Where the variant is offered. Absent means everywhere the verb is; `"ring"`
   * keeps it off the bar, which has a control for the same choice; `"keys"`
   * keeps it off both, for a variant only a modifier reaches.
   */
  readonly surface?: "ring" | "keys";
}

/** A verb, and the way of running it that a key or a wedge named. */
export interface VoxelVerbBinding {
  readonly verb: VoxelRegionVerbPlugin;
  readonly variant?: VoxelVerbVariant;
}

export interface VoxelRegionVerbPlugin {
  readonly id: string;
  readonly version: 1;
  readonly ui: {
    readonly label: string;
    readonly hint: string;
    /** SVG path in a 24 box, as a tool's icon is. */
    readonly icon: string;
    /** Verbs of one group stand together wherever verbs are listed. */
    readonly group: string;
    readonly order: number;
    /** Marks the destructive ones the way the ring already marks Delete. Defaults to the selection's own tone. */
    readonly tone?: EditorActionTone;
    /**
     * Key chords that run the verb while a voxel selection stands. The first is
     * the one shown. A chord is a `KeyboardEvent.key` — a single character in
     * lower case — behind `Mod+` for ⌘ or Ctrl and then `Shift+`, in that
     * order: `"Delete"`, `"Mod+c"`, `"Shift+ArrowLeft"`. See `voxelVerbKeyChord`.
     */
    readonly keys?: readonly string[];
    /** The ways of running it, each a wedge of the verb's own sub-ring and a mark on the bar. */
    readonly variants?: readonly VoxelVerbVariant[];
    /**
     * The sub-ring the verb's wedge stands in, by its caption. The ring nests
     * one level only, so a verb with variants — which opens a sub-ring of its
     * own — cannot also name one.
     */
    readonly menu?: string;
    /** False keeps the verb off the ring: a verb worth a key or a button but not a wedge. */
    readonly ring?: boolean;
    /**
     * True for a verb that needs no selection — select everything, paste. With
     * nothing selected it is pointed at the whole lattice, and its keys answer
     * anywhere in EDIT.
     */
    readonly standalone?: boolean;
  };
  /** Options, drawn beside the verbs. Shared controls read the same values the tools do. */
  readonly controls?: readonly ToolControl[];
  /** Why the verb cannot run on this target, as a sentence for the reader; undefined when it can. */
  unavailable(target: VoxelRegionVerbTarget): string | undefined;
  /** Pure. Called only when `unavailable` returned nothing for the same target. */
  run(context: VoxelRegionVerbContext): VoxelRegionVerbResult;
}

/** The owner a verb's own (non-shared) control values are stored under, clear of every tool id. */
export const verbValuesOwner = (verb: Pick<VoxelRegionVerbPlugin, "id">): string => `verb:${verb.id}`;

/** A chord as a verb declares one: `Mod+`, then `Shift+`, then a named key or one lower-case character. */
const KEY_CHORD = /^(Mod\+)?(Shift\+)?([A-Z][A-Za-z0-9]+|[^A-Z\s])$/;

/**
 * The chord a key event spells, in the grammar verbs declare theirs in.
 *
 * ⌘ and Ctrl are one modifier, so a chord means the same on every platform.
 * `undefined` under Alt, which belongs to the tools' own modifiers.
 */
export function voxelVerbKeyChord(event: Pick<KeyboardEvent, "key" | "shiftKey" | "metaKey" | "ctrlKey" | "altKey">): string | undefined {
  if (event.altKey || !event.key) return undefined;
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  return `${event.metaKey || event.ctrlKey ? "Mod+" : ""}${event.shiftKey ? "Shift+" : ""}${key}`;
}

const KEY_GLYPHS: Readonly<Record<string, string>> = {
  Delete: "del", Backspace: "⌫", Escape: "esc", ArrowLeft: "←", ArrowRight: "→", ArrowUp: "↑", ArrowDown: "↓",
  PageUp: "pg↑", PageDown: "pg↓",
};

/** A chord as a legend prints it: `"Mod+c"` is `⌘c`, `"Shift+ArrowLeft"` is `⇧←`. */
export function voxelVerbKeyLabel(chord: string): string {
  const [, mod, shift, key] = KEY_CHORD.exec(chord) ?? [];
  if (!key) throw new Error(`Not a voxel verb key chord: "${chord}"`);
  return `${mod ? "⌘" : ""}${shift ? "⇧" : ""}${KEY_GLYPHS[key] ?? key.toLowerCase()}`;
}

export function createVoxelRegionVerbRegistry(plugins: readonly VoxelRegionVerbPlugin[]) {
  const ids = new Set<string>();
  const keys = new Map<string, VoxelVerbBinding>();
  const shared = new Map<string, ToolControl>();
  for (const plugin of plugins) {
    if (!plugin.id.trim() || ids.has(plugin.id)) throw new Error(`Duplicate or empty voxel verb id: ${plugin.id}`);
    ids.add(plugin.id);
    const variants = plugin.ui.variants ?? [];
    if (plugin.version !== 1 || !plugin.ui.label.trim() || !plugin.ui.hint.trim() || !plugin.ui.group.trim()
      || !plugin.ui.icon.trim() || !Number.isFinite(plugin.ui.order)
      || (plugin.ui.menu !== undefined && (!plugin.ui.menu.trim() || variants.length > 0))
      || new Set(variants.map((variant) => variant.id)).size !== variants.length
      || variants.some((variant) => !variant.id.trim() || !variant.label.trim()
        || !Object.values(variant.values).every(Number.isFinite))
      || typeof plugin.unavailable !== "function" || typeof plugin.run !== "function") {
      throw new Error(`Invalid voxel verb definition: ${plugin.id}`);
    }
    // A key names one verb, run one way; a second claimant would be silently unreachable.
    const claim = (chords: readonly string[] | undefined, variant?: VoxelVerbVariant) => {
      for (const chord of chords ?? []) {
        if (!KEY_CHORD.test(chord) || keys.has(chord)) {
          throw new Error(`Invalid or duplicate key "${chord}" on voxel verb ${plugin.id}`);
        }
        keys.set(chord, variant ? { verb: plugin, variant } : { verb: plugin });
      }
    };
    claim(plugin.ui.keys);
    for (const variant of variants) claim(variant.keys, variant);
    validateControls(verbValuesOwner(plugin), plugin.controls ?? [], shared);
  }
  const ordered = Object.freeze([...plugins].sort((a, b) => a.ui.order - b.ui.order));
  return Object.freeze({
    verbs: ordered,
    get: (id: string | undefined) => ordered.find((verb) => verb.id === id),
    /** The verb a chord runs, and the variant it runs it as. See `voxelVerbKeyChord`. */
    forKey: (chord: string | undefined): VoxelVerbBinding | undefined => chord === undefined ? undefined : keys.get(chord),
  });
}

/** The variant of a verb an id names. An id the verb does not declare is a wiring mistake, not a choice. */
export function voxelVerbVariant(verb: VoxelRegionVerbPlugin, variantId: string | undefined): VoxelVerbVariant | undefined {
  if (variantId === undefined) return undefined;
  const variant = verb.ui.variants?.find((candidate) => candidate.id === variantId);
  if (!variant) throw new Error(`Voxel verb ${verb.id} has no variant "${variantId}"`);
  return variant;
}

/** Every cell of the scene's lattice: what a verb that needs no selection is pointed at when there is none. */
export function sceneLatticeRegion(scene: SceneDescription): VoxelRegion {
  const [i, j, k] = sceneLatticeDimensions(scene);
  return { minimum: [0, 0, 0], maximumExclusive: [i, j, k] };
}

/**
 * How many cells of a region are solid.
 *
 * A box is counted over the scene's baked pages, which is what the selection's
 * summary has always done. A masked region is counted cell by cell through the
 * voxel reader, because only its own cells are in question.
 */
function regionSolidCount(scene: SceneDescription, region: VoxelRegion, reader: () => VoxelReader): number {
  if (!region.mask) return solidVoxelClearPreview(scene, region, 0).affectedCount;
  const size = region.minimum.map((value, axis) => region.maximumExclusive[axis]! - value);
  let count = 0;
  for (let z = 0; z < size[2]!; z += 1) for (let y = 0; y < size[1]!; y += 1) for (let x = 0; x < size[0]!; x += 1) {
    if (region.mask[x + size[0]! * (y + size[1]! * z)] === 0) continue;
    if (reader().solid([region.minimum[0] + x, region.minimum[1] + y, region.minimum[2] + z])) count += 1;
  }
  return count;
}

/** A target over a standing selection. The reader and the count are built only if a verb asks. */
export function voxelRegionVerbTarget(input: {
  readonly scene: SceneDescription;
  readonly methodId: string;
  readonly region: VoxelRegion;
  readonly clipboard?: VoxelClipboard;
}): VoxelRegionVerbTarget {
  let reader: VoxelReader | undefined;
  const read = () => reader ??= voxelReaderForScene(input.scene);
  return {
    ...input,
    get reader() { return read(); },
    get solidCount() {
      let counts = solidCounts.get(input.scene);
      if (!counts) solidCounts.set(input.scene, counts = new WeakMap());
      let count = counts.get(input.region);
      if (count === undefined) counts.set(input.region, count = regionSolidCount(input.scene, input.region, read));
      return count;
    },
  };
}

/**
 * Counts already made. A scene document and a region are both immutable, and
 * the same pair is asked about by the bar, the ring, the entity's summary and
 * the verb that finally runs — one walk of the box serves them all.
 */
const solidCounts = new WeakMap<SceneDescription, WeakMap<VoxelRegion, number>>();

/**
 * The context a verb runs with: its target, its controls resolved from what
 * the host has stored, and over those whatever the chosen variant fixes.
 */
export function voxelRegionVerbContext(verb: VoxelRegionVerbPlugin, target: VoxelRegionVerbTarget,
  stored: StoredToolValues, options: { readonly variant?: VoxelVerbVariant; readonly view?: VoxelVerbView } = {},
): VoxelRegionVerbContext {
  const controls = verb.controls ?? [];
  const fixed = options.variant?.values ?? {};
  // Built on the target, not spread from it: a spread would read — and so build — the lazy members.
  return Object.create(target, {
    values: {
      value: { ...controlValues(controls, storedControlValues(verbValuesOwner(verb), controls, stored)), ...fixed },
      enumerable: true,
    },
    materialId: {
      value: toolMaterialId(controlValues([materialControl], { ...stored[SHARED_TOOL_VALUES_ID], ...fixed })),
      enumerable: true,
    },
    view: { value: options.view ?? DEFAULT_VOXEL_VERB_VIEW, enumerable: true },
  }) as VoxelRegionVerbContext;
}
