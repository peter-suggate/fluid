/**
 * What a pass declares so the frame can draw its picture.
 *
 * The rule this framework exists to enforce: a visualization is defined next to
 * the code that produces its data, and assembled centrally. Colours, legends and
 * the arithmetic behind a figure live with the shader or oracle that chose them,
 * because that is the only place they can be kept honest — and the frame never
 * has to be told a visualization exists.
 *
 * There are deliberately **two kinds**, and collapsing them is the way this goes
 * wrong:
 *
 * - A `decoration` is host-built world-space line work. Every decoration in a
 *   frame merges into one buffer and draws in one instanced call, however many
 *   passes contributed, so any number can be on at once.
 * - A `field` is a per-pixel WGSL fragment sampled inside a volume or slice
 *   raymarch. It cannot be host-built — that is why the technique modes live in
 *   a shader at all — and because fields are volume-composited, one shows at a
 *   time.
 *
 * The subject a visualization draws is opaque here. Each definition narrows it
 * with its own `accepts` guard, which is what keeps this module from having to
 * know about pressure cells, rays, or anything else a pass might select.
 */
import {
  DecorationBuilder,
  decorationSpaceValid,
  type DecorationSpace,
} from "./visualization-decorations";
import type { DecorationGeometry } from "./webgpu-decoration-overlay";

/** How the visualization reaches the screen. See the module comment. */
export type VisualizationKind = "decoration" | "field";

/**
 * The picture a field view is offered under when a surface has room for a
 * glyph and not for a name.
 *
 * A closed vocabulary declared here rather than a component import, for the
 * reason `EditorActionIcon` is: core may not import React, so a pass says
 * *what* its view is a picture of and the component layer decides what that
 * picture looks like. Declaring one is also the only way a view reaches the
 * quick bar at the tank corner — the strip is the handful of views worth
 * turning on without opening anything, and the passes that own them are the
 * only ones who can say which those are.
 */
export type FieldVisualizationIcon =
  | "grid" | "levels" | "density" | "surface" | "speed" | "pressure" | "flow" | "tracers";

/**
 * The glyph a legend entry stands for.
 *
 * A swatch alone cannot tell an arrow from a box from a dashed box, and those
 * are different claims: an arrow is a coupling with a direction, a box is an
 * extent, and a dash means derived rather than measured. The decorator that
 * draws the line work names its mark for the same reason it names its colour —
 * it is the only place the two can be kept in step with what is on screen.
 */
export type VisualizationLegendMark =
  | "line" | "arrow" | "box" | "dashed-box" | "ring" | "cross" | "point";

/** One entry of the legend a view is read with. */
export interface VisualizationLegendEntry {
  readonly swatch: string;
  readonly label: string;
  /** The shape drawn in this colour. Defaults to `line`. */
  readonly mark?: VisualizationLegendMark;
}

/**
 * A figure the visualization contributes to a HUD.
 *
 * `evidence` carries the same distinction the line work draws with solid versus
 * dashed: `gathered` was published by the frame and read back, `scheduled` is
 * what the encoded command graph does and may have been cut short. A consumer
 * must never add the two into one total.
 */
export interface VisualizationNote {
  readonly label: string;
  readonly value: string;
  readonly detail?: string;
  readonly evidence: "gathered" | "scheduled";
}

interface VisualizationCommon {
  /** Globally unique, `pass/name`. Stable: the UI persists toggles by it. */
  readonly id: string;
  /** The pass or oracle that owns the data. Groups the toggles, names the source. */
  readonly pass: string;
  readonly label: string;
  readonly description: string;
  /** Where the numbers come from, in one phrase. Shown under the label. */
  readonly source?: string;
  /** The entry's own colour, for a toggle chip. Distinct from the legend. */
  readonly swatch?: `#${string}`;
  readonly legend?: readonly VisualizationLegendEntry[];
  /** Free-form tag a consumer may group or filter by; the framework ignores it. */
  readonly group?: string;
  /** Off by default when a view is expensive or narrow. Defaults to on. */
  readonly defaultEnabled?: boolean;
}

export interface DecorationContext {
  readonly space: DecorationSpace;
  /**
   * Whether the subject is frozen (`selected`) or following the pointer
   * (`hover`).
   *
   * **Not a level of detail.** Both states draw the same picture. Hovering used
   * to draw an outline and withhold everything else, which made pinning a
   * prerequisite for reading anything and turned exploration into a sequence of
   * commitments; a build keyed on the drawn facts already skips the rebuild
   * while the pointer sits inside one cell, so the saving was never the point.
   * A view may use this for prominence. It must not use it to decide content.
   */
  readonly emphasis: "hover" | "selected";
  readonly widthScale: number;
}

export interface DecorationVisualization<Subject = unknown> extends VisualizationCommon {
  readonly kind: "decoration";
  accepts(subject: unknown): subject is Subject;
  /**
   * Everything this decoration's picture depends on.
   *
   * A gather generally re-runs every frame, so a subject's revision counter
   * ticks even when nothing drawn has changed. Keying on the drawn facts is
   * what lets the assembler skip the upload while a pointer sits still or a
   * camera orbits a frozen selection.
   */
  key(subject: Subject, context: DecorationContext): string;
  build(
    subject: Subject, context: DecorationContext, into: DecorationBuilder,
  ): readonly VisualizationNote[] | void;
}

/**
 * A per-pixel field.
 *
 * `mode` is the integer the shipping overlay still dispatches on. It stays part
 * of the declaration until the field harness is assembled from these entries
 * rather than from a hand-written branch chain, at which point it becomes an
 * implementation detail of whatever compiles the program.
 */
export interface FieldVisualization extends VisualizationCommon {
  readonly kind: "field";
  /** The overlay mode string the UI already selects by. */
  readonly mode: string;
  /**
   * The integer the technique overlay dispatches on. Absent for a field the
   * generic overlay renders, which has no code of its own. It stays part of the
   * declaration until the harness is assembled from these entries rather than
   * from a hand-written branch chain.
   */
  readonly modeCode?: number;
  /** Slice axis the view is authored for; `volume` is the ray-integrated form. */
  readonly axis: "x" | "y" | "z" | "volume";
  /**
   * The view draws its own geometry over the finished frame instead of being
   * sampled inside a raymarch, so it has no slice plane to choose.
   *
   * Declared rather than inferred from `axis`, because `volume` already means
   * something else here — ray-integrated through the whole depth — and a picker
   * that offered X/Y/Z for a view with no plane would be three buttons that
   * change nothing. Such a view keeps the slice control as its opacity, which
   * is what `volume` already uses it for.
   */
  readonly planeless?: boolean;
  /** Paper figure this view reproduces, when it reproduces one. */
  readonly figure?: string;
  /**
   * Omitted from generic catalog-driven pickers. A focused panel may still
   * select one explicitly; declaring it hidden prevents unrelated catalog
   * additions from silently expanding that panel or the Performance picker.
   */
  readonly hidden?: boolean;
  /**
   * The view is worth one glyph on the quick bar, and this is what it is a
   * picture of.
   *
   * Deliberately sparse: the bar is the short list a reader turns on while
   * watching the water, and a catalog where every entry claimed an icon would
   * be the full picker again with the labels taken away. Declaring one here is
   * the pass asserting that its view answers a question worth asking before
   * anything is selected.
   */
  readonly icon?: FieldVisualizationIcon;
}

export type Visualization = DecorationVisualization<never> | FieldVisualization;

/** Narrowing helpers, so a catalog can be one heterogeneous list. */
export function isDecorationVisualization(
  visualization: Visualization,
): visualization is DecorationVisualization<never> {
  return visualization.kind === "decoration";
}

export function isFieldVisualization(
  visualization: Visualization,
): visualization is FieldVisualization {
  return visualization.kind === "field";
}

/**
 * Declare a decoration with its subject type inferred from `accepts`.
 *
 * The cast at the boundary is the price of one heterogeneous catalog: every
 * definition narrows its own subject, so the assembler can hand each one an
 * `unknown` and let the guard decide.
 */
export function decorationVisualization<Subject>(
  definition: DecorationVisualization<Subject>,
): DecorationVisualization<never> {
  return definition as unknown as DecorationVisualization<never>;
}

export function fieldVisualization(definition: FieldVisualization): FieldVisualization {
  return definition;
}

/** True when `subject` is an object carrying every named field. */
export function hasFields(subject: unknown, fields: readonly string[]): boolean {
  if (typeof subject !== "object" || subject === null) return false;
  return fields.every((field) => field in subject);
}

export interface AssembleDecorationsOptions {
  readonly definitions: readonly Visualization[];
  /**
   * What is selected. Each definition decides which of these it can draw, so a
   * frame holding a picked cell and a traced ray at once assembles both into
   * the same buffer without either knowing about the other.
   */
  readonly subjects: readonly unknown[];
  readonly space: DecorationSpace;
  readonly emphasis?: "hover" | "selected";
  readonly widthScale?: number;
  /** Absent enables everything a definition does not itself default off. */
  readonly enabled?: (definition: DecorationVisualization<never>) => boolean;
}

export interface AssembledDecorationEntry {
  readonly id: string;
  readonly pass: string;
  readonly label: string;
  readonly segmentCount: number;
  readonly notes: readonly VisualizationNote[];
}

export interface AssembledDecorations {
  readonly geometry: DecorationGeometry;
  readonly entries: readonly AssembledDecorationEntry[];
  /**
   * Identity of the assembled picture. Unchanged means the GPU buffer already
   * holds it, so a consumer can skip the rebuild and the upload entirely.
   */
  readonly key: string;
}

const EMPTY_GEOMETRY: DecorationGeometry = Object.freeze({
  segments: new Float32Array(0), segmentCount: 0,
});

/**
 * Merge every enabled decoration that accepts the subject into one draw.
 *
 * Contributors write into a shared builder in catalog order, so what a pass
 * declares first draws first, and the whole overlay is one upload. A definition
 * that throws is dropped rather than taking the frame down with it: a broken
 * diagnostic must not break the thing it is diagnosing.
 */
export function assembleDecorations(
  options: AssembleDecorationsOptions,
): AssembledDecorations {
  const context: DecorationContext = {
    space: options.space,
    emphasis: options.emphasis ?? "selected",
    widthScale: Number.isFinite(options.widthScale)
      ? Math.max(0.1, Math.min(4, options.widthScale!)) : 1,
  };
  if (!decorationSpaceValid(options.space)) {
    return { geometry: EMPTY_GEOMETRY, entries: [], key: "invalid-space" };
  }
  const builder = new DecorationBuilder(options.space);
  const entries: AssembledDecorationEntry[] = [];
  const keyParts: string[] = [context.emphasis, String(context.widthScale),
    options.space.dimensions.join(","), options.space.extent_m.join(","),
    options.space.origin_m.join(",")];

  for (const definition of options.definitions) {
    if (!isDecorationVisualization(definition)) continue;
    if (options.enabled ? !options.enabled(definition) : definition.defaultEnabled === false) continue;
    for (const subject of options.subjects) {
      if (!definition.accepts(subject)) continue;
      const before = builder.segmentCount;
      let notes: readonly VisualizationNote[] = [];
      try {
        keyParts.push(`${definition.id}:${definition.key(subject as never, context)}`);
        notes = definition.build(subject as never, context, builder) ?? [];
      } catch {
        // A decorator that cannot describe this subject contributes nothing. The
        // frame it is drawn over is not its to break.
        continue;
      }
      const segmentCount = builder.segmentCount - before;
      if (segmentCount === 0 && notes.length === 0) continue;
      entries.push({
        id: definition.id, pass: definition.pass, label: definition.label,
        segmentCount, notes,
      });
    }
  }

  return { geometry: builder.finish(), entries, key: keyParts.join("|") };
}

/** Every field declared, in catalog order — the list the mode UI is built from. */
export function fieldVisualizations(
  definitions: readonly Visualization[],
): readonly FieldVisualization[] {
  return definitions.filter(isFieldVisualization);
}

/** Every decoration declared, whether or not it accepts the current subject. */
export function decorationVisualizations(
  definitions: readonly Visualization[],
): readonly DecorationVisualization<never>[] {
  return definitions.filter(isDecorationVisualization);
}
