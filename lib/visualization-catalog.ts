/**
 * Every visualization the app can draw, in one list.
 *
 * Assembly is central; declaration is not. Each entry below is defined beside
 * the pass, shader or oracle that produces its data, and this module only
 * decides the order they compose in — which for decorations is also the order
 * they draw, so later entries sit over earlier ones.
 *
 * This is a static list rather than a mutable registry on purpose. A
 * `register()` singleton would be shorter to write and worse to live with:
 * ordering would depend on import timing, hot reload would double-register, and
 * a test could not build a catalog without dragging in the whole app. Adding a
 * visualization means exporting it from its pass and adding one line here.
 */
import { blastRadiusVisualizations } from "./fluid-blast-radius";
import { fineBandCellVisualizations } from "./fine-band-cell-visualizations";
import { fineFloodVisualizations } from "./fine-flood-provenance";
import { fluidCellVisualizations } from "./fluid-cell-visualizations";
import { octreeFieldVisualizations } from "./octree-technique-debug";
import { svoPixelTraceVisualizations } from "./svo-pixel-trace";
import {
  decorationVisualizations,
  fieldVisualizations,
  type Visualization,
  type VisualizationLegendMark,
} from "./visualization-registry";

export const VISUALIZATION_CATALOG: readonly Visualization[] = Object.freeze([
  // Gathered state first, so a derived shell never draws over the cell it
  // describes.
  ...fluidCellVisualizations,
  // The fine band's detail draws over the cell it describes and under the
  // derived shells, which is the order a reader builds the picture up in:
  // the leaf, then what is inside it, then the bounds around it.
  ...fineBandCellVisualizations,
  ...fineFloodVisualizations,
  ...blastRadiusVisualizations,
  ...svoPixelTraceVisualizations,
  ...octreeFieldVisualizations,
]);

/** Field views, in catalog order — the list the mode UI is built from. */
export const VISUALIZATION_FIELDS = fieldVisualizations(VISUALIZATION_CATALOG);

/** Decorations, whether or not any are applicable to the current selection. */
export const VISUALIZATION_DECORATIONS = decorationVisualizations(VISUALIZATION_CATALOG);

/** Ids grouped by owning pass, for a panel that lists them under headings. */
export function visualizationsByPass(
  definitions: readonly Visualization[] = VISUALIZATION_CATALOG,
): readonly { readonly pass: string; readonly entries: readonly Visualization[] }[] {
  const grouped = new Map<string, Visualization[]>();
  for (const definition of definitions) {
    const existing = grouped.get(definition.pass);
    if (existing) existing.push(definition);
    else grouped.set(definition.pass, [definition]);
  }
  return [...grouped.entries()].map(([pass, entries]) => ({ pass, entries: Object.freeze(entries) }));
}

/**
 * Catalog ids for a set of group tags.
 *
 * A consumer keeps its reader's vocabulary — "stencil", "cone", "bricks" — and
 * this turns it into the ids the assembler enables, so neither side has to
 * learn the other's names. Groups are globally unique across passes
 * (`duplicateVisualizationGroups` is what keeps that true), which is why no
 * pass filter is needed here.
 */
export function visualizationIdsForGroups(
  groups: readonly string[],
  definitions: readonly Visualization[] = VISUALIZATION_CATALOG,
): readonly string[] {
  const wanted = new Set(groups);
  return definitions
    .filter((definition) => definition.group !== undefined && wanted.has(definition.group))
    .map((definition) => definition.id);
}

/**
 * Catalog entries for a set of group tags, in catalog order.
 *
 * What a toggle strip is built from: the label, swatch and description on each
 * chip come from the pass that draws it, so a chip cannot describe a colour the
 * decorator no longer uses.
 */
export function visualizationsForGroups(
  groups: readonly string[],
  definitions: readonly Visualization[] = VISUALIZATION_CATALOG,
): readonly Visualization[] {
  const wanted = new Set(groups);
  return definitions.filter(
    (definition) => definition.group !== undefined && wanted.has(definition.group));
}

/**
 * The glyph a view's own swatch stands for.
 *
 * A legend has to match the shape on screen, not only its colour, and the shape
 * is the decorator's own business — so it is read from the first legend entry
 * the pass declared rather than chosen by whatever is drawing the legend. `line`
 * is the default because a view that never said is almost always segments.
 */
export function visualizationMark(definition: Visualization): VisualizationLegendMark {
  return definition.legend?.[0]?.mark ?? "line";
}

/**
 * Group tags claimed by more than one pass.
 *
 * `visualizationIdsForGroups` resolves a tag without asking which pass owns it,
 * so a collision would silently switch on another pass's line work. Exercised by
 * the catalog test, for the same reason duplicate ids are.
 */
export function duplicateVisualizationGroups(
  definitions: readonly Visualization[] = VISUALIZATION_CATALOG,
): readonly string[] {
  const owner = new Map<string, string>(), duplicates = new Set<string>();
  for (const definition of definitions) {
    if (definition.group === undefined) continue;
    const existing = owner.get(definition.group);
    if (existing !== undefined && existing !== definition.pass) duplicates.add(definition.group);
    else owner.set(definition.group, definition.pass);
  }
  return [...duplicates];
}

/**
 * Duplicate ids are a programming error, not a runtime condition: two entries
 * sharing an id would make a persisted toggle ambiguous. Exercised by the
 * catalog test rather than thrown at import, so a bad entry fails a build rather
 * than a user's frame.
 */
export function duplicateVisualizationIds(
  definitions: readonly Visualization[] = VISUALIZATION_CATALOG,
): readonly string[] {
  const seen = new Set<string>(), duplicates = new Set<string>();
  for (const { id } of definitions) {
    if (seen.has(id)) duplicates.add(id);
    seen.add(id);
  }
  return [...duplicates];
}
