/**
 * The liquid-fraction view, defined once for every renderer that draws it.
 *
 * V/K — conserved liquid volume over the open capacity of the cell holding it —
 * is one quantity, and a reader who learns to read it in the 2-D advance lab
 * should not have to learn it again in the 3-D dense-grid overlay. Before this
 * module the two agreed by transliteration: the shader carried its own floor,
 * its own knee and three hand-typed colours; `advance-lab/lenses.ts` carried a
 * second copy of the floor and the ramp; and the registry legend carried a
 * third copy of the colours as `#rrggbb` swatches nothing checked. Three copies
 * of one definition is three chances to drift, and all three had.
 *
 * So the bands, the thresholds, the arithmetic and the colours live here, and
 * each renderer consumes them in its own terms:
 *
 * - The WGSL overlay interpolates {@link fractionViewShaderConstants} into its
 *   template and reads the constants by name, so the pixels are these numbers.
 * - The advance lab maps {@link FractionTone} onto its own CSS-variable palette
 *   in one table, so the 2-D picture follows the page theme while still
 *   agreeing about *which band a cell is in*.
 * - The visualization registry hands {@link FRACTION_VIEW_BANDS} to a legend,
 *   which is therefore derived rather than hand-kept.
 *
 * This module is in `lib/core` rather than beside the adaptive-volume method
 * that produces V/K because both of its renderers are core: `lib/core` may not
 * import a method (see `tools/check-module-boundaries.ts` rule 2), so a copy
 * under `lib/methods/adaptive-volume/` could not be the shared one.
 */

/**
 * Below this a cell is vacuum, not dilute.
 *
 * Six decades under a full cell — the bottom of the residue band transport
 * actually produces. A cell under it keeps its grid lines and nothing else:
 * "is there any liquid here at all" is the first question this view has to
 * answer, and a floor of tinted haze over empty cells is how that answer gets
 * lost.
 */
export const FRACTION_FLOOR = 1e-6;

/**
 * Half a cell: where the projection starts calling the cell liquid.
 *
 * A physical threshold, not a cosmetic one — it is `pressureLiquid`. Below it
 * mass is carried by transport and ignored by pressure, which is why the views
 * spend their tint there and leave the liquid half to the geometry.
 */
export const FRACTION_LIQUID_KNEE = 0.5;

/**
 * Past capacity: V is more than the open volume K of the cell holding it.
 *
 * One threshold, where there were three. The 2-D lab washed a cell alarm at
 * `> 1 + 1e-4` while boxing it amber at `> 1 + 1e-6`, and the 3-D overlay
 * hatched at `> 1.000001`. Two of the three already agreed, and they are the
 * two that are right: an excess is a fault the projection has to drain, and the
 * only excess worth suppressing is the one a float32 volume record cannot tell
 * from a full cell. `1e-4` was a tolerance nothing justified — it hid four
 * decades of real overfill from the wash while the box beside it drew.
 */
export const FRACTION_OVERFULL = 1 + FRACTION_FLOOR;

/** The four readings of V/K, in increasing order of fill. */
export type FractionBand = "vacuum" | "dilute" | "liquid" | "overfull";

/**
 * A renderer-neutral colour role. Not a colour and not a CSS variable: the
 * canvas resolves these against the page theme, the shader against its own
 * display-space literals, and neither has to know about the other's palette.
 */
export type FractionTone = "empty" | "residue" | "liquid" | "excess";

/** Which band a cell's V/K is in. The one place the thresholds are spelled. */
export function fractionBand(fill: number): FractionBand {
  if (!Number.isFinite(fill) || fill <= FRACTION_FLOOR) return "vacuum";
  if (fill > FRACTION_OVERFULL) return "overfull";
  if (fill < FRACTION_LIQUID_KNEE) return "dilute";
  return "liquid";
}

/** True when the cell holds more liquid than it has room for. */
export function cellFillIsOverCapacity(fill: number): boolean {
  return fractionBand(fill) === "overfull";
}

/** Linear diagnostic opacity for authoritative cell fill. Zero stays clear. */
export function cellFillOpacity(fill: number): number {
  return Math.min(1, Math.max(0, Number.isFinite(fill) ? fill : 0));
}

/**
 * Where a sub-knee fraction sits on the residue ramp, in [0, 1].
 *
 * Volume fraction is not a linear quantity down here. Transport leaves residue
 * across every decade between the floor and about 10⁻², and a linear ramp over
 * [0, ½] buries all of it in the bottom two percent: every residue cell then
 * draws the same near-nothing at the same near-zero alpha, which is the one
 * failure this view exists to prevent. A unit of the ramp is a fixed number of
 * decades instead, so the low end separates from itself.
 *
 * The shader's `dilute` term in the surface-density branch is this same
 * expression, over the same two constants.
 */
export function fractionResidueRamp(fill: number): number {
  return Math.min(1, Math.max(0,
    Math.log2(Math.max(fill, FRACTION_FLOOR) / FRACTION_FLOOR)
    / Math.log2(FRACTION_LIQUID_KNEE / FRACTION_FLOOR)));
}

/**
 * The fraction as the fewest characters that keep it honest.
 *
 * A cell is a handful of pixels wide, so the readout is sized to the answer
 * rather than formatted uniformly: `.42` for anything the two decimals can
 * carry, a bare `1` for a full cell, and `1e-4` once two decimals would round
 * a resolved residue cell to `.00` and make it indistinguishable from vacuum.
 * Overfull keeps its whole value — `1.04` is a fault the lab reports, and the
 * digit that says how far past capacity the cell is, is the point of it. An
 * excess under half a percent still prints `1.00`, which two decimals cannot
 * separate from full; that is what the wash is for, and why the wash and this
 * readout now branch on the same {@link fractionBand}.
 */
export function fractionReadout(fill: number): string {
  if (fractionBand(fill) === "overfull") return fill.toFixed(2);
  if (fill >= 0.995) return "1";
  if (fill >= 0.005) return fill.toFixed(2).slice(1);
  return `1e${Math.round(Math.log10(Math.max(fill, FRACTION_FLOOR)))}`;
}

/** One band of the fraction view, as every surface that draws it reads it. */
export interface FractionBandPaint {
  readonly band: FractionBand;
  readonly tone: FractionTone;
  /**
   * The band's colour as the pixels it becomes — display space, `#rrggbb`.
   *
   * A legend can use this directly and the shader converts it once; see
   * {@link fractionViewShaderConstants} for why these are display values
   * rather than linear ones.
   */
  readonly swatch: `#${string}`;
  /** The band's name, for a legend line or a probe key. */
  readonly label: string;
  /** What being in this band says about the cell. One line, for a probe. */
  readonly note: string;
}

/**
 * The colours, in display space.
 *
 * `empty`, `liquid` and `excess` are the three literals the dense-grid overlay
 * was carrying inline, rewritten as the hex they were rounded from: the shader
 * multiplies them back out to within a fraction of one code value, which is
 * under what a display can show. `excess` is `#f0c252` rather than the `#ffc252`
 * its `vec3f(1.0, …)` implied because `sceneColor` clamps a channel at 0.94 on
 * the way in — 240/255 and 255/255 leave that function as the same colour, so
 * this is the value that is both bit-identical in the shader and honest in a
 * legend.
 *
 * `residue` has no counterpart in the 3-D branch, which ramps `empty → liquid`
 * straight through the dilute decades; it is the lab's residue amber, declared
 * here so the band has a colour wherever a surface wants to name it.
 */
export const FRACTION_VIEW_BANDS: readonly FractionBandPaint[] = Object.freeze([
  Object.freeze<FractionBandPaint>({
    band: "vacuum", tone: "empty", swatch: "#13100c",
    label: "V/K ≤ 10⁻⁶ — vacuum",
    note: "no liquid this view will claim to resolve; the cell keeps its grid lines",
  }),
  Object.freeze<FractionBandPaint>({
    band: "dilute", tone: "residue", swatch: "#e0aa62",
    label: "dilute residue, by decade",
    note: "under half a cell, where a cut line is a sliver too thin to see",
  }),
  Object.freeze<FractionBandPaint>({
    band: "liquid", tone: "liquid", swatch: "#539ade",
    label: "V/K — conservative cell fill",
    note: "the conserved quantity itself, read off the compact record",
  }),
  Object.freeze<FractionBandPaint>({
    band: "overfull", tone: "excess", swatch: "#f0c252",
    label: "V/K > 1 — overcapacity",
    note: "past capacity: the wash is the fault, not the water",
  }),
]);

/** One band by name. Throws rather than returning a silent default colour. */
export function fractionBandPaint(band: FractionBand): FractionBandPaint {
  const paint = FRACTION_VIEW_BANDS.find((entry) => entry.band === band);
  if (!paint) throw new RangeError(`no paint declared for fraction band ${band}`);
  return paint;
}

/**
 * A WGSL float literal for `value`.
 *
 * WGSL has no implicit int-to-float conversion in a `const` initializer typed
 * `f32`, so `1` would be an integer and `1.0` a float. Everything that leaves
 * here carries a `.` or an `e`.
 */
export function wgslFloatLiteral(value: number): string {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${value} is not a WGSL float literal`);
  }
  const text = Object.is(value, -0) ? "-0" : String(value);
  return /[.e]/i.test(text) ? text : `${text}.0`;
}

/**
 * `#rrggbb` as the `vec3f` the overlay shader wants.
 *
 * No gamma conversion happens here, and that is the whole reason this helper is
 * one line: `sceneColor()` in the overlay already takes a *display*-space
 * triple and inverts the tonemap and the transfer function itself — its
 * comment says so — so the literals it was written with are the pixels, in the
 * same space as a CSS hex. Converting here as well would apply the transfer
 * function twice.
 */
export function wgslDisplayColor(swatch: string): string {
  const hex = /^#([0-9a-f]{6})$/i.exec(swatch.trim());
  if (!hex) throw new RangeError(`${swatch} is not a #rrggbb colour`);
  const packed = Number.parseInt(hex[1]!, 16);
  const channels = [(packed >> 16) & 0xff, (packed >> 8) & 0xff, packed & 0xff];
  return `vec3f(${channels.map((c) => wgslFloatLiteral(c / 255)).join(", ")})`;
}

/**
 * The fraction view as module-scope WGSL constants.
 *
 * Interpolated into the dense-grid overlay's template so the shader reads the
 * same numbers this module hands the lab and the legend. It declares only
 * constants, so it may sit anywhere above their first use.
 */
export const fractionViewShaderConstants = /* wgsl */ `
// Generated from lib/core/fluid-fraction-view.ts. Do not retype these values
// here: the 2-D advance lab, this shader and the overlay legend are all
// reading the one definition, and a literal typed back in is a fourth copy.
const FRACTION_FLOOR: f32 = ${wgslFloatLiteral(FRACTION_FLOOR)};
const FRACTION_LIQUID_KNEE: f32 = ${wgslFloatLiteral(FRACTION_LIQUID_KNEE)};
const FRACTION_OVERFULL: f32 = ${wgslFloatLiteral(FRACTION_OVERFULL)};
// Display-space, for sceneColor(): see wgslDisplayColor's comment.
const FRACTION_EMPTY_DISPLAY: vec3f = ${wgslDisplayColor(fractionBandPaint("vacuum").swatch)};
const FRACTION_LIQUID_DISPLAY: vec3f = ${wgslDisplayColor(fractionBandPaint("liquid").swatch)};
const FRACTION_EXCESS_DISPLAY: vec3f = ${wgslDisplayColor(fractionBandPaint("overfull").swatch)};
`;
