/**
 * A light's colour, in the vocabulary lighting tools already use.
 *
 * Every lighting UI a reader is likely to have met — a DCC package, a game
 * engine, a photo editor, a smart bulb — offers colour two ways: as a
 * *temperature* on a warm-to-cold scale, because that is how real sources
 * differ, and as a free pick for a gel or a screen. This module is both halves
 * as plain numbers, so an entity can declare a colour field without knowing how
 * it is drawn.
 *
 * Colours here are scene-linear chromaticities normalised so their brightest
 * channel is 1. Brightness is a light's *strength*, a separate number, and a
 * colour that also carried brightness would make the two controls fight: picking
 * a darker swatch would dim the light behind the strength readout's back.
 */

export type LinearRgb = readonly [number, number, number];

export const LIGHT_TEMPERATURE_MIN_K = 1500;
export const LIGHT_TEMPERATURE_MAX_K = 12000;

/** Named sources, warm to cold — the stops a lighting desk's presets carry. */
export const LIGHT_TEMPERATURE_PRESETS: ReadonlyArray<{ readonly id: string; readonly label: string; readonly kelvin: number }> = Object.freeze([
  { id: "candle", label: "Candle", kelvin: 1900 },
  { id: "tungsten", label: "Tungsten", kelvin: 2700 },
  { id: "halogen", label: "Halogen", kelvin: 3400 },
  { id: "daylight", label: "Daylight", kelvin: 5600 },
  { id: "overcast", label: "Overcast", kelvin: 7000 },
  { id: "sky", label: "Blue sky", kelvin: 10000 },
]);

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

function srgbToLinear(value: number): number {
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}

function linearToSrgb(value: number): number {
  return value <= 0.0031308 ? 12.92 * value : 1.055 * value ** (1 / 2.4) - 0.055;
}

/** Brightest channel to 1; black stays black rather than dividing by zero. */
export function lightChromaticity(color: LinearRgb): LinearRgb {
  const peak = Math.max(color[0], color[1], color[2]);
  return peak > 1e-9 ? [color[0] / peak, color[1] / peak, color[2] / peak] : [1, 1, 1];
}

/**
 * The colour of a blackbody at `kelvin`, as a linear chromaticity.
 *
 * Tanner Helland's fit to the CIE blackbody locus, in display sRGB, then taken
 * to linear. It is a fit rather than a Planck integral, which is what every
 * tool that offers a Kelvin slider uses: accurate to a few percent over the
 * range a lamp occupies, and monotone, so the slider never reverses.
 */
export function kelvinToLinear(kelvin: number): LinearRgb {
  const t = Math.min(LIGHT_TEMPERATURE_MAX_K, Math.max(LIGHT_TEMPERATURE_MIN_K, kelvin)) / 100;
  const red = t <= 66 ? 255 : 329.698727446 * (t - 60) ** -0.1332047592;
  const green = t <= 66
    ? 99.4708025861 * Math.log(t) - 161.1195681661
    : 288.1221695283 * (t - 60) ** -0.0755148492;
  const blue = t >= 66 ? 255 : t <= 19 ? 0 : 138.5177312231 * Math.log(t - 10) - 305.0447927307;
  return lightChromaticity([red, green, blue].map((channel) => srgbToLinear(clamp01(channel / 255))) as unknown as LinearRgb);
}

/**
 * The temperature whose colour this is, when it is one.
 *
 * Undefined for a colour off the blackbody locus — a green gel, a magenta
 * screen — because naming the nearest Kelvin for those would claim a warmth the
 * light does not have. The tolerance is loose enough to recognise every colour
 * the slider itself writes, and every warm-white a preset scene authored by
 * hand, and tight enough to refuse a tint.
 */
export function linearToKelvin(color: LinearRgb): number | undefined {
  const target = lightChromaticity(color);
  let best: { kelvin: number; error: number } | undefined;
  for (let kelvin = LIGHT_TEMPERATURE_MIN_K; kelvin <= LIGHT_TEMPERATURE_MAX_K; kelvin += 50) {
    const candidate = kelvinToLinear(kelvin);
    const error = Math.max(...candidate.map((channel, index) => Math.abs(channel - target[index]!)));
    if (!best || error < best.error) best = { kelvin, error };
  }
  return best && best.error < 0.06 ? best.kelvin : undefined;
}

/** `#rrggbb` in display sRGB, for a swatch and the native colour picker. */
export function linearToHex(color: LinearRgb): string {
  return `#${lightChromaticity(color)
    .map((channel) => Math.round(clamp01(linearToSrgb(channel)) * 255).toString(16).padStart(2, "0"))
    .join("")}`;
}

export function hexToLinear(hex: string): LinearRgb | undefined {
  const match = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!match) return undefined;
  const value = Number.parseInt(match[1]!, 16);
  return lightChromaticity([16, 8, 0].map((shift) => srgbToLinear(((value >> shift) & 0xff) / 255)) as unknown as LinearRgb);
}

/** The readout beside a swatch: the temperature when there is one, else the hex. */
export function describeLightColor(color: LinearRgb): string {
  const kelvin = linearToKelvin(color);
  return kelvin === undefined ? linearToHex(color) : `${kelvin} K`;
}
