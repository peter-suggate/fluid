import {
  aabb,
  C,
  V,
  type EnvironmentLinearColor,
  type EnvironmentSceneryModule,
} from "./builder";

/**
 * The white cyclorama measurement studio: the environment every numeric preset
 * reads its numbers against. An infinity curve sweeps up behind the tank, a
 * datum grid is inlaid in the floor, a low step wedge calibrates value, and one
 * overhead softbox does the lighting. Nothing else. The whole room is authored
 * as near-neutral greys so shading, occlusion and falloff carry the form, and
 * every element is pushed out past |x| or |z| = 0.68 s so the container volume,
 * and the space directly above it, stay empty for the water.
 */

/** Front faces of both cyclorama panels, in units of s. Measured to fill the frame at 4:3 through 21:9. */
const CYC_FACE = 1.55;
/** Half extent of each panel along its long axis; the shell plate is widened to carry it. */
const CYC_HALF = 2.5;
const CYC_HEIGHT = 2.4;
/** Cove radius. Tangent to the floor at 1.13 s and to the panel face at 0.42 s up, so the junction has no seam. */
const COVE = .42;

/** Datum lines, in units of s across their short axis. The -1.4 s pair is omitted: the cove swallows it. */
const DATUM_OFFSETS = [-.7, .7, 1.4] as const;
/** Neutral step wedge, dark to light. The only high-contrast element, and it is 0.1 s tall at the back-left cove base. */
const CALIBRATION_STEPS = [.05, .22, .52, .90] as const;

const CYC_WHITE = C(.80, .805, .81);
const DATUM_GREY = C(.44, .445, .45);

export const defaultStudioScenery: EnvironmentSceneryModule = {
  id: "default",
  build: (b, context) => {
    const { roomHalf_m: roomHalf, shellThickness_m: thickness, s } = context;
    const t = thickness * .5;
    const floorY = -.012;
    // The plate is widened where the preset's own room would be tighter than the
    // cyclorama, so the sweep always stands on floor rather than over its edge.
    const plateHalf = V(Math.max(roomHalf.x, 2.7 * s), t, Math.max(roomHalf.z, 2.7 * s));
    b.box("shell/floor", "shell-floor", V(0, floorY - t, 0), plateHalf, C(.62, .625, .625), 0, ["shell", "floor"], true);

    // Infinity curve. Two panels and their coves meet in a filleted corner, so
    // the background behind the tank has no horizon line and no visible join.
    const face = CYC_FACE * s, half = CYC_HALF * s, panel = .05 * s, mid = floorY + .5 * CYC_HEIGHT * s;
    b.box("cyc/back-panel", "cyclorama", V(0, mid, -face - panel), V(half, .5 * CYC_HEIGHT * s, panel), CYC_WHITE, 0, ["cyclorama"]);
    b.ellipsoid("cyc/back-cove", "cyclorama", V(0, floorY, -face), V(half, COVE * s, COVE * s), CYC_WHITE, 0, ["cyclorama", "cove"]);
    b.box("cyc/left-panel", "cyclorama", V(-face - panel, mid, 0), V(panel, .5 * CYC_HEIGHT * s, half), CYC_WHITE, 0, ["cyclorama"]);
    b.ellipsoid("cyc/left-cove", "cyclorama", V(-face, floorY, 0), V(COVE * s, COVE * s, half), CYC_WHITE, 0, ["cyclorama", "cove"]);
    b.cylinder("cyc/corner-fillet", "cyclorama", V(-face + .3 * s, mid, -face + .3 * s), .3 * s, .5 * CYC_HEIGHT * s, CYC_WHITE, 0, ["cyclorama", "cove"]);

    // Datum grid, half sunk into the plate at 0.7 s pitch. Each line clears the
    // container on its short axis and runs off into the cove on its long one.
    const rib = V(half, .008 * s, .02 * s);
    for (let i = 0; i < DATUM_OFFSETS.length; i++) {
      b.box(`grid/datum-x-${i + 1}`, "grid-datum", V(0, floorY, DATUM_OFFSETS[i] * s), rib, DATUM_GREY, 0, ["grid", "datum"]);
      b.box(`grid/datum-z-${i + 1}`, "grid-datum", V(DATUM_OFFSETS[i] * s, floorY, 0), V(rib.z, rib.y, rib.x), DATUM_GREY, 0, ["grid", "datum"]);
    }

    // Value reference at the cove base, off to the back left of frame.
    for (let i = 0; i < CALIBRATION_STEPS.length; i++) {
      const value = CALIBRATION_STEPS[i];
      const step: EnvironmentLinearColor = C(value, value * 1.005, value * 1.01);
      b.box(`calibration/step-${i + 1}`, "calibration-step", V((-1.45 + .20 * i) * s, floorY + .05 * s, -.85 * s), V(.075 * s, .05 * s, .075 * s), step, 0, ["calibration"]);
    }

    // One overhead softbox, hung well clear of the container, plus a dim
    // groundrow that lifts the cove toe so the curve reads as infinite.
    b.box("light/softbox", "softbox", V(0, floorY + 1.72 * s, -.15 * s), V(.72 * s, .03 * s, .55 * s), C(.96, .97, .98), 1.0, ["softbox", "light", "emits-negative-y"]);
    b.box("light/groundrow", "groundrow", V(0, floorY + .025 * s, -1.06 * s), V(1.9 * s, .02 * s, .05 * s), C(.97, .965, .95), .34, ["groundrow", "light", "emits-positive-y"]);

    return {
      kind: "floor",
      floorY_m: floorY,
      bounds_m: aabb(V(0, floorY, 0), V(plateHalf.x, 0, plateHalf.z)),
      primitives: b.shell,
      materialModel: "default-floor",
    };
  },
};
