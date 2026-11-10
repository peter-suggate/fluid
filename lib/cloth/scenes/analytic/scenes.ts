/**
 * The analytic cloth scenes as data: each a cloth document (or several), the
 * voxel solids it rests on, and the closed-form answer it is gated against.
 * Scene coordinates throughout: the container is centred on x and z with its
 * floor at y = 0, and the container's floor (its frictionless shell) is the
 * ground every scene lands on.
 */
import type { CameraState, Vec3 } from "../../../core/model";
import type { ClothDocument, ClothMaterial, ClothNewtonSettings, ClothTopologyData, ClothVector } from "../../types";
import { herringBoneSheet } from "../bridson-2002/mesh";
import { CLOTH_ANALYTIC_STEP_S, CURTAIN, FREE_FALL, GRAVITY, INCLINE, PEIRCE, SHEET, SWING } from "./parameters";
import { bdf2StartOffset_m, bendingLength_m, heavyElastica, inclineAcceleration_m_s2, peirceChordAngle_deg } from "./references";

/** A static voxel solid, as an exact signed distance in scene metres (1-Lipschitz, exact zero set). */
export type ClothAnalyticSolidShape =
  /** Axis-aligned box. */
  | { readonly kind: "box"; readonly lo_m: Vec3; readonly hi_m: Vec3 }
  /**
   * A wedge standing on the floor: x in [x0, x1], |z - zc| <= halfWidth, y in [0, top(x)], with its top a slope
   * falling along +x at `angle_deg` from `top_m` at x0.
   */
  | { readonly kind: "wedge"; readonly x0_m: number; readonly x1_m: number; readonly zc_m: number; readonly halfWidth_m: number; readonly top_m: number; readonly angle_deg: number };

export interface ClothAnalyticSolid {
  readonly id: string;
  readonly shape: ClothAnalyticSolidShape;
  /** Coulomb coefficient against cloth (one value for static and kinetic). */
  readonly friction: number;
}

export type ClothAnalyticKind = "free-fall" | "hanging-curtain" | "swinging-sheet" | "peirce-cantilever" | "friction-incline";

export interface ClothAnalyticScene {
  readonly id: string;
  readonly kind: ClothAnalyticKind;
  readonly name: string;
  /** The expected analytic result, one line, for the library card. */
  readonly expectation: string;
  /** What is measured and how close it must be, for the card's status and the tests. */
  readonly gate: string;
  readonly duration_s: number;
  readonly step_s: number;
  readonly cell_m: number;
  /** Container extents (each a whole number of 8-cell bricks). */
  readonly extents_m: Vec3;
  readonly cloth: readonly ClothDocument[];
  readonly solids: readonly ClothAnalyticSolid[];
  readonly camera: Partial<CameraState>;
  /** Kind-specific measurement data the tests read (vertex sets, references). */
  readonly probe: Readonly<Record<string, number | readonly number[]>>;
}

const g = GRAVITY.value;
const linear = (srgb: number) => srgb <= 0.04045 ? srgb / 12.92 : ((srgb + 0.055) / 1.055) ** 2.4;
/** A display colour given as sRGB hex, stored linear (the cloth surface writes linear HDR). */
const colour = (hex: number): ClothVector => [linear(((hex >> 16) & 255) / 255), linear(((hex >> 8) & 255) / 255), linear((hex & 255) / 255)];

function sheetMaterial(id: string, hex: number, overrides: Partial<ClothMaterial> = {}): ClothMaterial {
  return {
    id, arealDensity_kg_m2: SHEET.arealDensity.value, stretchWarp_N_m: SHEET.stretch.value, stretchWeft_N_m: SHEET.stretch.value,
    shear_N_m: SHEET.shear.value, poisson: SHEET.poisson.value, compressionRatio: SHEET.compressionRatio.value, bending_N_m: SHEET.bending.value,
    damping_s: SHEET.damping.value, thickness_m: SHEET.thickness.value, strainLimit: SHEET.strainLimit.value, baseColor: colour(hex), ...overrides,
  };
}

/** `planeRecords`: the scene's interim plane-record capacity per vertex (see parameters.ts), else the default. */
function document(id: string, data: ClothTopologyData, material: ClothMaterial, planeRecords?: number): ClothDocument {
  return { schemaVersion: "2.0.0", id, geometry: { kind: "inline", data }, materials: [material],
    solver: { integrator: "bdf2", newton: clothAnalyticNewton(), ...(planeRecords ? { contactPlaneRecordsPerVertex: planeRecords } : {}) } };
}

/** The Newton-PCG settings every analytic document runs (the Bridson scenes'). */
export function clothAnalyticNewton(): ClothNewtonSettings {
  return { iterations: SHEET.newtonIterations.value, pcgIterations: SHEET.newtonPcgIterations.value,
    pcgTolerance: SHEET.newtonPcgTolerance.value, residualTolerance: SHEET.newtonResidualTolerance.value };
}

/** Map a flat herring-bone sheet's (x, y0, z) rest positions through `place`. */
function placed(data: ClothTopologyData, place: (x: number, z: number) => readonly [number, number, number]): ClothTopologyData {
  const rest: number[] = [];
  for (let v = 0; v < data.restPositions_m.length; v += 3) rest.push(...place(data.restPositions_m[v]!, data.restPositions_m[v + 2]!));
  return { ...data, restPositions_m: rest };
}

const nodes = (length_m: number, cell_m: number) => Math.round(length_m / cell_m) + 1;
const mm = (m: number) => (1000 * m).toFixed(2);

/** Scene 1. */
export function clothAnalyticFreeFall(): ClothAnalyticScene {
  const side = FREE_FALL.side.value, n = nodes(side, SHEET.cell.value), y0 = FREE_FALL.height.value;
  const data = herringBoneSheet({ columns: n, rows: n, width_m: side, depth_m: side, origin_m: [-side / 2, y0, -side / 2] });
  const offset = -bdf2StartOffset_m(CLOTH_ANALYTIC_STEP_S, -g);
  return {
    id: "cloth-analytic-free-fall", kind: "free-fall", name: "Cloth analytic · Free fall",
    expectation: `An unpinned flat sheet falls as y = y₀ − g t²/2, flat and unstrained; BDF2 is exact for constant gravity after its implicit Euler start, which leaves a fixed ${mm(offset)} mm (3/4 g Δt²).`,
    gate: `centroid within 0.1 mm of the BE + BDF2 recurrence and its velocity within 1 mm/s of g t over the first ${FREE_FALL.gateSteps.value} steps; every vertex within 0.1 mm of the centroid plane; edge strain under 1e-4`,
    duration_s: FREE_FALL.duration.value, step_s: CLOTH_ANALYTIC_STEP_S, cell_m: 0.02, extents_m: { x: 1.6, y: 1.6, z: 1.6 },
    cloth: [document("cloth-analytic-free-fall-sheet", data, sheetMaterial("free-fall-teal", 0x12a6a0), FREE_FALL.planeRecords.value)],
    solids: [],
    camera: { azimuth_rad: 0.62, elevation_rad: 0.22, distance_m: 3.1, target_m: { x: 0, y: 0.72, z: 0 } },
    probe: { y0, gateSteps: FREE_FALL.gateSteps.value },
  };
}

/** Scene 2. */
export function clothAnalyticHangingCurtain(): ClothAnalyticScene {
  const m = CURTAIN.nodesAcross.value, h = CURTAIN.width.value / (m - 1), n = Math.floor(CURTAIN.aspect.value * (m - 1) + 1e-9) + 1, top = CURTAIN.top.value;
  const flat = herringBoneSheet({ columns: m, rows: n, width_m: CURTAIN.width.value, depth_m: (n - 1) * h, origin_m: [0, 0, 0], group: (_i, j) => j === 0 ? 1 : 0 });
  const data = placed(flat, (x, z) => [x - CURTAIN.width.value / 2, top - z, 0]);
  const k = SHEET.stretch.value, rho = SHEET.arealDensity.value, L = (n - 1) * h;
  return {
    id: "cloth-analytic-hanging-curtain", kind: "hanging-curtain", name: "Cloth analytic · Hanging curtain",
    expectation: `Pinned along its top edge, the sheet comes to rest with strain ρg(L − s)/E at depth s: ${(100 * rho * g * L / k).toFixed(1)} % at the rod, 0 at the hem (uniaxial StVK, exact per band).`,
    gate: `averaged over the second before ${CURTAIN.settle.value} s (the converged sheet still sways about its rest state): every interior band's mean strain within 1 % of its exact equilibrium, the hem's drop within 1 %`,
    duration_s: CURTAIN.duration.value, step_s: CLOTH_ANALYTIC_STEP_S, cell_m: 0.02, extents_m: { x: 1.6, y: 2.4, z: 0.96 },
    cloth: [document("cloth-analytic-hanging-curtain-sheet", data, sheetMaterial("curtain-crimson", 0xb3132e))],
    solids: [],
    camera: { azimuth_rad: 0.42, elevation_rad: 0.1, distance_m: 3.6, target_m: { x: 0, y: 1.12, z: 0 } },
    probe: { columns: m, rows: n, cell: h, top, settle: CURTAIN.settle.value },
  };
}

/** Scene 3. */
export function clothAnalyticSwingingSheet(): ClothAnalyticScene {
  const L = SWING.length.value, W = SWING.width.value, cols = nodes(L, SHEET.cell.value), rows = nodes(W, SHEET.cell.value), y = SWING.height.value;
  const data = herringBoneSheet({ columns: cols, rows, width_m: L, depth_m: W, origin_m: [0, y, -W / 2], group: i => i === 0 ? 1 : 0 });
  const freeEdge: number[] = [];
  // The free edge away from its two corners (the middle 60 %), where the sheet is a flat strip.
  for (let j = Math.round(0.2 * (rows - 1)); j <= Math.round(0.8 * (rows - 1)); j++) freeEdge.push(cols - 1 + cols * j);
  return {
    id: "cloth-analytic-swinging-sheet", kind: "swinging-sheet", name: "Cloth analytic · Swinging sheet",
    expectation: "Pinned along one edge and released level, the sheet's still-flat outer part carries only in-plane tension, so its free edge falls exactly as y₀ − g t²/2 until the swing reaches it; then it whips down under the pin.",
    gate: `free edge within 1 mm of the BDF2 free-fall recurrence over the first ${SWING.gateSteps.value} steps (exact while flat); the time it passes under the pin within one step of the same scene at ${SWING.convergedFactor.value}K`,
    duration_s: SWING.duration.value, step_s: CLOTH_ANALYTIC_STEP_S, cell_m: 0.02, extents_m: { x: 2.4, y: 1.6, z: 1.6 },
    cloth: [document("cloth-analytic-swinging-sheet-sheet", data, sheetMaterial("swing-amber", 0xf29c14), SWING.planeRecords.value)],
    solids: [],
    camera: { azimuth_rad: 0.3, elevation_rad: 0.12, distance_m: 3.9, target_m: { x: 0, y: 0.78, z: 0 } },
    probe: { y0: y, columns: cols, rows, freeEdge, gateSteps: SWING.gateSteps.value },
  };
}

/** Scene 4. */
export function clothAnalyticPeirceCantilever(): ClothAnalyticScene {
  const rho = PEIRCE.arealDensity.value, B = PEIRCE.bending.value, c = bendingLength_m(B, rho, g), h = PEIRCE.cell.value;
  const ledgeTop = PEIRCE.ledgeHeight.value, y = ledgeTop + PEIRCE.gap.value, clamp = PEIRCE.clamp.value, wStrip = PEIRCE.stripWidth.value;
  const hexes = [0x2f6fe0, 0x1fa65a, 0xe0562a];
  const cloth: ClothDocument[] = [], probe: Record<string, number | readonly number[]> = { bendingLength: c, ledgeTop };
  PEIRCE.ratios.forEach((ratio, s) => {
    // Whole cells: the overhang is the ratio's length rounded to the 5 mm lattice, and the reference is taken at that length.
    const cells = Math.round(ratio.value * c / h), overhang = cells * h, clampCells = Math.round(clamp / h);
    const cols = clampCells + cells + 1, rows = nodes(wStrip, h), zc = (s - 1) * 0.12;
    const data = herringBoneSheet({ columns: cols, rows, width_m: (cols - 1) * h, depth_m: wStrip, origin_m: [-clampCells * h, y, zc - wStrip / 2],
      group: i => i <= clampCells ? 1 : 0 });
    cloth.push(document(`cloth-analytic-peirce-strip-${s}`, data,
      sheetMaterial(`peirce-strip-${s}`, hexes[s]!, { arealDensity_kg_m2: rho, bending_N_m: B, stretchWarp_N_m: PEIRCE.stretch.value, stretchWeft_N_m: PEIRCE.stretch.value,
        shear_N_m: PEIRCE.shear.value, damping_s: PEIRCE.damping.value })));
    const exact = heavyElastica(overhang / c);
    probe[`strip${s}`] = [clampCells, cols, rows, overhang, exact.chordAngle_deg, peirceChordAngle_deg(overhang / c)];
  });
  const angles = PEIRCE.ratios.map((r, s) => `${(probe[`strip${s}`] as number[])[4]!.toFixed(1)}° at l/c ${r.value}`).join(", ");
  return {
    id: "cloth-analytic-peirce-cantilever", kind: "peirce-cantilever", name: "Cloth analytic · Peirce cantilever",
    expectation: `Strips overhanging a ledge droop to the heavy elastica of bending length c = (B/ρg)^⅓ = ${(1000 * c).toFixed(1)} mm: tip chord ${angles} (ASTM D1388's 41.5° at l = 2c).`,
    gate: `after ${PEIRCE.settle.value} s each strip's tip chord angle within 1° of the exact elastica (Peirce's formula reported beside it)`,
    duration_s: PEIRCE.duration.value, step_s: CLOTH_ANALYTIC_STEP_S, cell_m: 0.01, extents_m: { x: 0.96, y: 0.8, z: 0.64 },
    cloth,
    solids: [{ id: "ledge", friction: 0.5, shape: { kind: "box", lo_m: { x: -0.4, y: 0, z: -0.24 }, hi_m: { x: 0, y: ledgeTop, z: 0.24 } } }],
    camera: { azimuth_rad: 0.32, elevation_rad: 0.08, distance_m: 1.05, target_m: { x: 0.06, y: 0.43, z: 0 } },
    probe,
  };
}

/** Scene 5. */
export function clothAnalyticFrictionIncline(): ClothAnalyticScene {
  const mu = INCLINE.friction.value, L = INCLINE.slabLength.value, w = INCLINE.slabWidth.value, p = INCLINE.patch.value, h = SHEET.cell.value;
  const x0 = -0.55, foot = 0.05, solids: ClothAnalyticSolid[] = [], cloth: ClothDocument[] = [], probe: Record<string, number | readonly number[]> = {};
  const slopes = [[INCLINE.stickAngle.value, -0.25, 0x8a4fd8], [INCLINE.slideAngle.value, 0.25, 0xd8b52a]] as const;
  slopes.forEach(([angle, zc, hex], s) => {
    const t = angle * Math.PI / 180, x1 = x0 + L * Math.cos(t), top = foot + L * Math.sin(t);
    solids.push({ id: `incline-${s}`, friction: mu, shape: { kind: "wedge", x0_m: x0, x1_m: x1, zc_m: zc, halfWidth_m: w / 2, top_m: top, angle_deg: angle } });
    // The patch lies in the slope, its uphill edge 5 cm below the top edge, `gap` above the surface.
    const n = nodes(p, h), down = [Math.cos(t), -Math.sin(t)], normal = [Math.sin(t), Math.cos(t)], a0 = 0.05, gap = INCLINE.gap.value;
    const flat = herringBoneSheet({ columns: n, rows: n, width_m: p, depth_m: p, origin_m: [0, 0, 0] });
    const data = placed(flat, (u, v) => [x0 + (a0 + u) * down[0]! + gap * normal[0]!, top + (a0 + u) * down[1]! + gap * normal[1]!, zc - p / 2 + v]);
    cloth.push(document(`cloth-analytic-incline-patch-${s}`, data, sheetMaterial(`incline-patch-${s}`, hex), INCLINE.planeRecords.value));
    probe[`slope${s}`] = [angle, inclineAcceleration_m_s2(angle, mu, g), Math.cos(t), -Math.sin(t)];
  });
  const a = (probe.slope1 as number[])[1]!;
  return {
    id: "cloth-analytic-friction-incline", kind: "friction-incline", name: "Cloth analytic · Friction on an incline",
    expectation: `With μ = ${mu} (friction angle ${(Math.atan(mu) * 180 / Math.PI).toFixed(1)}°) the patch on the ${INCLINE.stickAngle.value}° slope sticks; the one on the ${INCLINE.slideAngle.value}° slope slides at a = g(sinθ − μ cosθ) = ${a.toFixed(3)} m/s².`,
    gate: `from ${INCLINE.gateStart.value} s to ${INCLINE.gateEnd.value} s: the sticking patch moves under 2 mm; the sliding patch's fitted acceleration along the slope within 5 % of g(sinθ − μ cosθ)`,
    duration_s: INCLINE.duration.value, step_s: CLOTH_ANALYTIC_STEP_S, cell_m: INCLINE.cell.value, extents_m: { x: 1.6, y: 0.96, z: 0.96 },
    cloth, solids,
    camera: { azimuth_rad: 0.55, elevation_rad: 0.28, distance_m: 2.2, target_m: { x: 0, y: 0.32, z: 0 } },
    probe,
  };
}

export const CLOTH_ANALYTIC_SCENES = [clothAnalyticFreeFall, clothAnalyticHangingCurtain, clothAnalyticSwingingSheet, clothAnalyticPeirceCantilever, clothAnalyticFrictionIncline] as const;
