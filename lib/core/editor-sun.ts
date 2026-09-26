import { add, normalize, scale, sub } from "./math";
import {
  boxCenter,
  sceneContainerBox,
  type BoxExtent,
  type EditorEntity,
  type EditorEntityContext,
  type EditorEntityDefinition,
  type EditorFieldRow,
  type EditorHandle,
} from "./editor-entity";
import type { EditorSelection } from "./editor-tools";
import type { SceneDescription, Vec3 } from "./model";
import { svoSceneLighting } from "../svo/features/lighting-visibility/svo-dry-scene-lighting";
import { lightChromaticity } from "./light-color";

/**
 * The sun, as something the editor can select.
 *
 * Every scene is lit by one directional key — `scene.lighting.directional`, laid
 * over the environment's own rig by `svoSceneLighting` — and it is the water's
 * only key too. It has no surface, so it is drawn the way lighting tools draw a
 * sun: a marker hung in the sky *in the direction the light comes from*, at a
 * fixed distance from the set, on a line back to the set's centre. Dragging the
 * marker aims the sun, which is the same gesture as aiming the hose by its tip.
 *
 * The distance is presentation only. A directional light has no position, so
 * the marker is placed where it is on screen and useful: far enough out to read
 * as "over there", near enough to stay in frame for the house cameras.
 *
 * Edits write `scene.lighting.directional` field by field, so a scene that has
 * only ever used its environment's rig keeps the rest of it (the fill, the
 * grade) when its sun is moved.
 */

export const SUN_SELECTION_ID = "sun";
export const SUN_SELECTION: EditorSelection = Object.freeze({ kind: "sun", id: SUN_SELECTION_ID });

/** The strength a slider can reach: well past the brightest authored rig (3.3). */
export const SUN_INTENSITY_MAX = 10;
/** Degrees above the horizon the sun may sink to. Below it the set is lit from underneath. */
const SUN_MINIMUM_ELEVATION_DEG = 0;

type Directional = NonNullable<NonNullable<SceneDescription["lighting"]>["directional"]>;

interface ResolvedSun {
  /** Toward the light, unit length. */
  readonly direction: Vec3;
  readonly colorLinear: readonly [number, number, number];
  readonly intensity: number;
}

export function resolvedSun(scene: SceneDescription): ResolvedSun {
  const directional = svoSceneLighting(scene).directional;
  const [x, y, z] = directional?.direction ?? [0, 1, 0];
  return {
    direction: normalize({ x, y, z }),
    colorLinear: directional?.colorLinear ?? [1, 1, 1],
    intensity: directional?.intensity ?? 1,
  };
}

/** The scene with its sun's authored fields patched, and everything else as it was. */
export function withSun(scene: SceneDescription, patch: Directional): Partial<SceneDescription> {
  return {
    lighting: {
      ...scene.lighting,
      directional: { ...scene.lighting?.directional, ...patch },
    },
  };
}

/** Where the marker hangs, and what it pivots about. */
function sunGeometry(scene: SceneDescription) {
  const container = sceneContainerBox(scene);
  const pivot_m = boxCenter(container);
  const { width_m, height_m, depth_m } = scene.container;
  const span_m = Math.max(width_m, height_m, depth_m);
  return { pivot_m, distance_m: 0.75 * span_m, radius_m: 0.04 * span_m };
}

const toDegrees = (radians: number) => radians * 180 / Math.PI;
const toRadians = (degrees: number) => degrees * Math.PI / 180;

/** Compass bearing and height of the sun, the pair every sun-position tool shows. */
export function sunAngles(direction: Vec3): { azimuth_deg: number; elevation_deg: number } {
  const unit = normalize(direction);
  const azimuth = toDegrees(Math.atan2(unit.x, unit.z));
  return {
    azimuth_deg: (azimuth + 360) % 360,
    elevation_deg: toDegrees(Math.asin(Math.max(-1, Math.min(1, unit.y)))),
  };
}

export function sunDirection(azimuth_deg: number, elevation_deg: number): readonly [number, number, number] {
  const elevation = toRadians(Math.max(SUN_MINIMUM_ELEVATION_DEG, Math.min(90, elevation_deg)));
  const azimuth = toRadians(azimuth_deg);
  const flat = Math.cos(elevation);
  return [flat * Math.sin(azimuth), Math.sin(elevation), flat * Math.cos(azimuth)];
}

/** A dragged marker position, as the direction it puts the sun in. */
function aimAt(pivot_m: Vec3, point_m: Vec3): readonly [number, number, number] | undefined {
  const offset = sub(point_m, pivot_m);
  if (!(Math.hypot(offset.x, offset.y, offset.z) > 1e-9)) return undefined;
  const { azimuth_deg, elevation_deg } = sunAngles(offset);
  return sunDirection(azimuth_deg, elevation_deg);
}

function sunEntityFor(context: EditorEntityContext): EditorEntity {
  const { scene } = context;
  const sun = resolvedSun(scene);
  const { pivot_m, distance_m, radius_m } = sunGeometry(scene);
  const marker_m = add(pivot_m, scale(sun.direction, distance_m));
  const box: BoxExtent = {
    min: { x: -radius_m, y: -radius_m, z: -radius_m },
    max: { x: radius_m, y: radius_m, z: radius_m },
  };
  const aim = (point_m: Vec3) => {
    const direction = aimAt(pivot_m, point_m);
    return direction && withSun(scene, { direction });
  };
  // The beam back to the set says which way the light travels; it is grabbable
  // along its length so a reader can aim by the line as well as by its end.
  const handles: EditorHandle[] = [
    {
      id: "aim",
      kind: "center",
      space: "world",
      label: "aim · drag the sun across the sky",
      position_m: marker_m,
      axes: ["x", "y", "z"],
      drag: aim,
    },
    {
      id: "beam",
      kind: "edge",
      space: "world",
      label: "aim · drag the sun across the sky",
      position_m: marker_m,
      segment: { from: pivot_m, to: marker_m },
      axes: ["x", "y", "z"],
      drag: aim,
    },
  ];
  const { azimuth_deg, elevation_deg } = sunAngles(sun.direction);
  const aimRow: EditorFieldRow = {
    id: "sun-aim",
    tag: "Aim",
    label: "Direction",
    hint: "Compass bearing and height above the horizon, in degrees",
  };
  return {
    selection: SUN_SELECTION,
    label: "SUN",
    tone: "prop",
    frame: { origin_m: marker_m, orientation: { w: 1, x: 0, y: 0, z: 0 } },
    box,
    sizeLabel: `${elevation_deg.toFixed(0)}° up · ${azimuth_deg.toFixed(0)}° bearing`,
    handles,
    draftSubject: "lighting",
    editLabel: () => "Aimed the sun",
    colors: [{
      id: "sun-color",
      label: "Colour",
      hint: "The sun's colour; its strength is set separately",
      value: lightChromaticity(sun.colorLinear),
      apply: (colorLinear) => withSun(scene, { colorLinear }),
    }],
    fields: [
      {
        id: "sun-intensity",
        label: "Strength",
        value: sun.intensity,
        step: 0.05,
        min: 0,
        max: SUN_INTENSITY_MAX,
        apply: (intensity) => withSun(scene, { intensity: Math.max(0, intensity) }),
      },
      {
        id: "sun-elevation",
        label: "Elevation",
        unit: "°",
        value: elevation_deg,
        step: 1,
        min: SUN_MINIMUM_ELEVATION_DEG,
        max: 90,
        row: aimRow,
        apply: (value) => withSun(scene, { direction: sunDirection(azimuth_deg, value) }),
      },
      {
        id: "sun-azimuth",
        label: "Bearing",
        unit: "°",
        value: azimuth_deg,
        step: 1,
        min: 0,
        max: 360,
        row: aimRow,
        apply: (value) => withSun(scene, { direction: sunDirection(value, elevation_deg) }),
      },
    ],
    summary: "Lights the whole set and the water. Exposure and fill belong to the environment.",
  };
}

export const sunEntity: EditorEntityDefinition = {
  kind: "sun",
  // Handles belong to the selection alone; see `surfacedEntities`.
  instances: () => [],
  find: (context, id) => id === SUN_SELECTION_ID ? sunEntityFor(context) : undefined,
  // No `pick`: the sun has no surface, so an invisible sphere in the sky would
  // steal clicks meant for nothing. The ring's Light ▸ Sun is the route in, and
  // once selected its handles are grabbable like any other entity's.
};
