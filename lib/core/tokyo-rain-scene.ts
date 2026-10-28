import { cloneScene, defaultScene, type CameraState, type SceneDescription } from "./model";
import type { SceneryNode } from "./scenery-graph";
import { meshAssetHeader } from "./imported-assets/asset-library";
import instances from "./imported-assets/tokyo-rain-instances.json";

export const TOKYO_RAIN_ID = "tokyo-rain";
export const TOKYO_RAIN_CELL_M = 0.0625;
export const TOKYO_RAIN_WATERLINE_M = 0.38;

/** Y-up application coordinates export to Blender as (x, -z, y).
 * A flooded central gutter is bounded by raised pavement. Roof runoff and
 * resolved drops enter the same live NB-FLIP solve as the standing water. */
export function createTokyoRainScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = TOKYO_RAIN_ID;
  scene.randomSeed = 20261028;
  scene.duration_s = 24;
  scene.systems = { fluid: true };
  scene.container = { width_m: 8, height_m: 2, depth_m: 16, fillFraction: 0,
    top: "open", fluidWallMode: "free-slip", vessel: "none" };
  scene.voxelDomain = { finestCellSize_m: TOKYO_RAIN_CELL_M, brickSize_cells: 8 };
  scene.nominalResolution = { length_m: TOKYO_RAIN_CELL_M };
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 120;
  scene.fluid = {
    density_kg_m3: 998.2, dynamicViscosity_Pa_s: 0, surfaceTension_N_m: 0,
    gravity_m_s2: { x: 0, y: -9.80665, z: 0 }, initialCondition: "tank-fill",
    initialVelocity_m_s: { x: 0, y: 0, z: 0 },
    initialLiquidVolumes: [{ shape: "box", min_m: { x: -2.85, y: .15, z: -7.3 },
      max_m: { x: 2.85, y: TOKYO_RAIN_WATERLINE_M, z: 7.3 } }],
    // Roof runoff is resolved at cell scale; airborne photographic rain in the
    // Blender set is visual atmosphere and is not used as a fluid substitute.
    inflow: { center_m: { x: -2.6, y: 1.55, z: -1.8 }, radius_m: .11,
      length_m: .018, velocity_m_s: { x: .25, y: -1.8, z: 0 },
      start_s: 0, end_s: 24, ramp_s: .25 },
    scheduledDrops: Array.from({ length: 80 }, (_, i) => ({ time_s: .1 + i * .23,
      volume: { shape: "sphere" as const, radius_m: .085,
        center_m: { x: -2.2 + ((i * 37) % 89) / 89 * 4.4,
          y: .85 + (i % 4) * .13, z: -6 + ((i * 53) % 97) / 97 * 12 } } })),
    pagePool: { blocks: 100 },
    optics: { indexOfRefraction: 1.333, absorption_mInv: [.3, .16, .08], roughness: .04 },
  };
  scene.terrain = { baseHeight_m: .18, features: [], grid: {
    kind: "grid", origin_m: { x: -4, z: -8 }, spacing_m: .125,
    size: { nx: 65, nz: 129 }, heights_m: Array.from({ length: 65 * 129 }, (_, i) => {
      const x = -4 + (i % 65) * .125, z = -8 + Math.floor(i / 65) * .125;
      return Math.abs(x) > 2.85 || Math.abs(z) > 7.3 ? .46 : .18;
    }),
  } };
  scene.rigidBodies = [];
  scene.solidVoxels = [];
  delete scene.solidBoundary;
  const nodes: SceneryNode[] = [{ kind: "terrain-shell", id: "tokyo/terrain", materialModel: "porcelain" },
    { kind: "box", id: "tokyo/asphalt",
    place: { units: "metres", position: { x: 0, y: .1, z: -5 } },
    halfSize: { x: 3.5, y: .1, z: 17 }, material: { colorLinear: [.028, .035, .042], surface: "stone" } }];
  // The domain's two retaining lips join the surrounding raised roadway.
  for (const [name, z, halfDepth] of [["near", 10, 2.7], ["far", -15, 7.7]] as const) {
    nodes.push({ kind: "box", id: `tokyo/road/${name}`,
      place: { units: "metres", position: { x: 0, y: .335, z } },
      halfSize: { x: 2.8, y: .155, z: halfDepth },
      material: { colorLinear: [.028, .035, .042], surface: "stone" } });
  }
  for (const side of [-1, 1]) {
    nodes.push({ kind: "box", id: `tokyo/pavement/${side}`,
      place: { units: "metres", position: { x: side * 3.04, y: .335, z: -5 } },
      halfSize: { x: .24, y: .155, z: 17 }, material: { colorLinear: [.16, .18, .19], surface: "stone" } });
    for (let i = 0; i < 7; i++) nodes.push({ kind: "box", id: `tokyo/building/${side}/${i}`,
      place: { units: "metres", position: { x: side * 4.12, y: 5.9,
        z: 3 - i * 3.6 - (side > 0 ? 1.25 : 0) } },
      halfSize: { x: 1.05, y: 2.9, z: 1.785 },
      material: { colorLinear: i % 3 === 0 ? [.24, .12, .065] : [.16, .17, .18], surface: "stone" } });
  }
  nodes.push({ kind: "box", id: "tokyo/end-street-building",
    place: { units: "metres", position: { x: 0, y: 5, z: -24 } },
    halfSize: { x: 7, y: 5, z: 1 }, material: { colorLinear: [.15, .16, .18], surface: "stone" } });
  for (const x of [-4, -1.8, .5, 2.8, 5]) for (const y of [2, 4.5, 7]) {
    nodes.push({ kind: "box", id: `tokyo/distant-window/${x}/${y}`,
      place: { units: "metres", position: { x, y, z: -22.97 } },
      halfSize: { x: .6, y: .725, z: .025 },
      material: { colorLinear: x === .5 && y === 2 ? [.36, .12, .025] : [.025, .04, .065],
        emission: x === .5 && y === 2 ? .7 : 0, surface: "none" } });
    for (const dx of [-.63, 0, .63]) nodes.push({ kind: "box", id: `tokyo/window-jamb/${x}/${y}/${dx}`,
      place: { units: "metres", position: { x: x + dx, y, z: -22.91 } },
      halfSize: { x: .035, y: .77, z: .045 }, material: { colorLinear: [.055,.065,.075], surface: "brushed-metal" } });
    for (const dy of [-.76, .05, .76]) nodes.push({ kind: "box", id: `tokyo/window-rail/${x}/${y}/${dy}`,
      place: { units: "metres", position: { x, y: y + dy, z: -22.91 } },
      halfSize: { x: .65, y: .035, z: .045 }, material: { colorLinear: [.055,.065,.075], surface: "brushed-metal" } });
  }
  // Blender-authored architecture and props use the native sampled-volume
  // loader. The only animated liquid remains the production NB-FLIP solver.
  for (const instance of instances) {
    const header = meshAssetHeader(instance.asset);
    const center = header.origin.map((value, axis) => value + header.dimensions[axis]! * header.cellSize / 2);
    const fallback = header.materials[0]!;
    // An authored fixture color is independent of the voxel palette order;
    // its first entry may be an opaque housing or a wooden lantern rib.
    const colorLinear = "emissionColorLinear" in instance
      ? instance.emissionColorLinear as [number, number, number] : fallback.baseColorLinear;
    const pointFixture = instance.id.includes("/lantern/") || instance.id.includes("/blade/");
    const emission = "emission" in instance ? (instance.id.includes("/blade/") ? 8 : instance.emission) : 0;
    nodes.push({ kind: "field-program", id: instance.id,
      tags: ["blender-tokyo", "static-environment", ...(pointFixture ? ["light", "point-light"] : [])],
      place: { units: "metres", position: { x: instance.position[0]! + center[0]!,
        y: instance.position[1]! + center[1]!, z: instance.position[2]! + center[2]! } },
      program: { ops: [], result: 0, sampledVolume: { asset: instance.asset } },
      material: { colorLinear, surface: "stone",
        emission },
    });
  }
  // Wall-mounted practicals match the source set. Small apertures sit in
  // opaque housings; the warm shop light models its actual front-side spill
  // across the rider rather than relying solely on distant backlighting.
  for (const lamp of [
    { id: "far-blue", p: [2.55,4,-18], target: [0,1.4,-3.5], color: [.22,.5,1], power: 180, wall: 3.04 },
    { id: "ramen-warm", p: [-2.4,2.3,0], target: [.35,1.4,-3.5], color: [1,.34,.11], power: 80, wall: -2.96 },
  ]) {
    const d = lamp.target.map((v,i) => v - lamp.p[i]!);
    const length = Math.hypot(...d); const [dx,dy,dz] = d.map(v => v / length) as [number,number,number];
    const qLength = Math.hypot(-dz,dx,1-dy);
    const orientation = { w: (1-dy)/qLength, x: -dz/qLength, y: 0, z: dx/qLength };
    const [x,y,z] = lamp.p as [number,number,number];
    nodes.push({ kind: "cone", id: `tokyo/${lamp.id}-practical`, tags: ["light", "spot-light", "fixture"],
      place: { units: "metres", position: { x,y,z }, orientation },
      baseRadius: .09, topRadius: .02, halfHeight: .014,
      material: { colorLinear: lamp.color as [number,number,number], emission: lamp.power, surface: "none" } });
    nodes.push({ kind: "box", id: `tokyo/${lamp.id}-housing`,
      place: { units: "metres", position: { x: x-dx*.09, y: y-dy*.09, z: z-dz*.09 }, orientation },
      halfSize: { x: .14, y: .07, z: .14 }, material: { colorLinear: [.025,.03,.035], surface: "brushed-metal" } });
    nodes.push({ kind: "box", id: `tokyo/${lamp.id}-mount`,
      place: { units: "metres", position: { x: (x+lamp.wall)/2, y: y+.12, z } },
      halfSize: { x: Math.abs(x-lamp.wall)/2, y: .025, z: .025 }, material: { colorLinear: [.035,.04,.045], surface: "brushed-metal" } });
  }
  scene.scenery = { palettes: {}, nodes };
  scene.surfaceStyle = "smooth";
  scene.lighting = {
    directional: { direction: [-.4, .8, .2], colorLinear: [.25, .4, .7], intensity: .4 },
    environment: { diffuseScale: 1, specularScale: 1, lowerRadianceLinear: [.003, .005, .009], upperRadianceLinear: [.006, .01, .016], accentRadianceLinear: [.012, .003, .009] },
    grade: { toneCurve: "agx", exposure: .8, whiteBalance: [1, 1, 1] },
  };
  return scene;
}

export const tokyoRainCamera: Partial<CameraState> = {
  azimuth_rad: Math.atan2(.8, 16.2), elevation_rad: Math.atan2(-1.25, Math.hypot(.8, 16.2)),
  distance_m: Math.hypot(.8, 1.25, 16.2),
  tanHalfFov: .32, target_m: { x: -.05, y: 2.6, z: -5.8 },
};
