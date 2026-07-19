import type { EnvironmentId } from "./environments";
import { environmentIndex } from "./environments";
import type { SceneDescription, Vec3 } from "./model";

/** Linear-light RGB, matching the constants authored in webgpu-environments.ts. */
export type EnvironmentLinearColor = readonly [number, number, number];

export interface EnvironmentProxyMaterial {
  readonly colorLinear: EnvironmentLinearColor;
  readonly emission: number;
  /** Surface parameter for the shared raster/voxel lighting model. */
  readonly roughness: number;
}

export interface EnvironmentProxyAabb {
  readonly min: Vec3;
  readonly max: Vec3;
}

interface EnvironmentProxyBase {
  /** Stable across rebuilds for the same environment preset. */
  readonly key: string;
  /** Dense, deterministic publication order across shell and prop primitives. */
  readonly ownerIndex: number;
  readonly group: string;
  readonly tags: readonly string[];
  readonly center_m: Vec3;
  readonly material: EnvironmentProxyMaterial;
  readonly aabb_m: EnvironmentProxyAabb;
}

export interface EnvironmentBoxProxy extends EnvironmentProxyBase {
  readonly kind: "box";
  readonly halfSize_m: Vec3;
}

export interface EnvironmentCylinderProxy extends EnvironmentProxyBase {
  readonly kind: "cylinder";
  readonly radius_m: number;
  readonly halfHeight_m: number;
  readonly axis: "y";
}

export interface EnvironmentEllipsoidProxy extends EnvironmentProxyBase {
  readonly kind: "ellipsoid";
  readonly radius_m: Vec3;
}

export type EnvironmentProxyPrimitive = EnvironmentBoxProxy | EnvironmentCylinderProxy | EnvironmentEllipsoidProxy;

export interface EnvironmentProxyShell {
  readonly kind: "room" | "floor" | "terrain-heightfield";
  readonly floorY_m: number;
  readonly bounds_m: EnvironmentProxyAabb;
  /** Shell faces suitable for voxel/debug publication. Garden terrain has none: its real heightfield remains authoritative. */
  readonly primitives: readonly EnvironmentBoxProxy[];
  readonly materialModel: "conservatory" | "courtyard" | "night-lab" | "gallery" | "bathhouse" | "station" | "default-floor" | "garden-terrain";
}

export interface EnvironmentProxyCatalog {
  readonly environmentId: EnvironmentId;
  readonly environmentIndex: number;
  readonly scale_m: number;
  readonly floorY_m: number;
  readonly shell: EnvironmentProxyShell;
  /** Authored props only; use environmentProxyPrimitives() when shell faces are also wanted. */
  readonly primitives: readonly EnvironmentProxyPrimitive[];
}

export interface EnvironmentProxyCatalogOptions {
  /** Physical thickness of the finite room shell faces. Defaults to the scene nominal resolution. */
  readonly shellThickness_m?: number;
}

const V = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
const C = (r: number, g: number, b: number): EnvironmentLinearColor => [r, g, b];
const cmul = (c: EnvironmentLinearColor, n: number): EnvironmentLinearColor => [c[0] * n, c[1] * n, c[2] * n];

function aabb(center: Vec3, radius: Vec3): EnvironmentProxyAabb {
  return {
    min: V(center.x - radius.x, center.y - radius.y, center.z - radius.z),
    max: V(center.x + radius.x, center.y + radius.y, center.z + radius.z)
  };
}

function roughnessFor(group: string, emission: number): number {
  if (emission > 0.2) return 0.28;
  if (/leaf|hedge|flower|fruit/.test(group)) return 0.86;
  if (/wood|cedar|bench|stool|bucket|tree/.test(group)) return 0.62;
  if (/stone|column|plinth|pot/.test(group)) return 0.78;
  if (/monitor|glass/.test(group)) return 0.18;
  if (/steel|metal|pipe|frame|fixture|instrument|console/.test(group)) return 0.34;
  return 0.52;
}

class ProxyBuilder {
  readonly props: EnvironmentProxyPrimitive[] = [];
  readonly shell: EnvironmentBoxProxy[] = [];
  private nextOwner = 0;

  constructor(private readonly environmentId: EnvironmentId) {}

  box(key: string, group: string, center_m: Vec3, halfSize_m: Vec3, colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = [], shell = false): EnvironmentBoxProxy {
    const proxy: EnvironmentBoxProxy = {
      kind: "box", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, halfSize_m, material: { colorLinear, emission, roughness: roughnessFor(group, emission) }, aabb_m: aabb(center_m, halfSize_m)
    };
    (shell ? this.shell : this.props).push(proxy);
    return proxy;
  }

  cylinder(key: string, group: string, center_m: Vec3, radius_m: number, halfHeight_m: number, colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = []): EnvironmentCylinderProxy {
    const radius = V(radius_m, halfHeight_m, radius_m);
    const proxy: EnvironmentCylinderProxy = {
      kind: "cylinder", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, radius_m, halfHeight_m, axis: "y",
      material: { colorLinear, emission, roughness: roughnessFor(group, emission) }, aabb_m: aabb(center_m, radius)
    };
    this.props.push(proxy);
    return proxy;
  }

  ellipsoid(key: string, group: string, center_m: Vec3, radius_m: Vec3, colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = []): EnvironmentEllipsoidProxy {
    const proxy: EnvironmentEllipsoidProxy = {
      kind: "ellipsoid", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, radius_m, material: { colorLinear, emission, roughness: roughnessFor(group, emission) }, aabb_m: aabb(center_m, radius_m)
    };
    this.props.push(proxy);
    return proxy;
  }
}

const roomMaterialModels: Record<Exclude<EnvironmentId, "default" | "garden">, EnvironmentProxyShell["materialModel"]> = {
  conservatory: "conservatory", courtyard: "courtyard", "night-lab": "night-lab", "concrete-gallery": "gallery",
  bathhouse: "bathhouse", "research-station": "station"
};

const roomRepresentativeColors: Record<Exclude<EnvironmentId, "default" | "garden">, { floor: EnvironmentLinearColor; wall: EnvironmentLinearColor; ceiling: EnvironmentLinearColor }> = {
  conservatory: { floor: C(.46, .43, .32), wall: C(.24, .42, .18), ceiling: C(.15, .24, .18) },
  courtyard: { floor: C(.55, .26, .16), wall: C(.72, .65, .52), ceiling: C(.55, .49, .38) },
  "night-lab": { floor: C(.285, .295, .310), wall: C(.56, .54, .51), ceiling: C(.46, .46, .45) },
  "concrete-gallery": { floor: C(.105, .115, .11), wall: C(.29, .30, .28), ceiling: C(.29, .30, .28) },
  bathhouse: { floor: C(.21, .215, .19), wall: C(.68, .59, .43), ceiling: C(.23, .18, .13) },
  "research-station": { floor: C(.012, .045, .065), wall: C(.012, .045, .065), ceiling: C(.009, .032, .047) }
};

function addRoomShell(builder: ProxyBuilder, id: Exclude<EnvironmentId, "default" | "garden">, roomHalf: Vec3, floorY: number, thickness: number, scale_m: number): EnvironmentProxyShell {
  const center = V(0, floorY + roomHalf.y, 0);
  const c = roomRepresentativeColors[id];
  const t = thickness * .5;
  builder.box("shell/floor", "shell-floor", V(0, floorY - t, 0), V(roomHalf.x, t, roomHalf.z), c.floor, 0, ["shell", "floor"], true);
  builder.box("shell/ceiling", "shell-ceiling", V(0, floorY + 2 * roomHalf.y + t, 0), V(roomHalf.x, t, roomHalf.z), c.ceiling, 0, ["shell", "ceiling"], true);
  builder.box("shell/wall-left", "shell-wall", V(-roomHalf.x - t, center.y, 0), V(t, roomHalf.y, roomHalf.z), c.wall, 0, ["shell", "wall"], true);
  builder.box("shell/wall-right", "shell-wall", V(roomHalf.x + t, center.y, 0), V(t, roomHalf.y, roomHalf.z), c.wall, 0, ["shell", "wall"], true);
  if (id === "night-lab") {
    // The raster room shader has a city window in its back wall. A single
    // union-only wall box concealed the authored thin-glass pane and forced
    // the whole night-lab SVO path to raster. Four ordinary analytic boxes
    // preserve the same wall while leaving an exact rectangular opening.
    const openingHalfWidth = Math.min(1.62 * scale_m, roomHalf.x - thickness);
    const openingHalfHeight = Math.min(.55 * scale_m, roomHalf.y - thickness);
    const openingCenterY = floorY + 1.60 * scale_m;
    const openingBottom = Math.max(floorY, openingCenterY - openingHalfHeight);
    const openingTop = Math.min(floorY + 2 * roomHalf.y, openingCenterY + openingHalfHeight);
    const sideHalfWidth = .5 * (roomHalf.x - openingHalfWidth);
    const sideCenterX = openingHalfWidth + sideHalfWidth;
    const bottomHalfHeight = .5 * (openingBottom - floorY);
    const topHalfHeight = .5 * (floorY + 2 * roomHalf.y - openingTop);
    builder.box("shell/wall-back-left", "shell-wall", V(-sideCenterX, center.y, -roomHalf.z - t), V(sideHalfWidth, roomHalf.y, t), c.wall, 0, ["shell", "wall", "window-cutout"], true);
    builder.box("shell/wall-back-right", "shell-wall", V(sideCenterX, center.y, -roomHalf.z - t), V(sideHalfWidth, roomHalf.y, t), c.wall, 0, ["shell", "wall", "window-cutout"], true);
    builder.box("shell/wall-back-bottom", "shell-wall", V(0, floorY + bottomHalfHeight, -roomHalf.z - t), V(openingHalfWidth, bottomHalfHeight, t), c.wall, 0, ["shell", "wall", "window-cutout"], true);
    builder.box("shell/wall-back-top", "shell-wall", V(0, openingTop + topHalfHeight, -roomHalf.z - t), V(openingHalfWidth, topHalfHeight, t), c.wall, 0, ["shell", "wall", "window-cutout"], true);
  } else {
    builder.box("shell/wall-back", "shell-wall", V(0, center.y, -roomHalf.z - t), V(roomHalf.x, roomHalf.y, t), c.wall, 0, ["shell", "wall"], true);
  }
  builder.box("shell/wall-front", "shell-wall", V(0, center.y, roomHalf.z + t), V(roomHalf.x, roomHalf.y, t), c.wall, 0, ["shell", "wall"], true);
  return { kind: "room", floorY_m: floorY, bounds_m: aabb(center, roomHalf), primitives: builder.shell, materialModel: roomMaterialModels[id] };
}

function buildConservatory(b: ProxyBuilder, s: number): void {
  const frame = C(.18, .28, .21);
  for (let i = -1; i <= 1; i++) b.box(`glazing/frame-${i + 1}`, "glazing-frame", V(i * 1.12 * s, .92 * s, -1.48 * s), V(.027 * s, .92 * s, .027 * s), frame, 0, ["fixture", "frame"]);
  b.box("glazing/rail-low", "glazing-frame", V(0, .62 * s, -1.48 * s), V(1.18 * s, .025 * s, .027 * s), frame);
  b.box("glazing/rail-high", "glazing-frame", V(0, 1.26 * s, -1.48 * s), V(1.18 * s, .025 * s, .027 * s), frame);
  const wood = C(.34, .23, .12);
  b.box("bench/seat", "wood-bench", V(-1.18 * s, .31 * s, -.70 * s), V(.52 * s, .055 * s, .20 * s), wood, 0, ["bench"]);
  b.box("bench/back", "wood-bench", V(-1.18 * s, .58 * s, -.87 * s), V(.52 * s, .26 * s, .045 * s), cmul(wood, .82), 0, ["bench"]);
  for (const i of [-1, 1]) b.box(`bench/leg-${i < 0 ? "left" : "right"}`, "wood-bench", V((-1.18 + .38 * i) * s, .15 * s, -.70 * s), V(.035 * s, .16 * s, .15 * s), cmul(wood, .65), 0, ["bench"]);
  b.box("planter/stone", "stone-planter", V(1.12 * s, .24 * s, -.86 * s), V(.28 * s, .24 * s, .28 * s), C(.38, .35, .26), 0, ["planter"]);
  const leaf = C(.055, .30, .14);
  b.ellipsoid("planter/foliage-main", "leaf-foliage", V(1.12 * s, .62 * s, -.86 * s), V(.42 * s, .38 * s, .30 * s), leaf, 0, ["plant"]);
  b.ellipsoid("planter/foliage-left", "leaf-foliage", V(.90 * s, .82 * s, -.84 * s), V(.26 * s, .38 * s, .20 * s), cmul(leaf, .82), 0, ["plant"]);
  b.ellipsoid("planter/foliage-right", "leaf-foliage", V(1.32 * s, .88 * s, -.90 * s), V(.24 * s, .42 * s, .19 * s), cmul(leaf, .72), 0, ["plant"]);
  for (let i = -1; i <= 1; i++) {
    b.cylinder(`pendant-${i + 1}/cord`, "fixture-cord", V(i * .56 * s, 1.50 * s, -1.04 * s), .012 * s, .34 * s, C(.12, .12, .12), 0, ["fixture"]);
    b.ellipsoid(`pendant-${i + 1}/globe`, "emissive-glass", V(i * .56 * s, 1.18 * s, -1.04 * s), V(.095 * s, .095 * s, .095 * s), C(.85, .68, .38), .48, ["fixture", "light"]);
  }
}

function buildCourtyard(b: ProxyBuilder, s: number): void {
  const limestone = C(.62, .52, .38);
  for (const i of [-1, 1]) {
    const side = i < 0 ? "left" : "right", x = i * 1.16 * s;
    b.cylinder(`column-${side}/shaft`, "stone-column", V(x, .76 * s, -1.34 * s), .13 * s, .76 * s, limestone, 0, ["column"]);
    b.box(`column-${side}/capital`, "stone-column", V(x, 1.51 * s, -1.34 * s), V(.19 * s, .055 * s, .19 * s), cmul(limestone, .9), 0, ["column"]);
    b.box(`column-${side}/base`, "stone-column", V(x, .055 * s, -1.34 * s), V(.20 * s, .055 * s, .20 * s), cmul(limestone, .75), 0, ["column"]);
  }
  const tile = C(.46, .20, .12);
  b.box("bench/seat", "tile-bench", V(-1.12 * s, .27 * s, -.54 * s), V(.50 * s, .065 * s, .18 * s), tile, 0, ["bench"]);
  for (const i of [-1, 1]) b.box(`bench/leg-${i < 0 ? "left" : "right"}`, "tile-bench", V((-1.12 + .38 * i) * s, .13 * s, -.54 * s), V(.045 * s, .14 * s, .14 * s), cmul(tile, .7), 0, ["bench"]);
  b.cylinder("citrus/pot", "terracotta-pot", V(1.12 * s, .22 * s, -.68 * s), .27 * s, .22 * s, C(.56, .23, .12), 0, ["pot", "plant"]);
  b.cylinder("citrus/trunk", "tree-trunk", V(1.12 * s, .70 * s, -.68 * s), .045 * s, .40 * s, C(.20, .12, .055), 0, ["tree", "plant"]);
  const citrus = C(.12, .34, .10);
  b.ellipsoid("citrus/canopy-main", "leaf-foliage", V(1.12 * s, 1.12 * s, -.68 * s), V(.48 * s, .42 * s, .40 * s), citrus, 0, ["tree", "plant"]);
  b.ellipsoid("citrus/canopy-left", "leaf-foliage", V(.86 * s, 1.20 * s, -.70 * s), V(.27 * s, .27 * s, .27 * s), cmul(citrus, .82), 0, ["tree", "plant"]);
  b.ellipsoid("citrus/canopy-right", "leaf-foliage", V(1.35 * s, 1.26 * s, -.65 * s), V(.25 * s, .25 * s, .25 * s), cmul(citrus, .9), 0, ["tree", "plant"]);
  for (const i of [-1, 1]) b.ellipsoid(`citrus/fruit-${i < 0 ? "left" : "right"}`, "fruit", V((1.12 + .16 * i) * s, 1.12 * s, -.42 * s), V(.045 * s, .045 * s, .045 * s), C(.92, .46, .06), .08, ["fruit", "plant", "emissive-surface-only"]);
}

function buildNightLab(b: ProxyBuilder, scene: SceneDescription, s: number, floorY: number, roomHalf: Vec3): void {
  const th = { x: scene.container.width_m * .5 + .30 * s, y: scene.container.depth_m * .5 + .26 * s };
  const zb = -roomHalf.z + .36 * s, ceilY = floorY + 2 * roomHalf.y;
  const steel = C(.30, .32, .34);
  b.box("desk/top", "lab-bench", V(0, -.021 * s, 0), V(th.x, .019 * s, th.y), C(.35, .26, .17), 0, ["desk", "bench"]);
  b.box("desk/apron", "lab-bench", V(0, -.074 * s, 0), V(th.x - .07 * s, .034 * s, th.y - .07 * s), C(.125, .135, .145), 0, ["desk", "bench"]);
  for (const i of [-1, 1]) for (const j of [-1, 1]) b.box(`desk/leg-${i < 0 ? "l" : "r"}${j < 0 ? "f" : "b"}`, "steel-frame", V(i * (th.x - .10 * s), floorY + .34 * s, j * (th.y - .10 * s)), V(.027 * s, .34 * s, .027 * s), steel, 0, ["desk", "bench"]);
  b.box("desk/lower-shelf", "steel-frame", V(0, floorY + .16 * s, 0), V(th.x - .13 * s, .013 * s, th.y - .13 * s), C(.20, .21, .22), 0, ["desk", "bench"]);
  b.box("desk/controller", "instrument", V(-.35 * th.x, floorY + .235 * s, .15 * th.y), V(.15 * s, .062 * s, .115 * s), C(.10, .155, .165), 0, ["desk", "instrument"]);
  const lamp = V(-(th.x - .17 * s), 0, th.y - .20 * s);
  b.cylinder("desk-lamp/base", "metal-fixture", V(lamp.x, .010 * s, lamp.z), .085 * s, .013 * s, C(.095, .10, .11), 0, ["desk", "fixture", "light"]);
  b.cylinder("desk-lamp/stem", "metal-fixture", V(lamp.x, .31 * s, lamp.z), .012 * s, .29 * s, C(.14, .15, .16), 0, ["desk", "fixture", "light"]);
  b.ellipsoid("desk-lamp/shade", "metal-fixture", V(lamp.x, .63 * s, lamp.z), V(.098 * s, .072 * s, .098 * s), C(.055, .058, .062), 0, ["desk", "fixture", "light"]);
  b.ellipsoid("desk-lamp/bulb", "emissive-glass", V(lamp.x, .585 * s, lamp.z), V(.046 * s, .046 * s, .046 * s), C(1, .78, .45), 2.8, ["desk", "fixture", "light"]);
  const stoolX = th.x + .45 * s;
  b.cylinder("stool/seat", "wood-stool", V(stoolX, floorY + .47 * s, .10 * s), .17 * s, .024 * s, C(.235, .155, .095), 0, ["stool", "chair"]);
  b.cylinder("stool/post", "steel-frame", V(stoolX, floorY + .235 * s, .10 * s), .024 * s, .215 * s, cmul(steel, .8), 0, ["stool", "chair"]);
  b.cylinder("stool/base", "steel-frame", V(stoolX, floorY + .02 * s, .10 * s), .145 * s, .014 * s, cmul(steel, .6), 0, ["stool", "chair"]);
  b.box("counter/cabinet", "lab-counter", V(0, floorY + .42 * s, zb), V(1.72 * s, .42 * s, .30 * s), C(.165, .195, .215), 0, ["counter"]);
  b.box("counter/worktop", "lab-counter", V(0, floorY + .862 * s, zb), V(1.80 * s, .022 * s, .34 * s), C(.54, .54, .52), 0, ["counter"]);
  b.box("counter/monitor-stand", "metal-fixture", V(.95 * s, floorY + .93 * s, zb), V(.05 * s, .055 * s, .05 * s), C(.06, .065, .07), 0, ["counter", "instrument"]);
  b.box("counter/monitor", "monitor-frame", V(.95 * s, floorY + 1.17 * s, zb + .05 * s), V(.30 * s, .19 * s, .014 * s), C(.030, .034, .040), 0, ["counter", "instrument", "monitor"]);
  b.box("counter/monitor-screen", "monitor-glass", V(.95 * s, floorY + 1.17 * s, zb + .068 * s), V(.265 * s, .155 * s, .004 * s), C(.25, .45, .58), 1, ["counter", "instrument", "monitor", "light", "emits-positive-z"]);
  b.box("counter/keyboard", "instrument", V(.95 * s, floorY + .892 * s, zb + .22 * s), V(.19 * s, .007 * s, .07 * s), C(.085, .09, .10), 0, ["counter", "instrument"]);
  b.cylinder("counter/instrument-a", "instrument", V(-.58 * s, floorY + .966 * s, zb), .070 * s, .082 * s, C(.30, .36, .39), .02, ["counter", "instrument", "emissive-surface-only"]);
  b.cylinder("counter/instrument-b", "instrument", V(-.86 * s, floorY + 1.014 * s, zb + .04 * s), .046 * s, .13 * s, C(.27, .33, .36), .02, ["counter", "instrument", "emissive-surface-only"]);
  b.ellipsoid("counter/instrument-c", "instrument", V(-1.12 * s, floorY + .980 * s, zb - .02 * s), V(.088 * s, .096 * s, .088 * s), C(.28, .35, .34), .02, ["counter", "instrument", "emissive-surface-only"]);
  b.box("counter/shelf", "metal-fixture", V(-.20 * s, floorY + 1.62 * s, zb - .05 * s), V(1.22 * s, .016 * s, .17 * s), C(.29, .30, .31), 0, ["counter", "shelf"]);
  const bottleColors = [C(.36, .14, .10), C(.10, .235, .255), C(.27, .27, .26)];
  for (let i = 0; i < 3; i++) b.box(`counter/shelf-bottle-${i + 1}`, "instrument", V((-.92 + .13 * i) * s, floorY + 1.751 * s, zb - .05 * s), V(.050 * s, .115 * s, .135 * s), bottleColors[i], 0, ["counter", "shelf", "instrument"]);
  for (const i of [-1, 1]) for (let j = 0; j < 2; j++) {
    const z = (j === 0 ? -.30 : .95) * s;
    b.box(`fixtures/troffer-${i < 0 ? "left" : "right"}-${j + 1}`, "emissive-fixture", V(i * .95 * s, ceilY - .035 * s, z), V(.55 * s, .012 * s, .20 * s), C(.92, .93, .90), 2.3, ["fixture", "light", "emits-negative-y"]);
  }
}

function buildGallery(b: ProxyBuilder, s: number): void {
  const portal = C(.82, .39, .17);
  b.box("portal/left", "emissive-fixture", V(-.74 * s, .82 * s, -1.42 * s), V(.045 * s, .82 * s, .08 * s), portal, .12, ["portal", "fixture", "emissive-surface-only"]);
  b.box("portal/right", "emissive-fixture", V(.74 * s, .82 * s, -1.42 * s), V(.045 * s, .82 * s, .08 * s), portal, .12, ["portal", "fixture", "emissive-surface-only"]);
  b.box("portal/top", "emissive-fixture", V(0, 1.60 * s, -1.42 * s), V(.78 * s, .045 * s, .08 * s), portal, .12, ["portal", "fixture", "emissive-surface-only"]);
  const bench = C(.12, .14, .13);
  b.box("bench/seat", "gallery-bench", V(-1.08 * s, .27 * s, -.42 * s), V(.55 * s, .065 * s, .18 * s), bench, 0, ["bench"]);
  for (const i of [-1, 1]) b.box(`bench/leg-${i < 0 ? "left" : "right"}`, "gallery-bench", V((-1.08 + .40 * i) * s, .14 * s, -.42 * s), V(.04 * s, .14 * s, .13 * s), cmul(bench, .7), 0, ["bench"]);
  b.box("sculpture/plinth", "stone-plinth", V(1.08 * s, .25 * s, -.86 * s), V(.28 * s, .25 * s, .28 * s), C(.43, .43, .40), 0, ["plinth", "sculpture"]);
  b.ellipsoid("sculpture/lower", "sculpture", V(1.08 * s, .72 * s, -.86 * s), V(.30 * s, .42 * s, .23 * s), C(.08, .12, .11), 0, ["sculpture"]);
  b.ellipsoid("sculpture/upper", "sculpture", V(.94 * s, .97 * s, -.84 * s), V(.18 * s, .28 * s, .16 * s), C(.40, .17, .10), 0, ["sculpture"]);
}

function buildBathhouse(b: ProxyBuilder, s: number): void {
  const cedar = C(.34, .20, .105);
  for (let i = -2; i <= 2; i++) b.box(`screen/post-${i + 3}`, "cedar-screen", V(i * .52 * s, .84 * s, -1.34 * s), V(.025 * s, .84 * s, .035 * s), cedar, 0, ["screen", "fixture"]);
  for (let i = 0; i <= 2; i++) b.box(`screen/rail-${i + 1}`, "cedar-screen", V(0, (.22 + .53 * i) * s, -1.34 * s), V(1.18 * s, .025 * s, .035 * s), cmul(cedar, .82), 0, ["screen", "fixture"]);
  const stone = C(.24, .25, .22);
  b.cylinder("stool/left", "stone-stool", V(-1.08 * s, .18 * s, -.42 * s), .27 * s, .18 * s, stone, 0, ["stool"]);
  b.cylinder("stool/right", "stone-stool", V(1.06 * s, .16 * s, -.70 * s), .24 * s, .16 * s, cmul(stone, .82), 0, ["stool"]);
  b.cylinder("bucket/body", "wood-bucket", V(-.70 * s, .20 * s, -1.02 * s), .25 * s, .20 * s, C(.42, .25, .12), 0, ["bucket"]);
  b.cylinder("bucket/rim", "cedar-bucket", V(-.70 * s, .42 * s, -1.02 * s), .19 * s, .035 * s, cedar, 0, ["bucket"]);
  for (const i of [-1, 1]) {
    const side = i < 0 ? "left" : "right", x = i * .68 * s;
    b.box(`lantern-${side}/shade`, "emissive-fixture", V(x, 1.22 * s, -1.18 * s), V(.12 * s, .19 * s, .12 * s), C(.72, .57, .34), .22, ["lantern", "fixture", "light"]);
    b.cylinder(`lantern-${side}/cord`, "cedar-fixture", V(x, 1.48 * s, -1.18 * s), .008 * s, .09 * s, cedar, 0, ["lantern", "fixture"]);
  }
}

function buildStation(b: ProxyBuilder, s: number, floorY: number, roomHalf: Vec3): void {
  const metal = C(.025, .09, .115);
  for (let i = -2; i <= 2; i++) b.box(`pressure-rib-${i + 3}`, "metal-frame", V(i * .58 * s, 1.35 * s, -1.42 * s), V(.025 * s, 1.35 * s, .05 * s), metal, 0, ["rib", "fixture"]);
  for (const i of [-1, 1]) {
    const side = i < 0 ? "left" : "right", x = i * 1.16 * s;
    b.box(`console-${side}/cabinet`, "metal-console", V(x, .37 * s, -.72 * s), V(.34 * s, .37 * s, .30 * s), cmul(metal, .72), 0, ["console"]);
    b.box(`console-${side}/monitor`, "monitor-glass", V(x, .63 * s, -.40 * s), V(.25 * s, .13 * s, .018 * s), C(.06, .48, .58), .30, ["console", "monitor", "light", "emits-positive-z"]);
    b.cylinder(`console-${side}/pipe`, "metal-pipe", V(x + .25 * s, .88 * s, -1.06 * s), .055 * s, .72 * s, C(.12, .25, .27), 0, ["console", "pipe"]);
  }
  b.box("equipment-case/body", "equipment-case", V(-.76 * s, .23 * s, -1.04 * s), V(.30 * s, .23 * s, .24 * s), C(.12, .15, .15), 0, ["equipment"]);
  b.box("equipment-case/lid", "equipment-case", V(-.76 * s, .47 * s, -1.04 * s), V(.25 * s, .018 * s, .19 * s), C(.74, .48, .16), .12, ["equipment", "emissive-surface-only"]);
  for (let i = -1; i <= 1; i++) b.ellipsoid(`indicator-${i + 2}`, "emissive-fixture", V(i * .52 * s, 1.36 * s, -1.12 * s), V(.055 * s, .055 * s, .055 * s), C(.10, .65, .72), .36, ["fixture", "light"]);
  // The raster station wall carries procedural circular portholes. The SVO
  // catalog cannot express a subtractive annulus, but it can retain a stable
  // finite observation-port frame and backing for the analytic thin pane.
  const portY = floorY + 1.55 * s, portZ = -roomHalf.z + .018 * s;
  const frame = C(.34, .27, .13), backing = C(.008, .035, .052);
  b.box("observation-port/backing", "porthole-backing", V(0, portY, portZ - .006 * s), V(.66 * s, .39 * s, .003 * s), backing, 0, ["porthole", "frame"]);
  b.box("observation-port/frame-left", "metal-frame", V(-.705 * s, portY, portZ), V(.045 * s, .435 * s, .018 * s), frame, 0, ["porthole", "frame", "fixture"]);
  b.box("observation-port/frame-right", "metal-frame", V(.705 * s, portY, portZ), V(.045 * s, .435 * s, .018 * s), frame, 0, ["porthole", "frame", "fixture"]);
  b.box("observation-port/frame-bottom", "metal-frame", V(0, portY - .435 * s, portZ), V(.75 * s, .045 * s, .018 * s), frame, 0, ["porthole", "frame", "fixture"]);
  b.box("observation-port/frame-top", "metal-frame", V(0, portY + .435 * s, portZ), V(.75 * s, .045 * s, .018 * s), frame, 0, ["porthole", "frame", "fixture"]);
}

function buildGarden(b: ProxyBuilder, scene: SceneDescription, s: number): void {
  const g = scene.terrain?.baseHeight_m ?? 0;
  const trunk = C(.52, .49, .45);
  const canopy = C(.86, .85, .82);
  const cap = C(.90, .88, .85);
  const gill = C(.55, .52, .49);
  const stem = C(.78, .75, .71);
  const pebble = C(.68, .67, .64);
  b.cylinder("tree-big/trunk", "tree-trunk", V(-.45 * s, g + .15 * s, -.117 * s), .055 * s, .15 * s, trunk, 0, ["tree"]);
  b.ellipsoid("tree-big/canopy-main", "leaf-foliage", V(-.40 * s, g + .50 * s, -.08 * s), V(.30 * s, .24 * s, .28 * s), canopy, 0, ["tree"]);
  b.ellipsoid("tree-big/canopy-side", "leaf-foliage", V(-.56 * s, g + .42 * s, -.20 * s), V(.17 * s, .14 * s, .16 * s), cmul(canopy, .92), 0, ["tree"]);
  b.ellipsoid("tree-big/canopy-top", "leaf-foliage", V(-.33 * s, g + .66 * s, -.02 * s), V(.15 * s, .12 * s, .14 * s), cmul(canopy, 1.04), 0, ["tree"]);
  b.cylinder("tree-round/trunk", "tree-trunk", V(.45 * s, g + .12 * s, -.25 * s), .045 * s, .12 * s, cmul(trunk, .95), 0, ["tree"]);
  b.ellipsoid("tree-round/canopy-main", "leaf-foliage", V(.45 * s, g + .40 * s, -.25 * s), V(.22 * s, .18 * s, .21 * s), cmul(canopy, .96), 0, ["tree"]);
  b.ellipsoid("tree-round/canopy-side", "leaf-foliage", V(.56 * s, g + .50 * s, -.19 * s), V(.12 * s, .10 * s, .11 * s), cmul(canopy, .88), 0, ["tree"]);
  b.cylinder("mushroom-grand/stem", "mushroom-stem", V(-.20 * s, g + .10 * s, -.33 * s), .05 * s, .10 * s, stem, 0, ["mushroom"]);
  b.cylinder("mushroom-grand/gill", "mushroom-gill", V(-.20 * s, g + .205 * s, -.33 * s), .115 * s, .012 * s, gill, 0, ["mushroom"]);
  b.ellipsoid("mushroom-grand/cap", "mushroom-cap", V(-.20 * s, g + .25 * s, -.33 * s), V(.14 * s, .088 * s, .14 * s), cap, 0, ["mushroom"]);
  b.cylinder("mushroom-sprout/stem", "mushroom-stem", V(-.13 * s, g + .055 * s, -.36 * s), .028 * s, .055 * s, cmul(stem, .9), 0, ["mushroom"]);
  b.ellipsoid("mushroom-sprout/cap", "mushroom-cap", V(-.13 * s, g + .135 * s, -.36 * s), V(.072 * s, .046 * s, .072 * s), cmul(cap, .90), 0, ["mushroom"]);
  b.cylinder("mushroom-tall/stem", "mushroom-stem", V(-.30 * s, g + .075 * s, .25 * s), .036 * s, .075 * s, cmul(stem, .96), 0, ["mushroom"]);
  b.cylinder("mushroom-tall/gill", "mushroom-gill", V(-.30 * s, g + .16 * s, .25 * s), .085 * s, .010 * s, gill, 0, ["mushroom"]);
  b.ellipsoid("mushroom-tall/cap", "mushroom-cap", V(-.30 * s, g + .20 * s, .25 * s), V(.105 * s, .066 * s, .105 * s), cmul(cap, .97), 0, ["mushroom"]);
  b.cylinder("mushroom-button/stem", "mushroom-stem", V(.38 * s, g + .06 * s, .07 * s), .032 * s, .06 * s, cmul(stem, .92), 0, ["mushroom"]);
  b.ellipsoid("mushroom-button/cap", "mushroom-cap", V(.38 * s, g + .15 * s, .07 * s), V(.085 * s, .054 * s, .085 * s), cmul(cap, .94), 0, ["mushroom"]);
  b.cylinder("mushroom-pip/stem", "mushroom-stem", V(.24 * s, g + .045 * s, -.345 * s), .024 * s, .045 * s, cmul(stem, .88), 0, ["mushroom"]);
  b.ellipsoid("mushroom-pip/cap", "mushroom-cap", V(.24 * s, g + .115 * s, -.345 * s), V(.06 * s, .04 * s, .06 * s), cmul(cap, .86), 0, ["mushroom"]);
  b.ellipsoid("pebble-1/body", "stone-pebble", V(.21 * s, g + .02 * s, -.29 * s), V(.05 * s, .034 * s, .046 * s), pebble, 0, ["pebble"]);
  b.ellipsoid("pebble-2/body", "stone-pebble", V(.26 * s, g + .015 * s, -.325 * s), V(.036 * s, .024 * s, .033 * s), cmul(pebble, .9), 0, ["pebble"]);
  b.ellipsoid("pebble-3/body", "stone-pebble", V(-.05 * s, g + .02 * s, .31 * s), V(.045 * s, .03 * s, .042 * s), cmul(pebble, .95), 0, ["pebble"]);
}

/**
 * Mirrors every world-space primitive in sampleEnvironmentProps(). Room shell
 * faces are separate because their shader material is spatially procedural.
 */
export function buildEnvironmentProxyCatalog(scene: SceneDescription, environmentId: EnvironmentId, options: EnvironmentProxyCatalogOptions = {}): EnvironmentProxyCatalog {
  const s = Math.max(scene.container.width_m, scene.container.height_m, scene.container.depth_m);
  const roomHalf = V(Math.max(scene.container.width_m * 2.8, s * 2.25), Math.max(scene.container.height_m * 1.85, s * 1.8), Math.max(scene.container.depth_m * 2.8, s * 2.25));
  const floorY = environmentId === "night-lab" ? -.72 * s : environmentId === "garden" ? (scene.terrain?.baseHeight_m ?? 0) : -.025;
  const thickness = options.shellThickness_m ?? scene.nominalResolution.length_m;
  if (!(thickness > 0) || !Number.isFinite(thickness)) throw new Error("Environment shell thickness must be positive and finite");
  const b = new ProxyBuilder(environmentId);
  let shell: EnvironmentProxyShell;
  if (environmentId === "default") {
    const t = thickness * .5;
    b.box("shell/floor", "shell-floor", V(0, -.012 - t, 0), V(roomHalf.x, t, roomHalf.z), C(.055, .068, .064), 0, ["shell", "floor"], true);
    shell = { kind: "floor", floorY_m: -.012, bounds_m: aabb(V(0, -.012, 0), V(roomHalf.x, 0, roomHalf.z)), primitives: b.shell, materialModel: "default-floor" };
  } else if (environmentId === "garden") {
    const terrainTop = Math.max(scene.container.height_m, scene.terrain?.baseHeight_m ?? 0);
    shell = {
      kind: "terrain-heightfield", floorY_m: floorY,
      bounds_m: { min: V(-roomHalf.x, 0, -roomHalf.z), max: V(roomHalf.x, terrainTop, roomHalf.z) },
      primitives: b.shell, materialModel: "garden-terrain"
    };
  } else shell = addRoomShell(b, environmentId, roomHalf, floorY, thickness, s);

  if (environmentId === "conservatory") buildConservatory(b, s);
  else if (environmentId === "courtyard") buildCourtyard(b, s);
  else if (environmentId === "night-lab") buildNightLab(b, scene, s, floorY, roomHalf);
  else if (environmentId === "concrete-gallery") buildGallery(b, s);
  else if (environmentId === "bathhouse") buildBathhouse(b, s);
  else if (environmentId === "research-station") buildStation(b, s, floorY, roomHalf);
  else if (environmentId === "garden") buildGarden(b, scene, s);

  return { environmentId, environmentIndex: environmentIndex(environmentId), scale_m: s, floorY_m: floorY, shell, primitives: b.props };
}

export function environmentProxyPrimitives(catalog: EnvironmentProxyCatalog, includeShell = true): readonly EnvironmentProxyPrimitive[] {
  return includeShell ? [...catalog.shell.primitives, ...catalog.primitives] : catalog.primitives;
}

export interface EnvironmentProxyMaterialEntry {
  /** Matches the primitive ownerIndex and is therefore directly GPU-indexable. */
  readonly index: number;
  readonly key: string;
  readonly material: EnvironmentProxyMaterial;
}

/** Stable owner-aligned material table; repeated authored materials intentionally retain separate named slots. */
export function environmentProxyMaterialTable(catalog: EnvironmentProxyCatalog, includeShell = true): readonly EnvironmentProxyMaterialEntry[] {
  return environmentProxyPrimitives(catalog, includeShell).map((primitive) => ({
    index: primitive.ownerIndex, key: primitive.key, material: primitive.material
  }));
}

export interface SparseBrickCoordinateOptions {
  readonly cellSize_m: number | Vec3;
  readonly worldOrigin_m: Vec3;
  readonly brickSize_cells: number;
}

export interface Integer3 { readonly x: number; readonly y: number; readonly z: number }
export interface VoxelCellRange { readonly minInclusive: Integer3; readonly maxInclusive: Integer3 }

function positiveCellSize(value: number | Vec3): Vec3 {
  const h = typeof value === "number" ? V(value, value, value) : value;
  if (!(h.x > 0) || !(h.y > 0) || !(h.z > 0) || !Number.isFinite(h.x + h.y + h.z)) throw new Error("Voxel cell size must be positive and finite");
  return h;
}

/** Conservative cell bounds; a face exactly on a cell boundary retains both touching cells. */
export function voxelCellRangeForAabb(bounds_m: EnvironmentProxyAabb, cellSize_m: number | Vec3, worldOrigin_m: Vec3): VoxelCellRange {
  const h = positiveCellSize(cellSize_m);
  return {
    minInclusive: {
      x: Math.floor((bounds_m.min.x - worldOrigin_m.x) / h.x),
      y: Math.floor((bounds_m.min.y - worldOrigin_m.y) / h.y),
      z: Math.floor((bounds_m.min.z - worldOrigin_m.z) / h.z)
    },
    maxInclusive: {
      x: Math.floor((bounds_m.max.x - worldOrigin_m.x) / h.x),
      y: Math.floor((bounds_m.max.y - worldOrigin_m.y) / h.y),
      z: Math.floor((bounds_m.max.z - worldOrigin_m.z) / h.z)
    }
  };
}

/** Unique lexicographically sorted sparse-brick coordinates covering the supplied AABBs. */
export function sparseBrickCoordinatesForAabbs(bounds: readonly EnvironmentProxyAabb[], options: SparseBrickCoordinateOptions): readonly Integer3[] {
  if (!Number.isInteger(options.brickSize_cells) || options.brickSize_cells <= 0) throw new Error("Sparse brick size must be a positive integer");
  const keys = new Map<string, Integer3>();
  for (const aabb_m of bounds) {
    const range = voxelCellRangeForAabb(aabb_m, options.cellSize_m, options.worldOrigin_m);
    const min = V(Math.floor(range.minInclusive.x / options.brickSize_cells), Math.floor(range.minInclusive.y / options.brickSize_cells), Math.floor(range.minInclusive.z / options.brickSize_cells));
    const max = V(Math.floor(range.maxInclusive.x / options.brickSize_cells), Math.floor(range.maxInclusive.y / options.brickSize_cells), Math.floor(range.maxInclusive.z / options.brickSize_cells));
    for (let z = min.z; z <= max.z; z++) for (let y = min.y; y <= max.y; y++) for (let x = min.x; x <= max.x; x++) keys.set(`${x},${y},${z}`, { x, y, z });
  }
  return [...keys.values()].sort((a, b) => a.z - b.z || a.y - b.y || a.x - b.x);
}

export function sparseBrickCoordinatesForEnvironment(catalog: EnvironmentProxyCatalog, options: SparseBrickCoordinateOptions, includeShell = false): readonly Integer3[] {
  return sparseBrickCoordinatesForAabbs(environmentProxyPrimitives(catalog, includeShell).map((primitive) => primitive.aabb_m), options);
}
