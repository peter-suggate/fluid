import type { EnvironmentId } from "./environments";
import type { SceneDescription } from "./model";
import type { SceneryGraph, SceneryNode } from "./scenery-graph";

/**
 * The declarative scenery an environment seeds a new scene with.
 *
 * These are *defaults*, not authority. A scene copies the graph it wants at
 * creation and owns it from then on; nothing here is consulted again, and
 * editing scenery is an ordinary edit to `scene.scenery`.
 *
 * The three paper figures — hose-filled tank, garden lighting, jet past sphere —
 * carry numbers recovered from the imperative modules they replaced, primitive
 * for primitive, because their composition is the published result. The other
 * five sets were re-authored: see the note above `courtyardNodes`.
 */

/** Glazed white clay. Faintly warm, and only ever varied in value. */
const CLAY = { tint: [1, .985, .955] as const };
/** Unglazed stone: the same value range a touch cooler, so rock parts from porcelain. */
const STONE = { tint: [.972, .984, 1] as const };

/**
 * Porcelain garden: a bonsai landscape in glazed white clay, built around the
 * pond. The ground is a real heightfield, so every object here is anchored to
 * the surface it actually stands on — which is what lets reeds stand in the
 * shallows and rocks bed into the knoll. The bridge, the lily pads and the
 * lamppost take the lawn datum instead: a bridge is one rigid object and must
 * not follow the bank it spans.
 *
 * Composition rule: the water is the subject. Nothing stands in the middle of
 * the basin; trees, mushrooms, rocks, reeds and the bridge ring it or lean
 * toward it. Palette is value only — the lantern ember and the lamppost are the
 * only saturated colour in the set, and are the only literal materials here.
 */
const GARDEN_NODES: readonly SceneryNode[] = [
  { kind: "terrain-shell", id: "shell", materialModel: "garden-terrain" },
  // The one grown object in the set: a seed, not geometry, so re-seeding re-grows it.
  { kind: "tree", id: "tree-hero", group: "tree-hero", place: { position: { x: -.45, y: 0, z: -.117 }, anchor: "terrain", ground: [-.45, -.117] },
    height: .66, rootRadius: .036, spread: .30, seed: 0x5eed9a3, lean: [1, .28], bark: "clay", leaf: "clay", sway: true },
  { kind: "group", id: "tree-knoll", place: { position: { x: 0, y: 0, z: 0 }, anchor: "terrain", ground: [-0.373, 0.233] }, children: [
    { kind: "cylinder", id: "tree-knoll/trunk", group: "tree-trunk", tags: ["tree"], place: { position: { x: -0.373, y: 0.135, z: 0.233 } }, radius: 0.038, halfHeight: 0.135, material: { palette: "clay", value: 0.49 } },
    { kind: "ellipsoid", id: "tree-knoll/canopy-low", group: "leaf-foliage", tags: ["tree"], place: { position: { x: -0.405, y: 0.24499999999999997, z: 0.20000000000000004 } }, radius: { x: 0.145, y: 0.078, z: 0.135 }, material: { palette: "clay", value: 0.82 } },
    { kind: "ellipsoid", id: "tree-knoll/canopy-main", group: "leaf-foliage", tags: ["tree"], place: { position: { x: -0.355, y: 0.3499999999999999, z: 0.245 } }, radius: { x: 0.17499999999999996, y: 0.09200000000000001, z: 0.165 }, material: { palette: "clay", value: 0.89 } },
    { kind: "ellipsoid", id: "tree-knoll/canopy-top", group: "leaf-foliage", tags: ["tree"], place: { position: { x: -0.32, y: 0.45, z: 0.215 } }, radius: { x: 0.115, y: 0.065, z: 0.11 }, material: { palette: "clay", value: 0.95 } },
  ] },
  { kind: "group", id: "mushroom-grand", place: { position: { x: 0, y: 0, z: 0 }, anchor: "terrain", ground: [-0.2, -0.33] }, children: [
    { kind: "cylinder", id: "mushroom-grand/stem", group: "mushroom-stem", tags: ["mushroom"], place: { position: { x: -0.20000000000000004, y: 0.10000000000000002, z: -0.33 } }, radius: 0.05000000000000001, halfHeight: 0.10000000000000002, material: { palette: "clay", value: 0.72 } },
    { kind: "cylinder", id: "mushroom-grand/gill", group: "mushroom-gill", tags: ["mushroom"], place: { position: { x: -0.20000000000000004, y: 0.205, z: -0.33 } }, radius: 0.115, halfHeight: 0.012000000000000002, material: { palette: "clay", value: 0.45 } },
    { kind: "ellipsoid", id: "mushroom-grand/cap", group: "mushroom-cap", tags: ["mushroom"], place: { position: { x: -0.20000000000000004, y: 0.24999999999999997, z: -0.33 } }, radius: { x: 0.14, y: 0.08800000000000001, z: 0.14 }, material: { palette: "clay", value: 0.94 } },
  ] },
  { kind: "group", id: "mushroom-tall", place: { position: { x: 0, y: 0, z: 0 }, anchor: "terrain", ground: [-0.3, 0.25] }, children: [
    { kind: "cylinder", id: "mushroom-tall/stem", group: "mushroom-stem", tags: ["mushroom"], place: { position: { x: -0.3, y: 0.075, z: 0.25 } }, radius: 0.036, halfHeight: 0.075, material: { palette: "clay", value: 0.7 } },
    { kind: "cylinder", id: "mushroom-tall/gill", group: "mushroom-gill", tags: ["mushroom"], place: { position: { x: -0.3, y: 0.16, z: 0.25 } }, radius: 0.085, halfHeight: 0.01, material: { palette: "clay", value: 0.45 } },
    { kind: "ellipsoid", id: "mushroom-tall/cap", group: "mushroom-cap", tags: ["mushroom"], place: { position: { x: -0.3, y: 0.20000000000000004, z: 0.25 } }, radius: { x: 0.105, y: 0.066, z: 0.105 }, material: { palette: "clay", value: 0.91 } },
  ] },
  { kind: "group", id: "mushroom-crown", place: { position: { x: 0, y: 0, z: 0 }, anchor: "terrain", ground: [0.413, 0.133] }, children: [
    { kind: "cylinder", id: "mushroom-crown/stem", group: "mushroom-stem", tags: ["mushroom"], place: { position: { x: 0.413, y: 0.11500000000000003, z: 0.133 } }, radius: 0.052, halfHeight: 0.115, material: { palette: "clay", value: 0.73 } },
    { kind: "cylinder", id: "mushroom-crown/gill", group: "mushroom-gill", tags: ["mushroom"], place: { position: { x: 0.413, y: 0.235, z: 0.133 } }, radius: 0.12, halfHeight: 0.013, material: { palette: "clay", value: 0.44 } },
    { kind: "ellipsoid", id: "mushroom-crown/cap", group: "mushroom-cap", tags: ["mushroom"], place: { position: { x: 0.413, y: 0.285, z: 0.133 } }, radius: { x: 0.15, y: 0.09500000000000001, z: 0.15 }, material: { palette: "clay", value: 0.95 } },
  ] },
  { kind: "group", id: "rock-sentinel", place: { position: { x: 0, y: 0, z: 0 }, anchor: "terrain", ground: [-0.393, -0.04] }, children: [
    { kind: "ellipsoid", id: "rock-sentinel/body", group: "stone-rock", tags: ["rock"], place: { position: { x: -0.393, y: 0.17499999999999996, z: -0.04 } }, radius: { x: 0.105, y: 0.17499999999999996, z: 0.09000000000000001 }, material: { palette: "stone", value: 0.6 } },
    { kind: "ellipsoid", id: "rock-sentinel/shoulder", group: "stone-rock", tags: ["rock"], place: { position: { x: -0.42, y: 0.085, z: -0.055 } }, radius: { x: 0.085, y: 0.085, z: 0.075 }, material: { palette: "stone", value: 0.66 } },
  ] },
  { kind: "group", id: "rock-recline", place: { position: { x: 0, y: 0, z: 0 }, anchor: "terrain", ground: [-0.433, 0.02] }, children: [
    { kind: "ellipsoid", id: "rock-recline/body", group: "stone-rock", tags: ["rock"], place: { position: { x: -0.433, y: 0.075, z: 0.02 } }, radius: { x: 0.17, y: 0.075, z: 0.115 }, material: { palette: "stone", value: 0.64 } },
  ] },
  { kind: "group", id: "rock-promontory-a", place: { position: { x: 0, y: 0, z: 0 }, anchor: "terrain", ground: [0.06, 0.267] }, children: [
    { kind: "ellipsoid", id: "rock-promontory-a/body", group: "stone-rock", tags: ["rock"], place: { position: { x: 0.06, y: 0.075, z: 0.267 } }, radius: { x: 0.10000000000000002, y: 0.075, z: 0.085 }, material: { palette: "stone", value: 0.63 } },
  ] },
  { kind: "group", id: "rock-near", place: { position: { x: 0, y: 0, z: 0 }, anchor: "terrain", ground: [0.433, 0.183] }, children: [
    { kind: "ellipsoid", id: "rock-near/body", group: "stone-rock", tags: ["rock"], place: { position: { x: 0.433, y: 0.09000000000000001, z: 0.18299999999999997 } }, radius: { x: 0.115, y: 0.09000000000000001, z: 0.10000000000000002 }, material: { palette: "stone", value: 0.61 } },
  ] },
  { kind: "group", id: "lantern-stone", place: { position: { x: 0, y: 0, z: 0 }, anchor: "terrain", ground: [0.0167, 0.233] }, children: [
    { kind: "cylinder", id: "lantern-stone/base", group: "stone-lantern", tags: ["lantern"], place: { position: { x: 0.0167, y: 0.02199999999999998, z: 0.233 } }, radius: 0.105, halfHeight: 0.022000000000000002, material: { palette: "stone", value: 0.72 } },
    { kind: "cylinder", id: "lantern-stone/post", group: "stone-lantern", tags: ["lantern"], place: { position: { x: 0.0167, y: 0.05800000000000002, z: 0.233 } }, radius: 0.038, halfHeight: 0.022000000000000002, material: { palette: "stone", value: 0.78 } },
    { kind: "box", id: "lantern-stone/wall-left", group: "stone-lantern", tags: ["lantern"], place: { position: { x: -0.0363, y: 0.10500000000000002, z: 0.233 } }, halfSize: { x: 0.014, y: 0.037, z: 0.058 }, material: { palette: "stone", value: 0.84 } },
    { kind: "box", id: "lantern-stone/wall-right", group: "stone-lantern", tags: ["lantern"], place: { position: { x: 0.0697, y: 0.10500000000000002, z: 0.233 } }, halfSize: { x: 0.014, y: 0.037, z: 0.058 }, material: { palette: "stone", value: 0.84 } },
    { kind: "ellipsoid", id: "lantern-stone/ember", group: "emissive-fixture", tags: ["lantern", "fixture", "light", "point-light"], place: { position: { x: 0.0167, y: 0.10500000000000002, z: 0.233 } }, radius: { x: 0.036, y: 0.033, z: 0.036 }, material: { colorLinear: [1, 0.62, 0.3], emission: 1.4 } },
    { kind: "cylinder", id: "lantern-stone/roof", group: "stone-lantern", tags: ["lantern"], place: { position: { x: 0.0167, y: 0.16, z: 0.233 } }, radius: 0.105, halfHeight: 0.022000000000000002, material: { palette: "stone", value: 0.87 } },
    { kind: "ellipsoid", id: "lantern-stone/finial", group: "stone-lantern", tags: ["lantern"], place: { position: { x: 0.0167, y: 0.20000000000000007, z: 0.233 } }, radius: { x: 0.028, y: 0.032, z: 0.028 }, material: { palette: "stone", value: 0.91 } },
  ] },
  { kind: "box", id: "bridge/deck-1", group: "bridge-deck", tags: ["bridge"], place: { position: { x: 0.0467, y: 0.02, z: 0.22 }, anchor: "floor" }, halfSize: { x: 0.032, y: 0.016, z: 0.05000000000000001 }, material: { palette: "clay", value: 0.86 } },
  { kind: "box", id: "bridge/deck-2", group: "bridge-deck", tags: ["bridge"], place: { position: { x: 0.10000000000000002, y: 0.042, z: 0.22 }, anchor: "floor" }, halfSize: { x: 0.032, y: 0.016, z: 0.05000000000000001 }, material: { palette: "clay", value: 0.88 } },
  { kind: "box", id: "bridge/deck-3", group: "bridge-deck", tags: ["bridge"], place: { position: { x: 0.1533, y: 0.05000000000000001, z: 0.22 }, anchor: "floor" }, halfSize: { x: 0.032, y: 0.016, z: 0.05000000000000001 }, material: { palette: "clay", value: 0.9 } },
  { kind: "box", id: "bridge/deck-4", group: "bridge-deck", tags: ["bridge"], place: { position: { x: 0.2067, y: 0.042, z: 0.22 }, anchor: "floor" }, halfSize: { x: 0.032, y: 0.016, z: 0.05000000000000001 }, material: { palette: "clay", value: 0.88 } },
  { kind: "box", id: "bridge/deck-5", group: "bridge-deck", tags: ["bridge"], place: { position: { x: 0.26, y: 0.02, z: 0.22 }, anchor: "floor" }, halfSize: { x: 0.032, y: 0.016, z: 0.05000000000000001 }, material: { palette: "clay", value: 0.86 } },
  { kind: "box", id: "bridge/footing-west", group: "stone-footing", tags: ["bridge"], place: { position: { x: 0.013, y: -0.034999999999999996, z: 0.22 }, anchor: "floor" }, halfSize: { x: 0.045000000000000005, y: 0.06, z: 0.055 }, material: { palette: "stone", value: 0.7 } },
  { kind: "box", id: "bridge/footing-east", group: "stone-footing", tags: ["bridge"], place: { position: { x: 0.293, y: -0.034999999999999996, z: 0.22 }, anchor: "floor" }, halfSize: { x: 0.045000000000000005, y: 0.06, z: 0.055 }, material: { palette: "stone", value: 0.7 } },
  { kind: "cylinder", id: "bridge/post-a", group: "bridge-post", tags: ["bridge"], place: { position: { x: 0.10000000000000002, y: 0.08800000000000001, z: 0.1767 }, anchor: "floor" }, radius: 0.014, halfHeight: 0.034, material: { palette: "clay", value: 0.84 } },
  { kind: "cylinder", id: "bridge/post-b", group: "bridge-post", tags: ["bridge"], place: { position: { x: 0.10000000000000002, y: 0.08800000000000001, z: 0.2633 }, anchor: "floor" }, radius: 0.014, halfHeight: 0.034, material: { palette: "clay", value: 0.84 } },
  { kind: "cylinder", id: "bridge/post-c", group: "bridge-post", tags: ["bridge"], place: { position: { x: 0.2067, y: 0.08800000000000001, z: 0.1767 }, anchor: "floor" }, radius: 0.014, halfHeight: 0.034, material: { palette: "clay", value: 0.84 } },
  { kind: "cylinder", id: "bridge/post-d", group: "bridge-post", tags: ["bridge"], place: { position: { x: 0.2067, y: 0.08800000000000001, z: 0.2633 }, anchor: "floor" }, radius: 0.014, halfHeight: 0.034, material: { palette: "clay", value: 0.84 } },
  { kind: "ellipsoid", id: "reed-east/tussock", group: "leaf-tussock", tags: ["reed", "bank"], place: { position: { x: 0.32, y: 0.015999999999999997, z: 0.0867 }, anchor: "terrain" }, radius: { x: 0.0651, y: 0.026, z: 0.0609 }, material: { palette: "clay", value: 0.62 } },
  { kind: "cylinder", id: "reed-east/blade-0", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: 0.32, y: 0.04700000000000001, z: 0.0867 }, anchor: "terrain" }, radius: 0.011247445848342029, halfHeight: 0.059, material: { palette: "clay", value: 0.7566646133735777 } },
  { kind: "ellipsoid", id: "reed-east/seed-head", group: "leaf-seedhead", tags: ["reed", "bank"], place: { position: { x: 0.32, y: 0.13799999999999998, z: 0.0867 }, anchor: "terrain" }, radius: { x: 0.019, y: 0.038, z: 0.019 }, material: { palette: "clay", value: 0.5 } },
  { kind: "cylinder", id: "reed-east/blade-1", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: 0.32531380513926683, y: 0.04401187916495838, z: 0.05736745436257081 }, anchor: "terrain" }, radius: 0.009485234419573099, halfHeight: 0.056011879164958374, material: { palette: "clay", value: 0.8373615223076195 } },
  { kind: "cylinder", id: "reed-east/blade-2", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: 0.30442193459756245, y: 0.03577451489062514, z: 0.05393796063660063 }, anchor: "terrain" }, radius: 0.011806411533839998, halfHeight: 0.04777451489062514, material: { palette: "clay", value: 0.8281182277202608 } },
  { kind: "cylinder", id: "reed-east/blade-3", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: 0.3333016468204892, y: 0.015669750106055296, z: 0.08567324293316525 }, anchor: "terrain" }, radius: 0.012664811117611823, halfHeight: 0.027669750106055286, material: { palette: "clay", value: 0.6254970542155207 } },
  { kind: "cylinder", id: "reed-east/blade-4", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: 0.30325927969084643, y: 0.0355413719783537, z: 0.07539363142332456 }, anchor: "terrain" }, radius: 0.010994944965317846, halfHeight: 0.04754137197835371, material: { palette: "clay", value: 0.7457325456105173 } },
  { kind: "cylinder", id: "reed-shore/blade-0", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: 0.30261891796724666, y: 0.03360709824436344, z: 0.2155403436393204 }, anchor: "terrain" }, radius: 0.009663550930581988, halfHeight: 0.04560709824436345, material: { palette: "clay", value: 0.7908227662649007 } },
  { kind: "cylinder", id: "reed-shore/blade-1", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: 0.2792498330501607, y: 0.014265795383369547, z: 0.2354503274598209 }, anchor: "terrain" }, radius: 0.010499558980409057, halfHeight: 0.026265795383369553, material: { palette: "clay", value: 0.755328127341345 } },
  { kind: "cylinder", id: "reed-shore/blade-2", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: 0.26555251188632467, y: 0.014604325440712274, z: 0.194197349578326 }, anchor: "terrain" }, radius: 0.010703069434426725, halfHeight: 0.026604325440712274, material: { palette: "clay", value: 0.8147466846369208 } },
  { kind: "cylinder", id: "reed-north/blade-0", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: 0.08146704095415301, y: 0.028206902154372066, z: 0.21983792491136148 }, anchor: "terrain" }, radius: 0.011049925292115658, halfHeight: 0.04020690215437207, material: { palette: "clay", value: 0.7884033679775895 } },
  { kind: "cylinder", id: "reed-north/blade-1", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: 0.0860794857840002, y: 0.02016153399087489, z: 0.2603073818504219 }, anchor: "terrain" }, radius: 0.010195249009570107, halfHeight: 0.03216153399087489, material: { palette: "clay", value: 0.7055713305901736 } },
  { kind: "cylinder", id: "reed-north/blade-2", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: 0.0688787175190334, y: 0.026288482046569694, z: 0.26400014482499706 }, anchor: "terrain" }, radius: 0.01123630381568335, halfHeight: 0.0382884820465697, material: { palette: "clay", value: 0.6336405403912068 } },
  { kind: "cylinder", id: "reed-west/blade-0", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: -0.3467, y: 0.042, z: -0.0467 }, anchor: "terrain" }, radius: 0.012200687512636185, halfHeight: 0.054, material: { palette: "clay", value: 0.8277028656471521 } },
  { kind: "ellipsoid", id: "reed-west/seed-head", group: "leaf-seedhead", tags: ["reed", "bank"], place: { position: { x: -0.3467, y: 0.12799999999999997, z: -0.0467 }, anchor: "terrain" }, radius: { x: 0.019, y: 0.038, z: 0.019 }, material: { palette: "clay", value: 0.5 } },
  { kind: "cylinder", id: "reed-west/blade-1", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: -0.3345620534816878, y: 0.03145437406003474, z: -0.025137075047901396 }, anchor: "terrain" }, radius: 0.011489564508005977, halfHeight: 0.04345437406003475, material: { palette: "clay", value: 0.7559448453038932 } },
  { kind: "cylinder", id: "reed-west/blade-2", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: -0.35845250171114024, y: 0.04006450979411602, z: -0.008922577188897808 }, anchor: "terrain" }, radius: 0.009440119699940085, halfHeight: 0.05206450979411601, material: { palette: "clay", value: 0.7845334346499295 } },
  { kind: "cylinder", id: "reed-beach/blade-0", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: 0.04598750586551584, y: 0.03380601084115916, z: -0.29588780801839754 }, anchor: "terrain" }, radius: 0.01040719501922652, halfHeight: 0.04580601084115915, material: { palette: "clay", value: 0.6861811308562756 } },
  { kind: "cylinder", id: "reed-beach/blade-1", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: 0.07457893071233017, y: 0.02024251699703744, z: -0.2865296045211196 }, anchor: "terrain" }, radius: 0.010246171857666225, halfHeight: 0.03224251699703745, material: { palette: "clay", value: 0.7971337881032379 } },
  { kind: "cylinder", id: "reed-beach/blade-2", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: 0.06681119498624076, y: 0.022221943407086664, z: -0.29060231313914614 }, anchor: "terrain" }, radius: 0.00999770512354374, halfHeight: 0.034221943407086654, material: { palette: "clay", value: 0.7726200712844731 } },
  { kind: "cylinder", id: "tuft-near/blade-0", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: 0.37963069430490504, y: 0.004994664401281619, z: 0.36649944095725967 }, anchor: "terrain" }, radius: 0.008878683935135603, halfHeight: 0.016994664401281626, material: { palette: "clay", value: 0.6368175203353167 } },
  { kind: "cylinder", id: "tuft-near/blade-1", group: "leaf-reed", tags: ["reed", "bank"], place: { position: { x: 0.3493941243931958, y: 0.01113958800514229, z: 0.3760306132318647 }, anchor: "terrain" }, radius: 0.01190577592372149, halfHeight: 0.02313958800514228, material: { palette: "clay", value: 0.7593222357798368 } },
  { kind: "cylinder", id: "lily/pad-east", group: "lilypad-wet", tags: ["lily", "waterline"], place: { position: { x: 0.3, y: -0.011700000000000007, z: 0.0433 }, anchor: "floor" }, radius: 0.045000000000000005, halfHeight: 0.009, material: { palette: "clay", value: 0.76 } },
  { kind: "cylinder", id: "lily/pad-lobe", group: "lilypad-wet", tags: ["lily", "waterline"], place: { position: { x: 0.2833, y: -0.011700000000000007, z: 0.1567 }, anchor: "floor" }, radius: 0.038, halfHeight: 0.009, material: { palette: "clay", value: 0.72 } },
  { kind: "cylinder", id: "lily/pad-bridge", group: "lilypad-wet", tags: ["lily", "waterline"], place: { position: { x: 0.13, y: -0.011700000000000007, z: 0.2367 }, anchor: "floor" }, radius: 0.042, halfHeight: 0.009, material: { palette: "clay", value: 0.78 } },
  { kind: "cylinder", id: "lily/pad-west", group: "lilypad-wet", tags: ["lily", "waterline"], place: { position: { x: -0.3333, y: -0.011700000000000007, z: -0.0733 }, anchor: "floor" }, radius: 0.04, halfHeight: 0.009, material: { palette: "clay", value: 0.74 } },
  { kind: "cylinder", id: "lily/pad-beach", group: "lilypad-wet", tags: ["lily", "waterline"], place: { position: { x: 0.04, y: -0.011700000000000007, z: -0.2867 }, anchor: "floor" }, radius: 0.036, halfHeight: 0.009, material: { palette: "clay", value: 0.77 } },
  { kind: "ellipsoid", id: "lily/bud-east", group: "lilypad-wet", tags: ["lily", "waterline"], place: { position: { x: 0.31, y: 0.02629999999999999, z: 0.03 }, anchor: "floor" }, radius: { x: 0.024000000000000004, y: 0.036, z: 0.024000000000000004 }, material: { palette: "clay", value: 0.93 } },
  { kind: "ellipsoid", id: "lily/bud-west", group: "lilypad-wet", tags: ["lily", "waterline"], place: { position: { x: -0.34, y: 0.02429999999999999, z: -0.09000000000000001 }, anchor: "floor" }, radius: { x: 0.021, y: 0.032, z: 0.021 }, material: { palette: "clay", value: 0.91 } },
  { kind: "ellipsoid", id: "pebble-1/body", group: "stone-pebble", tags: ["pebble"], place: { position: { x: 0.21, y: 0.021080000000000005, z: -0.29 }, anchor: "terrain" }, radius: { x: 0.05000000000000001, y: 0.034, z: 0.044500000000000005 }, material: { palette: "stone", value: 0.68 } },
  { kind: "ellipsoid", id: "pebble-2/body", group: "stone-pebble", tags: ["pebble"], place: { position: { x: 0.26, y: 0.014880000000000004, z: -0.325 }, anchor: "terrain" }, radius: { x: 0.036, y: 0.024000000000000004, z: 0.03204 }, material: { palette: "stone", value: 0.74 } },
  { kind: "ellipsoid", id: "pebble-3/body", group: "stone-pebble", tags: ["pebble"], place: { position: { x: -0.05000000000000001, y: 0.018600000000000005, z: 0.31 }, anchor: "terrain" }, radius: { x: 0.045000000000000005, y: 0.03, z: 0.04005 }, material: { palette: "stone", value: 0.7 } },
  { kind: "cylinder", id: "stepping/bank-in", group: "stone-stepping", tags: ["stepping"], place: { position: { x: -0.0633, y: 0.01100000000000001, z: -0.2633 }, anchor: "terrain" }, radius: 0.045000000000000005, halfHeight: 0.011000000000000001, material: { palette: "stone", value: 0.73 } },
  { kind: "cylinder", id: "stepping/bank-out", group: "stone-stepping", tags: ["stepping"], place: { position: { x: 0.2833, y: 0.01099999999999999, z: -0.0833 }, anchor: "terrain" }, radius: 0.045000000000000005, halfHeight: 0.011000000000000001, material: { palette: "stone", value: 0.71 } },
];

export const gardenSceneryGraph: SceneryGraph = {
  palettes: { clay: CLAY, stone: STONE },
  nodes: GARDEN_NODES,
};

/**
 * Hose-filled tank (paper Figure 3), staged in a Victorian glasshouse.
 *
 * Everything here exists to explain one event: a hose fills a shallow tank. The
 * coil lies on the flagstones at the near left, uncoils toward the tank, climbs
 * a cane and stops dead at the left glass — exactly where the scene's jet
 * enters — so the eye reads hose, then jet, then rising level. The staging
 * bench sits behind the tank in the paper camera, which makes its row of pots
 * the backdrop the water is measured against.
 *
 * A single near-neutral grey ramp: limewashed timber, chalk, flagstone, zinc,
 * porcelain foliage. Only the sun panels and the pendant globes carry colour,
 * which is why they are the only literal materials in the set.
 */
const CONSERVATORY_NODES: readonly SceneryNode[] = [
  {
    kind: "room-shell", id: "shell", materialModel: "conservatory",
    floor: { colorLinear: [.40, .395, .385] },
    wall: { colorLinear: [1, 1, 1] },
    ceiling: { colorLinear: [.78, .785, .79] },
  },
  // The glass itself: six panes in the mullioned bay, declared beside the frame
  // that carries them rather than in the glass path's own table.
  { kind: "glazing", id: "glazing/pane-left-low", place: { position: { x: -.56, y: .2975, z: -1.48 } }, half: [.53, .2975] },
  { kind: "glazing", id: "glazing/pane-left-middle", place: { position: { x: -.56, y: .94, z: -1.48 } }, half: [.53, .295] },
  { kind: "glazing", id: "glazing/pane-left-high", place: { position: { x: -.56, y: 1.5625, z: -1.48 } }, half: [.53, .2775] },
  { kind: "glazing", id: "glazing/pane-right-low", place: { position: { x: .56, y: .2975, z: -1.48 } }, half: [.53, .2975] },
  { kind: "glazing", id: "glazing/pane-right-middle", place: { position: { x: .56, y: .94, z: -1.48 } }, half: [.53, .295] },
  { kind: "glazing", id: "glazing/pane-right-high", place: { position: { x: .56, y: 1.5625, z: -1.48 } }, half: [.53, .2775] },
  { kind: "box", id: "glazing/frame-0", group: "glazing-frame", tags: ["fixture", "frame"], place: { position: { x: -1.12, y: 0.9200000000000002, z: -1.48 } }, halfSize: { x: 0.027, y: 0.9200000000000002, z: 0.027 }, material: { palette: "lime", value: 1 } },
  { kind: "box", id: "glazing/frame-1", group: "glazing-frame", tags: ["fixture", "frame"], place: { position: { x: 0, y: 0.9200000000000002, z: -1.48 } }, halfSize: { x: 0.027, y: 0.9200000000000002, z: 0.027 }, material: { palette: "lime", value: 1 } },
  { kind: "box", id: "glazing/frame-2", group: "glazing-frame", tags: ["fixture", "frame"], place: { position: { x: 1.12, y: 0.9200000000000002, z: -1.48 } }, halfSize: { x: 0.027, y: 0.9200000000000002, z: 0.027 }, material: { palette: "lime", value: 1 } },
  { kind: "box", id: "glazing/rail-low", group: "glazing-frame", tags: [], place: { position: { x: 0, y: 0.62, z: -1.48 } }, halfSize: { x: 1.18, y: 0.025, z: 0.027 }, material: { palette: "lime", value: 1 } },
  { kind: "box", id: "glazing/rail-high", group: "glazing-frame", tags: [], place: { position: { x: 0, y: 1.26, z: -1.48 } }, halfSize: { x: 1.18, y: 0.025, z: 0.027 }, material: { palette: "lime", value: 1 } },
  { kind: "box", id: "glazing/mullion-1", group: "glazing-frame", tags: ["fixture", "frame"], place: { position: { x: -2.3, y: 0.95, z: -1.48 } }, halfSize: { x: 0.034, y: 0.95, z: 0.03 }, material: { palette: "lime", value: 0.9974509803921568 } },
  { kind: "box", id: "glazing/mullion-2", group: "glazing-frame", tags: ["fixture", "frame"], place: { position: { x: -1.7100000000000002, y: 0.95, z: -1.48 } }, halfSize: { x: 0.034, y: 0.95, z: 0.03 }, material: { palette: "lime", value: 1.0449019607843137 } },
  { kind: "box", id: "glazing/mullion-3", group: "glazing-frame", tags: ["fixture", "frame"], place: { position: { x: 1.7100000000000002, y: 0.95, z: -1.48 } }, halfSize: { x: 0.034, y: 0.95, z: 0.03 }, material: { palette: "lime", value: 0.9919607843137255 } },
  { kind: "box", id: "glazing/mullion-4", group: "glazing-frame", tags: ["fixture", "frame"], place: { position: { x: 2.3, y: 0.95, z: -1.48 } }, halfSize: { x: 0.034, y: 0.95, z: 0.03 }, material: { palette: "lime", value: 1.0398039215686274 } },
  { kind: "box", id: "glazing/rail-outer-low", group: "glazing-frame", tags: ["fixture", "frame"], place: { position: { x: 0, y: 0.62, z: -1.5250000000000001 } }, halfSize: { x: 2.42, y: 0.022, z: 0.022 }, material: { palette: "lime", value: 0.94 } },
  { kind: "box", id: "glazing/rail-outer-high", group: "glazing-frame", tags: ["fixture", "frame"], place: { position: { x: 0, y: 1.26, z: -1.5250000000000001 } }, halfSize: { x: 2.42, y: 0.022, z: 0.022 }, material: { palette: "lime", value: 0.94 } },
  { kind: "box", id: "glazing/sill", group: "glazing-frame", tags: ["fixture", "frame"], place: { position: { x: 0, y: 0.07, z: -1.48 }, anchor: "floor" }, halfSize: { x: 2.42, y: 0.07, z: 0.05500000000000001 }, material: { palette: "lime", value: 0.84 } },
  { kind: "box", id: "glazing/head", group: "glazing-frame", tags: ["fixture", "frame"], place: { position: { x: 0, y: 1.9, z: -1.48 } }, halfSize: { x: 2.42, y: 0.06, z: 0.05500000000000001 }, material: { palette: "lime", value: 0.9 } },
  { kind: "box", id: "sun/bay-left", group: "emissive-sky", tags: ["light", "emits-positive-z"], place: { position: { x: -0.58, y: 1.02, z: -1.61 } }, halfSize: { x: 0.5, y: 0.6, z: 0.022 }, material: { palette: "sun", value: 1, emission: 2.4 } },
  { kind: "box", id: "sun/bay-right", group: "emissive-sky", tags: ["light", "emits-positive-z"], place: { position: { x: 0.6, y: 0.8600000000000001, z: -1.61 } }, halfSize: { x: 0.4600000000000001, y: 0.5, z: 0.022 }, material: { palette: "sun", value: 0.94, emission: 1.5 } },
  { kind: "box", id: "bench/seat", group: "wood-bench", tags: ["bench"], place: { position: { x: -1.18, y: 0.6, z: -0.7 }, anchor: "floor" }, halfSize: { x: 0.62, y: 0.035, z: 0.24 }, material: { palette: "lime", value: 1 } },
  { kind: "box", id: "bench/back", group: "wood-bench", tags: ["bench"], place: { position: { x: -1.18, y: 0.8, z: -0.9200000000000002 }, anchor: "floor" }, halfSize: { x: 0.62, y: 0.16, z: 0.035 }, material: { palette: "lime", value: 0.88 } },
  { kind: "box", id: "bench/leg-left", group: "wood-bench", tags: ["bench"], place: { position: { x: -1.7000000000000002, y: 0.3, z: -0.8 }, anchor: "floor" }, halfSize: { x: 0.045, y: 0.3, z: 0.045 }, material: { palette: "lime", value: 0.72 } },
  { kind: "box", id: "bench/leg-right", group: "wood-bench", tags: ["bench"], place: { position: { x: -0.6599999999999999, y: 0.3, z: -0.8 }, anchor: "floor" }, halfSize: { x: 0.045, y: 0.3, z: 0.045 }, material: { palette: "lime", value: 0.72 } },
  { kind: "box", id: "bench/leg-front-left", group: "wood-bench", tags: ["bench"], place: { position: { x: -1.7000000000000002, y: 0.3, z: -0.56 }, anchor: "floor" }, halfSize: { x: 0.045, y: 0.3, z: 0.045 }, material: { palette: "lime", value: 0.66 } },
  { kind: "box", id: "bench/leg-front-right", group: "wood-bench", tags: ["bench"], place: { position: { x: -0.6599999999999999, y: 0.3, z: -0.56 }, anchor: "floor" }, halfSize: { x: 0.045, y: 0.3, z: 0.045 }, material: { palette: "lime", value: 0.66 } },
  { kind: "box", id: "bench/lower-shelf", group: "wood-bench", tags: ["bench"], place: { position: { x: -1.18, y: 0.22000000000000003, z: -0.7 }, anchor: "floor" }, halfSize: { x: 0.56, y: 0.022, z: 0.19 }, material: { palette: "lime", value: 0.6 } },
  { kind: "box", id: "bench/front-lip", group: "wood-bench", tags: ["bench"], place: { position: { x: -1.18, y: 0.665, z: -0.47 }, anchor: "floor" }, halfSize: { x: 0.62, y: 0.03, z: 0.022 }, material: { palette: "lime", value: 0.78 } },
  { kind: "cone", id: "bench/pot-a", group: "terracotta-pot", tags: ["pot"], place: { position: { x: -1.66, y: 0.76, z: -0.78 }, anchor: "floor" }, baseRadius: 0.0828, topRadius: 0.11500000000000002, halfHeight: 0.125, material: { palette: "clay", value: 0.995921568627451 } },
  { kind: "cone", id: "bench/pot-b", group: "terracotta-pot", tags: ["pot"], place: { position: { x: -1.4, y: 0.725, z: -0.62 }, anchor: "floor" }, baseRadius: 0.061200000000000004, topRadius: 0.085, halfHeight: 0.09, material: { palette: "clay", value: 1.071843137254902 } },
  { kind: "cone", id: "bench/pot-c", group: "terracotta-pot", tags: ["pot"], place: { position: { x: -0.98, y: 0.775, z: -0.74 }, anchor: "floor" }, baseRadius: 0.09, topRadius: 0.125, halfHeight: 0.14, material: { palette: "clay", value: 0.9871372549019608 } },
  { kind: "cone", id: "bench/pot-d", group: "terracotta-pot", tags: ["pot"], place: { position: { x: -0.8, y: 0.71, z: -0.8800000000000001 }, anchor: "floor" }, baseRadius: 0.0504, topRadius: 0.07, halfHeight: 0.075, material: { palette: "clay", value: 1.063686274509804 } },
  { kind: "cone", id: "bench/pot-upturned", group: "terracotta-pot", tags: ["pot"], place: { position: { x: -1.19, y: 0.715, z: -0.8600000000000001 }, anchor: "floor" }, baseRadius: 0.10500000000000001, topRadius: 0.0756, halfHeight: 0.08, material: { palette: "clay", value: 0.82 } },
  { kind: "box", id: "bench/seed-tray", group: "wood-bench", tags: ["bench"], place: { position: { x: -1.36, y: 0.667, z: -0.545 }, anchor: "floor" }, halfSize: { x: 0.2, y: 0.032, z: 0.10500000000000001 }, material: { palette: "chalk", value: 0.92 } },
  { kind: "ellipsoid", id: "bench/seedling-a", group: "porcelain-foliage", tags: ["plant"], place: { position: { x: -1.66, y: 0.9349999999999999, z: -0.78 }, anchor: "floor" }, radius: { x: 0.11500000000000002, y: 0.09, z: 0.1 }, material: { palette: "porcelain", value: 1 } },
  { kind: "ellipsoid", id: "bench/seedling-b", group: "porcelain-foliage", tags: ["plant"], place: { position: { x: -0.98, y: 0.975, z: -0.74 }, anchor: "floor" }, radius: { x: 0.135, y: 0.11000000000000001, z: 0.12 }, material: { palette: "porcelain", value: 0.9 } },
  { kind: "cylinder", id: "bench/can-body", group: "zinc-metal", tags: ["watering-can"], place: { position: { x: -0.8600000000000001, y: 0.74, z: -0.8 }, anchor: "floor" }, radius: 0.10500000000000001, halfHeight: 0.10500000000000001, material: { palette: "zinc", value: 1 } },
  { kind: "box", id: "bench/can-spout", group: "zinc-metal", tags: ["watering-can"], place: { position: { x: -0.68, y: 0.765, z: -0.8 }, anchor: "floor" }, halfSize: { x: 0.1, y: 0.028, z: 0.028 }, material: { palette: "zinc", value: 0.86 } },
  { kind: "box", id: "bench/can-handle", group: "zinc-metal", tags: ["watering-can"], place: { position: { x: -0.8600000000000001, y: 0.8799999999999999, z: -0.8 }, anchor: "floor" }, halfSize: { x: 0.035, y: 0.05500000000000001, z: 0.085 }, material: { palette: "zinc", value: 0.74 } },
  { kind: "box", id: "planter/stone", group: "stone-planter", tags: ["planter"], place: { position: { x: 1.12, y: 0.26, z: -0.8600000000000001 }, anchor: "floor" }, halfSize: { x: 0.28, y: 0.26, z: 0.28 }, material: { palette: "chalk", value: 1 } },
  { kind: "box", id: "planter/plinth", group: "stone-planter", tags: ["planter"], place: { position: { x: 1.12, y: 0.035, z: -0.8600000000000001 }, anchor: "floor" }, halfSize: { x: 0.315, y: 0.035, z: 0.315 }, material: { palette: "chalk", value: 0.78 } },
  { kind: "box", id: "planter/rim", group: "stone-planter", tags: ["planter"], place: { position: { x: 1.12, y: 0.545, z: -0.8600000000000001 }, anchor: "floor" }, halfSize: { x: 0.315, y: 0.035, z: 0.315 }, material: { palette: "chalk", value: 1.08 } },
  { kind: "cylinder", id: "planter/stem", group: "porcelain-foliage", tags: ["plant"], place: { position: { x: 1.12, y: 0.72, z: -0.8600000000000001 }, anchor: "floor" }, radius: 0.048, halfHeight: 0.18, material: { palette: "porcelain", value: 0.74 } },
  { kind: "ellipsoid", id: "planter/foliage-main", group: "porcelain-foliage", tags: ["plant"], place: { position: { x: 1.12, y: 0.9000000000000001, z: -0.8600000000000001 }, anchor: "floor" }, radius: { x: 0.4, y: 0.34, z: 0.3 }, material: { palette: "porcelain", value: 1 } },
  { kind: "ellipsoid", id: "planter/foliage-left", group: "porcelain-foliage", tags: ["plant"], place: { position: { x: 0.8800000000000001, y: 1.08, z: -0.8400000000000001 }, anchor: "floor" }, radius: { x: 0.24, y: 0.34, z: 0.19 }, material: { palette: "porcelain", value: 0.84 } },
  { kind: "ellipsoid", id: "planter/foliage-right", group: "porcelain-foliage", tags: ["plant"], place: { position: { x: 1.36, y: 1.14, z: -0.9000000000000001 }, anchor: "floor" }, radius: { x: 0.22000000000000003, y: 0.38, z: 0.18 }, material: { palette: "porcelain", value: 0.7 } },
  { kind: "ellipsoid", id: "planter/foliage-crown", group: "porcelain-foliage", tags: ["plant"], place: { position: { x: 1.1, y: 1.34, z: -0.8800000000000001 }, anchor: "floor" }, radius: { x: 0.16, y: 0.22000000000000003, z: 0.15 }, material: { palette: "porcelain", value: 0.96 } },
  { kind: "box", id: "trough/body", group: "zinc-metal", tags: ["trough"], place: { position: { x: 1.52, y: 0.3, z: -0.06 }, anchor: "floor" }, halfSize: { x: 0.3, y: 0.16, z: 0.52 }, material: { palette: "zinc", value: 1 } },
  { kind: "box", id: "trough/rim-left", group: "zinc-metal", tags: ["trough"], place: { position: { x: 1.22, y: 0.47, z: -0.06 }, anchor: "floor" }, halfSize: { x: 0.035, y: 0.03, z: 0.55 }, material: { palette: "zinc", value: 1.14 } },
  { kind: "box", id: "trough/rim-right", group: "zinc-metal", tags: ["trough"], place: { position: { x: 1.8199999999999998, y: 0.47, z: -0.06 }, anchor: "floor" }, halfSize: { x: 0.035, y: 0.03, z: 0.55 }, material: { palette: "zinc", value: 1.14 } },
  { kind: "box", id: "trough/rim-front", group: "zinc-metal", tags: ["trough"], place: { position: { x: 1.52, y: 0.47, z: -0.58 }, anchor: "floor" }, halfSize: { x: 0.33, y: 0.03, z: 0.035 }, material: { palette: "zinc", value: 1.06 } },
  { kind: "box", id: "trough/rim-back", group: "zinc-metal", tags: ["trough"], place: { position: { x: 1.52, y: 0.47, z: 0.4600000000000001 }, anchor: "floor" }, halfSize: { x: 0.33, y: 0.03, z: 0.035 }, material: { palette: "zinc", value: 1.06 } },
  { kind: "box", id: "trough/trestle-front", group: "wood-bench", tags: ["trough"], place: { position: { x: 1.52, y: 0.07, z: -0.43999999999999995 }, anchor: "floor" }, halfSize: { x: 0.24, y: 0.07, z: 0.05 }, material: { palette: "lime", value: 0.62 } },
  { kind: "box", id: "trough/trestle-back", group: "wood-bench", tags: ["trough"], place: { position: { x: 1.52, y: 0.07, z: 0.31999999999999995 }, anchor: "floor" }, halfSize: { x: 0.24, y: 0.07, z: 0.05 }, material: { palette: "lime", value: 0.62 } },
  { kind: "torus", id: "hose/coil-outer", group: "rubber-hose", tags: ["hose"], place: { position: { x: -1.02, y: 0.05500000000000001, z: 0.82 }, anchor: "floor" }, majorRadius: 0.32, minorRadius: 0.05500000000000001, material: { palette: "rubber", value: 1 } },
  { kind: "torus", id: "hose/coil-inner", group: "rubber-hose", tags: ["hose"], place: { position: { x: -1.02, y: 0.145, z: 0.82 }, anchor: "floor" }, majorRadius: 0.19, minorRadius: 0.05, material: { palette: "rubber", value: 1.35 } },
  { kind: "capsule", id: "hose/run-1", group: "rubber-hose", tags: ["hose"], place: { position: { x: 0, y: 0, z: 0 }, anchor: "floor" }, from: { x: -1.02, y: 0.049999999999999975, z: 0.5 }, to: { x: -0.9000000000000002, y: 0.050000000000000024, z: 0.3 }, radius: 0.05, material: { palette: "rubber", value: 1.2404411764705883 } },
  { kind: "capsule", id: "hose/run-2", group: "rubber-hose", tags: ["hose"], place: { position: { x: 0, y: 0, z: 0 }, anchor: "floor" }, from: { x: -0.9000000000000001, y: 0.049999999999999975, z: 0.3 }, to: { x: -0.74, y: 0.050000000000000024, z: 0.07999999999999999 }, radius: 0.05, material: { palette: "rubber", value: 1.4183823529411765 } },
  { kind: "capsule", id: "hose/run-3", group: "rubber-hose", tags: ["hose"], place: { position: { x: 0, y: 0, z: 0 }, anchor: "floor" }, from: { x: -0.7400000000000001, y: 0.05000000000000001, z: 0.08000000000000002 }, to: { x: -0.66, y: 0.04999999999999999, z: 0.04 }, radius: 0.05, material: { colorLinear: [0.0914889705882353, 0.0951485294117647, 0.1000279411764706] } },
  // hose/riser spanned floor and absolute datums; authored against the floor.
  { kind: "capsule", id: "hose/riser", group: "rubber-hose", tags: ["hose"], place: { position: { x: 0, y: 0, z: 0 }, anchor: "floor" }, from: { x: -0.66, y: 0.05000000000000003, z: 0.04 }, to: { x: -0.615, y: 0.4808333333333334, z: 0.04 }, radius: 0.05, material: { palette: "rubber", value: 1.5 } },
  { kind: "cylinder", id: "hose/cane", group: "wood-bench", tags: ["hose"], place: { position: { x: -0.68, y: 0.36, z: 0.1 }, anchor: "floor", orientation: { w: 0.9993399039968005, x: -0.009980192759758566, y: 0, z: -0.03493067465915498 } }, radius: 0.022, halfHeight: 0.36, material: { palette: "lime", value: 0.7 } },
  { kind: "box", id: "hose/nozzle-arm", group: "zinc-metal", tags: ["hose", "fixture"], place: { position: { x: -0.585, y: 0.4600000000000001, z: 0.04 } }, halfSize: { x: 0.07, y: 0.045, z: 0.045 }, material: { palette: "zinc", value: 0.92 } },
  { kind: "box", id: "hose/nozzle-collar", group: "zinc-metal", tags: ["hose", "fixture"], place: { position: { x: -0.545, y: 0.4600000000000001, z: 0.04 } }, halfSize: { x: 0.033, y: 0.03, z: 0.03 }, material: { palette: "zinc", value: 1.2 } },
  { kind: "cylinder", id: "can/body", group: "zinc-metal", tags: ["watering-can"], place: { position: { x: 1.08, y: 0.17, z: 0.42000000000000004 }, anchor: "floor" }, radius: 0.17, halfHeight: 0.17, material: { palette: "zinc", value: 0.96 } },
  { kind: "cone", id: "can/shoulder", group: "zinc-metal", tags: ["watering-can"], place: { position: { x: 1.08, y: 0.4, z: 0.42000000000000004 }, anchor: "floor" }, baseRadius: 0.155, topRadius: 0.10500000000000001, halfHeight: 0.06, material: { palette: "zinc", value: 1.12 } },
  { kind: "box", id: "can/spout", group: "zinc-metal", tags: ["watering-can"], place: { position: { x: 0.82, y: 0.3, z: 0.42000000000000004 }, anchor: "floor" }, halfSize: { x: 0.16, y: 0.035, z: 0.035 }, material: { palette: "zinc", value: 0.84 } },
  { kind: "ellipsoid", id: "can/rose", group: "zinc-metal", tags: ["watering-can"], place: { position: { x: 0.665, y: 0.3, z: 0.42000000000000004 }, anchor: "floor" }, radius: { x: 0.06, y: 0.06, z: 0.06 }, material: { palette: "zinc", value: 1.22 } },
  { kind: "box", id: "can/handle", group: "zinc-metal", tags: ["watering-can"], place: { position: { x: 1.08, y: 0.48, z: 0.42000000000000004 }, anchor: "floor" }, halfSize: { x: 0.045, y: 0.09, z: 0.13 }, material: { palette: "zinc", value: 0.78 } },
  { kind: "cone", id: "pots/floor-a", group: "terracotta-pot", tags: ["pot"], place: { position: { x: -1.7200000000000002, y: 0.19, z: -0.3 }, anchor: "floor" }, baseRadius: 0.11520000000000001, topRadius: 0.16, halfHeight: 0.19, material: { colorLinear: [0.6377901960784315, 0.6037098039215686, 0.5696294117647058] } },
  { kind: "cone", id: "pots/floor-b", group: "terracotta-pot", tags: ["pot"], place: { position: { x: -1.44, y: 0.145, z: -0.14 }, anchor: "floor" }, baseRadius: 0.08639999999999999, topRadius: 0.12, halfHeight: 0.145, material: { palette: "clay", value: 1.0686274509803921 } },
  { kind: "cone", id: "pots/floor-c", group: "terracotta-pot", tags: ["pot"], place: { position: { x: -1.62, y: 0.1, z: 0.12 }, anchor: "floor" }, baseRadius: 0.0936, topRadius: 0.13, halfHeight: 0.1, material: { palette: "clay", value: 0.9635294117647059 } },
  { kind: "cone", id: "pots/floor-d", group: "terracotta-pot", tags: ["pot"], place: { position: { x: -1.34, y: 0.16, z: 0.34 }, anchor: "floor" }, baseRadius: 0.0756, topRadius: 0.10500000000000001, halfHeight: 0.16, material: { palette: "clay", value: 1.0584313725490198 } },
  { kind: "cone", id: "pots/stack-base", group: "terracotta-pot", tags: ["pot"], place: { position: { x: -1.8500000000000003, y: 0.085, z: 1.25 }, anchor: "floor" }, baseRadius: 0.18, topRadius: 0.1296, halfHeight: 0.085, material: { palette: "clay", value: 1.0530588235294118 } },
  { kind: "cone", id: "pots/stack-mid", group: "terracotta-pot", tags: ["pot"], place: { position: { x: -1.8500000000000003, y: 0.235, z: 1.25 }, anchor: "floor" }, baseRadius: 0.155, topRadius: 0.11159999999999999, halfHeight: 0.075, material: { palette: "clay", value: 0.9365882352941176 } },
  { kind: "cone", id: "pots/stack-top", group: "terracotta-pot", tags: ["pot"], place: { position: { x: -1.8500000000000003, y: 0.375, z: 1.25 }, anchor: "floor" }, baseRadius: 0.13, topRadius: 0.0936, halfHeight: 0.065, material: { palette: "clay", value: 1.0409803921568628 } },
  { kind: "box", id: "paving/slab-1", group: "flagstone-paving", tags: ["paving"], place: { position: { x: -1.95, y: 0.03, z: -0.75 }, anchor: "floor" }, halfSize: { x: 0.3, y: 0.03, z: 0.3 }, material: { colorLinear: [0.38908078431372545, 0.38465941176470586, 0.37581666666666663] } },
  { kind: "box", id: "paving/slab-2", group: "flagstone-paving", tags: ["paving"], place: { position: { x: -1.35, y: 0.03, z: -0.15 }, anchor: "floor" }, halfSize: { x: 0.3, y: 0.03, z: 0.3 }, material: { palette: "stone", value: 1.0076470588235293 } },
  { kind: "box", id: "paving/slab-3", group: "flagstone-paving", tags: ["paving"], place: { position: { x: -1.8500000000000003, y: 0.03, z: 0.45000000000000007 }, anchor: "floor" }, halfSize: { x: 0.3, y: 0.03, z: 0.3 }, material: { palette: "stone", value: 0.87 } },
  { kind: "box", id: "paving/slab-4", group: "flagstone-paving", tags: ["paving"], place: { position: { x: -1.25, y: 0.03, z: 1.05 }, anchor: "floor" }, halfSize: { x: 0.3, y: 0.03, z: 0.3 }, material: { palette: "stone", value: 0.9943921568627451 } },
  { kind: "box", id: "paving/slab-5", group: "flagstone-paving", tags: ["paving"], place: { position: { x: -0.8500000000000001, y: 0.03, z: 0.3 }, anchor: "floor" }, halfSize: { x: 0.3, y: 0.03, z: 0.3 }, material: { palette: "stone", value: 1.117764705882353 } },
  { kind: "box", id: "paving/slab-6", group: "flagstone-paving", tags: ["paving"], place: { position: { x: -1.8000000000000003, y: 0.03, z: 1.3 }, anchor: "floor" }, halfSize: { x: 0.3, y: 0.03, z: 0.3 }, material: { palette: "stone", value: 0.9801176470588235 } },
  { kind: "box", id: "paving/slab-7", group: "flagstone-paving", tags: ["paving"], place: { position: { x: -2.25, y: 0.03, z: 0.35 }, anchor: "floor" }, halfSize: { x: 0.3, y: 0.03, z: 0.3 }, material: { palette: "stone", value: 1.1045098039215686 } },
  { kind: "box", id: "paving/slab-8", group: "flagstone-paving", tags: ["paving"], place: { position: { x: 0.9000000000000001, y: 0.03, z: -0.35 }, anchor: "floor" }, halfSize: { x: 0.3, y: 0.03, z: 0.3 }, material: { colorLinear: [0.4254196078431372, 0.42058529411764706, 0.41091666666666665] } },
  { kind: "box", id: "paving/slab-9", group: "flagstone-paving", tags: ["paving"], place: { position: { x: 1.5, y: 0.03, z: -0.95 }, anchor: "floor" }, halfSize: { x: 0.3, y: 0.03, z: 0.3 }, material: { palette: "stone", value: 1.090235294117647 } },
  { kind: "box", id: "paving/slab-10", group: "flagstone-paving", tags: ["paving"], place: { position: { x: -0.3, y: 0.03, z: 1.15 }, anchor: "floor" }, halfSize: { x: 0.3, y: 0.03, z: 0.3 }, material: { palette: "stone", value: 0.9536078431372549 } },
  { kind: "box", id: "topiary/plinth", group: "stone-planter", tags: ["planter"], place: { position: { x: -2.15, y: 0.3, z: -1.05 }, anchor: "floor" }, halfSize: { x: 0.2, y: 0.3, z: 0.2 }, material: { palette: "chalk", value: 0.86 } },
  { kind: "ellipsoid", id: "topiary/ball", group: "porcelain-foliage", tags: ["plant"], place: { position: { x: -2.15, y: 0.78, z: -1.05 }, anchor: "floor" }, radius: { x: 0.26, y: 0.24, z: 0.26 }, material: { palette: "porcelain", value: 0.92 } },
  { kind: "ellipsoid", id: "topiary/finial", group: "porcelain-foliage", tags: ["plant"], place: { position: { x: -2.15, y: 1.02, z: -1.05 }, anchor: "floor" }, radius: { x: 0.15, y: 0.14, z: 0.15 }, material: { palette: "porcelain", value: 1 } },
  { kind: "cylinder", id: "pendant-0/cord", group: "fixture-cord", tags: ["fixture"], place: { position: { x: -0.56, y: 1.5, z: -1.04 } }, radius: 0.012, halfHeight: 0.34, material: { palette: "slate", value: 1 } },
  { kind: "ellipsoid", id: "pendant-0/globe", group: "emissive-glass", tags: ["fixture", "light"], place: { position: { x: -0.56, y: 1.18, z: -1.04 } }, radius: { x: 0.095, y: 0.095, z: 0.095 }, material: { colorLinear: [0.85, 0.68, 0.38], emission: 0.48 } },
  { kind: "cylinder", id: "pendant-1/cord", group: "fixture-cord", tags: ["fixture"], place: { position: { x: 0, y: 1.5, z: -1.04 } }, radius: 0.012, halfHeight: 0.34, material: { palette: "slate", value: 1 } },
  { kind: "ellipsoid", id: "pendant-1/globe", group: "emissive-glass", tags: ["fixture", "light"], place: { position: { x: 0, y: 1.18, z: -1.04 } }, radius: { x: 0.095, y: 0.095, z: 0.095 }, material: { colorLinear: [0.85, 0.68, 0.38], emission: 0.48 } },
  { kind: "cylinder", id: "pendant-2/cord", group: "fixture-cord", tags: ["fixture"], place: { position: { x: 0.56, y: 1.5, z: -1.04 } }, radius: 0.012, halfHeight: 0.34, material: { palette: "slate", value: 1 } },
  { kind: "ellipsoid", id: "pendant-2/globe", group: "emissive-glass", tags: ["fixture", "light"], place: { position: { x: 0.56, y: 1.18, z: -1.04 } }, radius: { x: 0.095, y: 0.095, z: 0.095 }, material: { colorLinear: [0.85, 0.68, 0.38], emission: 0.48 } },
];

export const conservatorySceneryGraph: SceneryGraph = {
  palettes: {
    lime: { tint: [.84, .83, .805] }, chalk: { tint: [.70, .695, .68] },
    clay: { tint: [.655, .620, .585] }, stone: { tint: [.44, .435, .425] },
    slate: { tint: [.255, .26, .27] }, zinc: { tint: [.53, .545, .565] },
    rubber: { tint: [.075, .078, .082] }, porcelain: { tint: [.86, .865, .85] },
    sun: { tint: [1, .88, .62] },
  },
  nodes: CONSERVATORY_NODES,
};

/**
 * Jet past a sphere (paper Figure 6), staged in a night laboratory.
 *
 * Near-neutral greys separated by value alone, with hue reserved for emitters:
 * one warm tungsten source and the cool family it plays against. The back wall
 * carries a city window, declared as an opening so the four boxes around it are
 * derived from the room rather than frozen at one container size — a single
 * union-only wall box would conceal the authored thin-glass pane behind it.
 */
const NIGHT_LAB_NODES: readonly SceneryNode[] = [
  {
    kind: "room-shell", id: "shell", materialModel: "night-lab",
    floor: { colorLinear: [.172, .178, .186] },
    wall: { colorLinear: [1, 1, 1] },
    ceiling: { colorLinear: [.455, .458, .462] },
    backWall: { halfWidth: 1.62, halfHeight: .55, centerY: 1.60, glazing: "window/city-glazing" },
  },
  { kind: "box", id: "desk/top", group: "lab-bench", tags: ["desk", "bench"], place: { position: { x: 0, y: -0.021, z: 0 } }, halfSize: { x: 0.8, y: 0.019, z: 0.5933333333333334 }, material: { colorLinear: [0.6, 0.6, 0.592] } },
  { kind: "box", id: "desk/apron", group: "lab-bench", tags: ["desk", "bench"], place: { position: { x: 0, y: -0.074, z: 0 } }, halfSize: { x: 0.73, y: 0.034, z: 0.5233333333333333 }, material: { palette: "graphite", value: 1 } },
  { kind: "box", id: "desk/leg-lf", group: "steel-frame", tags: ["desk", "bench"], place: { position: { x: -0.7, y: 0.34, z: -0.49333333333333335 }, anchor: "floor" }, halfSize: { x: 0.027, y: 0.34, z: 0.027 }, material: { palette: "steel", value: 1 } },
  { kind: "box", id: "desk/leg-lb", group: "steel-frame", tags: ["desk", "bench"], place: { position: { x: -0.7, y: 0.34, z: 0.49333333333333335 }, anchor: "floor" }, halfSize: { x: 0.027, y: 0.34, z: 0.027 }, material: { palette: "steel", value: 1 } },
  { kind: "box", id: "desk/leg-rf", group: "steel-frame", tags: ["desk", "bench"], place: { position: { x: 0.7, y: 0.34, z: -0.49333333333333335 }, anchor: "floor" }, halfSize: { x: 0.027, y: 0.34, z: 0.027 }, material: { palette: "steel", value: 1 } },
  { kind: "box", id: "desk/leg-rb", group: "steel-frame", tags: ["desk", "bench"], place: { position: { x: 0.7, y: 0.34, z: 0.49333333333333335 }, anchor: "floor" }, halfSize: { x: 0.027, y: 0.34, z: 0.027 }, material: { palette: "steel", value: 1 } },
  { kind: "box", id: "desk/lower-shelf", group: "steel-frame", tags: ["desk", "bench"], place: { position: { x: 0, y: 0.16000000000000006, z: 0 }, anchor: "floor" }, halfSize: { x: 0.6699999999999999, y: 0.013, z: 0.4633333333333333 }, material: { palette: "slate", value: 1 } },
  { kind: "box", id: "desk/controller", group: "instrument", tags: ["desk", "instrument"], place: { position: { x: -0.27999999999999997, y: 0.23499999999999993, z: 0.089 }, anchor: "floor" }, halfSize: { x: 0.15, y: 0.062, z: 0.11500000000000002 }, material: { palette: "pewter", value: 1 } },
  { kind: "box", id: "desk/mat", group: "floor-mat", tags: ["desk", "floor"], place: { position: { x: 0, y: 0.005999999999999987, z: 0.18 }, anchor: "floor" }, halfSize: { x: 1.15, y: 0.006, z: 0.95 }, material: { colorLinear: [0.115, 0.119, 0.124] } },
  { kind: "cylinder", id: "desk-lamp/base", group: "metal-fixture", tags: ["desk", "fixture"], place: { position: { x: -0.63, y: 0.01, z: -0.35333333333333333 } }, radius: 0.085, halfHeight: 0.013, material: { palette: "ink", value: 1 } },
  { kind: "cylinder", id: "desk-lamp/stem", group: "metal-fixture", tags: ["desk", "fixture"], place: { position: { x: -0.63, y: 0.31, z: -0.35333333333333333 } }, radius: 0.012, halfHeight: 0.29, material: { palette: "graphite", value: 1 } },
  { kind: "ellipsoid", id: "desk-lamp/shade", group: "metal-fixture", tags: ["desk", "fixture"], place: { position: { x: -0.63, y: 0.63, z: -0.35333333333333333 } }, radius: { x: 0.104, y: 0.076, z: 0.104 }, material: { colorLinear: [0.05, 0.051, 0.054] } },
  { kind: "ellipsoid", id: "desk-lamp/bulb", group: "emissive-glass", tags: ["desk", "fixture", "light"], place: { position: { x: -0.63, y: 0.58, z: -0.35333333333333333 } }, radius: { x: 0.048, y: 0.048, z: 0.048 }, material: { palette: "tungsten", value: 1, emission: 3.4 } },
  { kind: "cylinder", id: "stool/seat", group: "stool-seat", tags: ["stool", "chair"], place: { position: { x: 1.25, y: 0.47, z: 0.1 }, anchor: "floor" }, radius: 0.17, halfHeight: 0.024, material: { palette: "bone", value: 1 } },
  { kind: "cylinder", id: "stool/post", group: "steel-frame", tags: ["stool", "chair"], place: { position: { x: 1.25, y: 0.23499999999999993, z: 0.1 }, anchor: "floor" }, radius: 0.024, halfHeight: 0.21500000000000002, material: { palette: "steel", value: 0.8 } },
  { kind: "cylinder", id: "stool/base", group: "steel-frame", tags: ["stool", "chair"], place: { position: { x: 1.25, y: 0.020000000000000018, z: 0.1 }, anchor: "floor" }, radius: 0.145, halfHeight: 0.014, material: { palette: "steel", value: 0.6 } },
  { kind: "box", id: "counter/cabinet", group: "lab-counter", tags: ["counter"], place: { position: { x: 0, y: 0.42000000000000004, z: -1.89 }, anchor: "floor" }, halfSize: { x: 1.7200000000000002, y: 0.42000000000000004, z: 0.3 }, material: { palette: "slate", value: 1 } },
  { kind: "box", id: "counter/worktop", group: "lab-counter", tags: ["counter"], place: { position: { x: 0, y: 0.862, z: -1.89 }, anchor: "floor" }, halfSize: { x: 1.8000000000000003, y: 0.022, z: 0.34 }, material: { palette: "ash", value: 1 } },
  { kind: "box", id: "counter/monitor-stand", group: "metal-fixture", tags: ["counter", "instrument"], place: { position: { x: 0.95, y: 0.9300000000000002, z: -1.89 }, anchor: "floor" }, halfSize: { x: 0.05, y: 0.05500000000000001, z: 0.05 }, material: { palette: "ink", value: 1 } },
  { kind: "box", id: "counter/monitor", group: "monitor-frame", tags: ["counter", "instrument", "monitor"], place: { position: { x: 0.95, y: 1.17, z: -1.8399999999999999 }, anchor: "floor" }, halfSize: { x: 0.3, y: 0.19, z: 0.014 }, material: { colorLinear: [0.03, 0.034, 0.04] } },
  { kind: "box", id: "counter/monitor-screen", group: "monitor-glass", tags: ["counter", "instrument", "monitor", "light", "emits-positive-z"], place: { position: { x: 0.95, y: 1.17, z: -1.822 }, anchor: "floor" }, halfSize: { x: 0.265, y: 0.155, z: 0.004 }, material: { palette: "displayCool", value: 1, emission: 1.35 } },
  { kind: "box", id: "counter/keyboard", group: "instrument", tags: ["counter", "instrument"], place: { position: { x: 0.95, y: 0.892, z: -1.6699999999999997 }, anchor: "floor" }, halfSize: { x: 0.19, y: 0.007, z: 0.07 }, material: { palette: "graphite", value: 1 } },
  { kind: "cylinder", id: "counter/instrument-a", group: "instrument", tags: ["counter", "instrument", "emissive-surface-only"], place: { position: { x: -0.58, y: 0.9660000000000001, z: -1.89 }, anchor: "floor" }, radius: 0.07, halfHeight: 0.082, material: { palette: "pewter", value: 1, emission: 0.02 } },
  { kind: "cylinder", id: "counter/instrument-b", group: "instrument", tags: ["counter", "instrument", "emissive-surface-only"], place: { position: { x: -0.8600000000000001, y: 1.014, z: -1.8499999999999999 }, anchor: "floor" }, radius: 0.046, halfHeight: 0.13, material: { palette: "pewter", value: 0.86, emission: 0.02 } },
  { kind: "ellipsoid", id: "counter/instrument-c", group: "instrument", tags: ["counter", "instrument", "emissive-surface-only"], place: { position: { x: -1.12, y: 0.98, z: -1.91 }, anchor: "floor" }, radius: { x: 0.088, y: 0.096, z: 0.088 }, material: { palette: "pewter", value: 0.94, emission: 0.02 } },
  { kind: "box", id: "counter/shelf", group: "metal-fixture", tags: ["counter", "shelf"], place: { position: { x: -0.2, y: 1.62, z: -1.94 }, anchor: "floor" }, halfSize: { x: 1.22, y: 0.016, z: 0.17 }, material: { palette: "steel", value: 1 } },
  { kind: "box", id: "counter/shelf-bottle-1", group: "instrument", tags: ["counter", "shelf", "instrument"], place: { position: { x: -0.9200000000000002, y: 1.7510000000000001, z: -1.94 }, anchor: "floor" }, halfSize: { x: 0.05, y: 0.11500000000000002, z: 0.135 }, material: { palette: "chalk", value: 1 } },
  { kind: "box", id: "counter/shelf-bottle-2", group: "instrument", tags: ["counter", "shelf", "instrument"], place: { position: { x: -0.79, y: 1.7510000000000001, z: -1.94 }, anchor: "floor" }, halfSize: { x: 0.05, y: 0.11500000000000002, z: 0.135 }, material: { colorLinear: [0.3, 0.305, 0.31] } },
  { kind: "box", id: "counter/shelf-bottle-3", group: "instrument", tags: ["counter", "shelf", "instrument"], place: { position: { x: -0.66, y: 1.7510000000000001, z: -1.94 }, anchor: "floor" }, halfSize: { x: 0.05, y: 0.11500000000000002, z: 0.135 }, material: { palette: "bone", value: 1 } },
  { kind: "box", id: "fixtures/troffer-left-1", group: "emissive-fixture", tags: ["fixture", "light", "emits-negative-y"], place: { position: { x: -0.3, y: 2.14, z: 0.14 }, anchor: "floor" }, halfSize: { x: 0.78, y: 0.013, z: 0.13 }, material: { palette: "coolBar", value: 1, emission: 2.4 } },
  { kind: "box", id: "fixtures/troffer-left-2", group: "emissive-fixture", tags: ["fixture", "light", "emits-negative-y"], place: { position: { x: -1.35, y: 3.5650000000000004, z: -0.4 }, anchor: "floor" }, halfSize: { x: 0.55, y: 0.012, z: 0.2 }, material: { palette: "coolBar", value: 1, emission: 2.4 } },
  { kind: "box", id: "fixtures/troffer-right-1", group: "emissive-fixture", tags: ["fixture", "light", "emits-negative-y"], place: { position: { x: 0.34, y: 2.14, z: -0.32 }, anchor: "floor" }, halfSize: { x: 0.72, y: 0.013, z: 0.13 }, material: { palette: "coolBar", value: 1, emission: 2.4 } },
  { kind: "box", id: "fixtures/troffer-right-2", group: "emissive-fixture", tags: ["fixture", "light", "emits-negative-y"], place: { position: { x: 1.35, y: 3.5650000000000004, z: 0.7 }, anchor: "floor" }, halfSize: { x: 0.55, y: 0.012, z: 0.2 }, material: { palette: "coolBar", value: 1, emission: 2.4 } },
  { kind: "cylinder", id: "rig/pendant-rod-1", group: "metal-fixture", tags: ["fixture"], place: { position: { x: -0.8800000000000001, y: 2.87, z: 0.14 }, anchor: "floor" }, radius: 0.01, halfHeight: 0.7300000000000001, material: { palette: "steel", value: 1 } },
  { kind: "cylinder", id: "rig/pendant-rod-2", group: "metal-fixture", tags: ["fixture"], place: { position: { x: 0.28, y: 2.87, z: 0.14 }, anchor: "floor" }, radius: 0.01, halfHeight: 0.7300000000000001, material: { palette: "steel", value: 1 } },
  { kind: "cylinder", id: "rig/pendant-rod-3", group: "metal-fixture", tags: ["fixture"], place: { position: { x: -0.2, y: 2.87, z: -0.32 }, anchor: "floor" }, radius: 0.01, halfHeight: 0.7300000000000001, material: { palette: "steel", value: 1 } },
  { kind: "cylinder", id: "rig/pendant-rod-4", group: "metal-fixture", tags: ["fixture"], place: { position: { x: 0.8800000000000001, y: 2.87, z: -0.32 }, anchor: "floor" }, radius: 0.01, halfHeight: 0.7300000000000001, material: { palette: "steel", value: 1 } },
  { kind: "cylinder", id: "rig/tripod-leg-a", group: "tripod-steel", tags: ["rig", "instrument"], place: { position: { x: -0.22000000000000003, y: -0.31, z: 1.17 } }, radius: 0.02, halfHeight: 0.41, material: { palette: "steel", value: 1 } },
  { kind: "cylinder", id: "rig/tripod-leg-b", group: "tripod-steel", tags: ["rig", "instrument"], place: { position: { x: 0.18, y: -0.31, z: 1.17 } }, radius: 0.02, halfHeight: 0.41, material: { palette: "steel", value: 1 } },
  { kind: "cylinder", id: "rig/tripod-leg-c", group: "tripod-steel", tags: ["rig", "instrument"], place: { position: { x: -0.02, y: -0.31, z: 1.48 } }, radius: 0.02, halfHeight: 0.41, material: { palette: "steel", value: 1 } },
  { kind: "box", id: "rig/tripod-spreader", group: "tripod-steel", tags: ["rig", "instrument"], place: { position: { x: -0.02, y: -0.4, z: 1.2750000000000001 } }, halfSize: { x: 0.21500000000000002, y: 0.012, z: 0.185 }, material: { palette: "graphite", value: 1 } },
  { kind: "cylinder", id: "rig/tripod-column", group: "tripod-steel", tags: ["rig", "instrument"], place: { position: { x: -0.02, y: 0.24, z: 1.26 } }, radius: 0.036, halfHeight: 0.16, material: { palette: "pewter", value: 1 } },
  { kind: "box", id: "rig/tripod-head", group: "tripod-steel", tags: ["rig", "instrument"], place: { position: { x: -0.02, y: 0.44500000000000006, z: 1.26 } }, halfSize: { x: 0.05, y: 0.045, z: 0.05 }, material: { palette: "ink", value: 1 } },
  { kind: "box", id: "rig/camera-body", group: "camera-instrument", tags: ["rig", "instrument"], place: { position: { x: -0.02, y: 0.555, z: 1.27 } }, halfSize: { x: 0.1, y: 0.078, z: 0.13 }, material: { colorLinear: [0.07, 0.073, 0.078] } },
  { kind: "ellipsoid", id: "rig/camera-lens", group: "camera-instrument", tags: ["rig", "instrument"], place: { position: { x: -0.02, y: 0.555, z: 1.1 } }, radius: { x: 0.057999999999999996, y: 0.057999999999999996, z: 0.1 }, material: { palette: "ink", value: 1 } },
  { kind: "box", id: "rig/ring-light", group: "emissive-fixture", tags: ["rig", "fixture", "light", "emits-negative-z"], place: { position: { x: -0.02, y: 0.555, z: 1.0550000000000002 } }, halfSize: { x: 0.185, y: 0.185, z: 0.01 }, material: { palette: "ringCool", value: 1, emission: 1.7 } },
  { kind: "box", id: "rig/camera-cable-a", group: "cable-run", tags: ["rig", "cable"], place: { position: { x: 0.039999999999999994, y: 0.4, z: 1.42 } }, halfSize: { x: 0.014, y: 0.1, z: 0.014 }, material: { palette: "ink", value: 1 } },
  { kind: "box", id: "rig/camera-cable-b", group: "cable-run", tags: ["rig", "cable"], place: { position: { x: 0.039999999999999994, y: 0.3, z: 1.56 } }, halfSize: { x: 0.014, y: 0.014, z: 0.145 }, material: { palette: "ink", value: 1 } },
  { kind: "box", id: "calib/board", group: "calibration-target", tags: ["calibration"], place: { position: { x: -0.1, y: 0.55, z: -0.47 } }, halfSize: { x: 0.42000000000000004, y: 0.55, z: 0.014 }, material: { palette: "chalk", value: 1 } },
  { kind: "box", id: "calib/foot-left", group: "steel-frame", tags: ["calibration"], place: { position: { x: -0.44000000000000006, y: 0.022, z: -0.505 } }, halfSize: { x: 0.09, y: 0.022, z: 0.052 }, material: { palette: "steel", value: 1 } },
  { kind: "box", id: "calib/foot-right", group: "steel-frame", tags: ["calibration"], place: { position: { x: 0.24000000000000005, y: 0.022, z: -0.505 } }, halfSize: { x: 0.09, y: 0.022, z: 0.052 }, material: { palette: "steel", value: 1 } },
  { kind: "box", id: "calib/tile-r0c0", group: "calibration-target", tags: ["calibration"], place: { position: { x: -0.41500000000000004, y: 0.19, z: -0.44649999999999995 } }, halfSize: { x: 0.1, y: 0.17, z: 0.01 }, material: { colorLinear: [0.048, 0.049, 0.051] } },
  { kind: "box", id: "calib/tile-r0c2", group: "calibration-target", tags: ["calibration"], place: { position: { x: 0.0050000000000000044, y: 0.19, z: -0.44649999999999995 } }, halfSize: { x: 0.1, y: 0.17, z: 0.01 }, material: { colorLinear: [0.048, 0.049, 0.051] } },
  { kind: "box", id: "calib/tile-r1c1", group: "calibration-target", tags: ["calibration"], place: { position: { x: -0.20500000000000002, y: 0.55, z: -0.44649999999999995 } }, halfSize: { x: 0.1, y: 0.17, z: 0.01 }, material: { colorLinear: [0.048, 0.049, 0.051] } },
  { kind: "box", id: "calib/tile-r1c3", group: "calibration-target", tags: ["calibration"], place: { position: { x: 0.21500000000000002, y: 0.55, z: -0.44649999999999995 } }, halfSize: { x: 0.1, y: 0.17, z: 0.01 }, material: { colorLinear: [0.048, 0.049, 0.051] } },
  { kind: "box", id: "calib/tile-r2c0", group: "calibration-target", tags: ["calibration"], place: { position: { x: -0.41500000000000004, y: 0.9100000000000001, z: -0.44649999999999995 } }, halfSize: { x: 0.1, y: 0.17, z: 0.01 }, material: { colorLinear: [0.048, 0.049, 0.051] } },
  { kind: "box", id: "calib/tile-r2c2", group: "calibration-target", tags: ["calibration"], place: { position: { x: 0.0050000000000000044, y: 0.9100000000000001, z: -0.44649999999999995 } }, halfSize: { x: 0.1, y: 0.17, z: 0.01 }, material: { colorLinear: [0.048, 0.049, 0.051] } },
  { kind: "cylinder", id: "laser/post", group: "laser-instrument", tags: ["laser", "instrument"], place: { position: { x: 0.68, y: 0.15, z: -0.02 } }, radius: 0.045, halfHeight: 0.15, material: { palette: "steel", value: 1 } },
  { kind: "box", id: "laser/housing", group: "laser-instrument", tags: ["laser", "instrument"], place: { position: { x: 0.68, y: 0.4, z: -0.02 } }, halfSize: { x: 0.10500000000000001, y: 0.1, z: 0.145 }, material: { colorLinear: [0.062, 0.065, 0.069] } },
  { kind: "box", id: "laser/aperture", group: "emissive-fixture", tags: ["laser", "fixture", "light", "emits-negative-x"], place: { position: { x: 0.5615000000000001, y: 0.38, z: -0.02 } }, halfSize: { x: 0.0065, y: 0.085, z: 0.012 }, material: { palette: "sheetCool", value: 1, emission: 2.4 } },
  { kind: "box", id: "laser/fin-1", group: "steel-frame", tags: ["laser", "instrument"], place: { position: { x: 0.68, y: 0.512, z: -0.075 } }, halfSize: { x: 0.085, y: 0.014, z: 0.012 }, material: { palette: "pewter", value: 1 } },
  { kind: "box", id: "laser/fin-2", group: "steel-frame", tags: ["laser", "instrument"], place: { position: { x: 0.68, y: 0.512, z: -0.02 } }, halfSize: { x: 0.085, y: 0.014, z: 0.012 }, material: { palette: "pewter", value: 1 } },
  { kind: "box", id: "laser/fin-3", group: "steel-frame", tags: ["laser", "instrument"], place: { position: { x: 0.68, y: 0.512, z: 0.035 } }, halfSize: { x: 0.085, y: 0.014, z: 0.012 }, material: { palette: "pewter", value: 1 } },
  { kind: "box", id: "rack-a/plinth", group: "steel-frame", tags: ["rack"], place: { position: { x: -2.4, y: 0.03500000000000003, z: -0.14 }, anchor: "floor" }, halfSize: { x: 0.24, y: 0.035, z: 0.28 }, material: { palette: "ink", value: 1 } },
  { kind: "box", id: "rack-a/case", group: "rack-console", tags: ["rack", "instrument"], place: { position: { x: -2.4, y: 0.6699999999999999, z: -0.14 }, anchor: "floor" }, halfSize: { x: 0.26, y: 0.6, z: 0.3 }, material: { palette: "slate", value: 1 } },
  { kind: "box", id: "rack-a/ear-far", group: "rack-steel-panel", tags: ["rack", "instrument"], place: { position: { x: -2.128, y: 0.6699999999999999, z: -0.385 }, anchor: "floor" }, halfSize: { x: 0.016, y: 0.5599999999999999, z: 0.045 }, material: { palette: "pewter", value: 1 } },
  { kind: "box", id: "rack-a/ear-near", group: "rack-steel-panel", tags: ["rack", "instrument"], place: { position: { x: -2.128, y: 0.6699999999999999, z: 0.10499999999999998 }, anchor: "floor" }, halfSize: { x: 0.016, y: 0.5599999999999999, z: 0.045 }, material: { palette: "pewter", value: 1 } },
  { kind: "box", id: "rack-a/unit-1", group: "rack-steel-panel", tags: ["rack", "instrument"], place: { position: { x: -2.132, y: 1.0699999999999998, z: -0.14 }, anchor: "floor" }, halfSize: { x: 0.012, y: 0.1, z: 0.235 }, material: { palette: "ash", value: 1 } },
  { kind: "box", id: "rack-a/unit-2", group: "rack-steel-panel", tags: ["rack", "instrument"], place: { position: { x: -2.132, y: 0.6499999999999999, z: -0.14 }, anchor: "floor" }, halfSize: { x: 0.012, y: 0.13, z: 0.235 }, material: { colorLinear: [0.3, 0.306, 0.312] } },
  { kind: "box", id: "rack-a/led", group: "rack-steel-panel", tags: ["rack", "instrument", "emissive-surface-only"], place: { position: { x: -2.116, y: 1.0699999999999998, z: 0.019999999999999997 }, anchor: "floor" }, halfSize: { x: 0.006, y: 0.012, z: 0.045 }, material: { colorLinear: [0.36, 0.62, 1], emission: 0.07 } },
  { kind: "box", id: "rack-b/plinth", group: "steel-frame", tags: ["rack"], place: { position: { x: -2.4, y: 0.03500000000000003, z: 0.66 }, anchor: "floor" }, halfSize: { x: 0.24, y: 0.035, z: 0.28 }, material: { palette: "ink", value: 1 } },
  { kind: "box", id: "rack-b/case", group: "rack-console", tags: ["rack", "instrument"], place: { position: { x: -2.4, y: 0.59, z: 0.66 }, anchor: "floor" }, halfSize: { x: 0.26, y: 0.52, z: 0.3 }, material: { palette: "slate", value: 1 } },
  { kind: "box", id: "rack-b/ear-far", group: "rack-steel-panel", tags: ["rack", "instrument"], place: { position: { x: -2.128, y: 0.59, z: 0.41500000000000004 }, anchor: "floor" }, halfSize: { x: 0.016, y: 0.48, z: 0.045 }, material: { palette: "pewter", value: 1 } },
  { kind: "box", id: "rack-b/ear-near", group: "rack-steel-panel", tags: ["rack", "instrument"], place: { position: { x: -2.128, y: 0.59, z: 0.9050000000000001 }, anchor: "floor" }, halfSize: { x: 0.016, y: 0.48, z: 0.045 }, material: { palette: "pewter", value: 1 } },
  { kind: "box", id: "rack-b/unit-1", group: "rack-steel-panel", tags: ["rack", "instrument"], place: { position: { x: -2.132, y: 0.99, z: 0.66 }, anchor: "floor" }, halfSize: { x: 0.012, y: 0.1, z: 0.235 }, material: { palette: "ash", value: 1 } },
  { kind: "box", id: "rack-b/unit-2", group: "rack-steel-panel", tags: ["rack", "instrument"], place: { position: { x: -2.132, y: 0.57, z: 0.66 }, anchor: "floor" }, halfSize: { x: 0.012, y: 0.13, z: 0.235 }, material: { colorLinear: [0.3, 0.306, 0.312] } },
  { kind: "box", id: "rack-b/led", group: "rack-steel-panel", tags: ["rack", "instrument", "emissive-surface-only"], place: { position: { x: -2.116, y: 0.99, z: 0.8200000000000001 }, anchor: "floor" }, halfSize: { x: 0.006, y: 0.012, z: 0.045 }, material: { colorLinear: [0.36, 0.62, 1], emission: 0.07 } },
  { kind: "box", id: "cable/back-hang", group: "cable-run", tags: ["cable"], place: { position: { x: -0.8, y: -0.16, z: -0.3 } }, halfSize: { x: 0.02, y: 0.1, z: 0.02 }, material: { colorLinear: [0.055, 0.057, 0.06] } },
  { kind: "box", id: "cable/back-sag", group: "cable-run", tags: ["cable"], place: { position: { x: -0.9000000000000001, y: -0.32, z: -0.3 } }, halfSize: { x: 0.11000000000000001, y: 0.02, z: 0.02 }, material: { colorLinear: [0.055, 0.057, 0.06] } },
  { kind: "box", id: "cable/back-drop", group: "cable-run", tags: ["cable"], place: { position: { x: -1.02, y: -0.5, z: -0.3 } }, halfSize: { x: 0.02, y: 0.18, z: 0.02 }, material: { colorLinear: [0.055, 0.057, 0.06] } },
  { kind: "box", id: "cable/back-floor", group: "cable-run", tags: ["cable"], place: { position: { x: -1.55, y: 0.021999999999999985, z: -0.3 }, anchor: "floor" }, halfSize: { x: 0.55, y: 0.02, z: 0.022 }, material: { colorLinear: [0.062, 0.064, 0.068] } },
  { kind: "box", id: "cable/front-hang", group: "cable-run", tags: ["cable"], place: { position: { x: -0.8, y: -0.16, z: 0.36 } }, halfSize: { x: 0.02, y: 0.1, z: 0.02 }, material: { colorLinear: [0.055, 0.057, 0.06] } },
  { kind: "box", id: "cable/front-sag", group: "cable-run", tags: ["cable"], place: { position: { x: -0.9000000000000001, y: -0.32, z: 0.36 } }, halfSize: { x: 0.11000000000000001, y: 0.02, z: 0.02 }, material: { colorLinear: [0.055, 0.057, 0.06] } },
  { kind: "box", id: "cable/front-drop", group: "cable-run", tags: ["cable"], place: { position: { x: -1.02, y: -0.5, z: 0.36 } }, halfSize: { x: 0.02, y: 0.18, z: 0.02 }, material: { colorLinear: [0.055, 0.057, 0.06] } },
  { kind: "box", id: "cable/front-floor", group: "cable-run", tags: ["cable"], place: { position: { x: -1.55, y: 0.021999999999999985, z: 0.36 }, anchor: "floor" }, halfSize: { x: 0.55, y: 0.02, z: 0.022 }, material: { colorLinear: [0.062, 0.064, 0.068] } },
  { kind: "cylinder", id: "cable/coil", group: "cable-run", tags: ["cable"], place: { position: { x: -1.62, y: 0.03500000000000003, z: 0.96 }, anchor: "floor" }, radius: 0.19, halfHeight: 0.035, material: { colorLinear: [0.072, 0.074, 0.078] } },
  { kind: "box", id: "bench/notebook-a", group: "paper-stock", tags: ["bench", "paper"], place: { position: { x: 0.42000000000000004, y: 0.012, z: 0.5 } }, halfSize: { x: 0.125, y: 0.012, z: 0.14 }, material: { palette: "paper", value: 1 } },
  { kind: "box", id: "bench/notebook-b", group: "paper-stock", tags: ["bench", "paper"], place: { position: { x: 0.4, y: 0.036, z: 0.51 } }, halfSize: { x: 0.11500000000000002, y: 0.012, z: 0.125 }, material: { palette: "bone", value: 1 } },
  { kind: "box", id: "bench/pen", group: "steel-frame", tags: ["bench", "instrument"], place: { position: { x: 0.2, y: 0.026, z: 0.53 } }, halfSize: { x: 0.07, y: 0.007, z: 0.01 }, material: { palette: "ink", value: 1 } },
  { kind: "box", id: "bench/scale-bar", group: "paper-stock", tags: ["bench", "instrument"], place: { position: { x: -0.06, y: 0.008, z: 0.555 } }, halfSize: { x: 0.4, y: 0.008, z: 0.016 }, material: { palette: "paper", value: 1 } },
  { kind: "box", id: "bench/tray-1", group: "tray-steel", tags: ["bench", "tray"], place: { position: { x: -0.58, y: 0.014, z: 0.5 } }, halfSize: { x: 0.175, y: 0.014, z: 0.11500000000000002 }, material: { palette: "ash", value: 1 } },
  { kind: "box", id: "bench/tray-2", group: "tray-steel", tags: ["bench", "tray"], place: { position: { x: -0.575, y: 0.042, z: 0.505 } }, halfSize: { x: 0.16999999999999998, y: 0.014, z: 0.11000000000000001 }, material: { palette: "bone", value: 1 } },
  { kind: "box", id: "bench/tray-3", group: "tray-steel", tags: ["bench", "tray"], place: { position: { x: -0.57, y: 0.07, z: 0.51 } }, halfSize: { x: 0.16499999999999998, y: 0.014, z: 0.10500000000000001 }, material: { palette: "chalk", value: 1 } },
  { kind: "box", id: "scope/body", group: "scope-instrument", tags: ["rack", "instrument"], place: { position: { x: -2.35, y: 1.365, z: -0.1 }, anchor: "floor" }, halfSize: { x: 0.15, y: 0.095, z: 0.10500000000000001 }, material: { palette: "graphite", value: 1 } },
  { kind: "box", id: "scope/screen", group: "scope-glass", tags: ["rack", "instrument", "emissive-surface-only"], place: { position: { x: -2.195, y: 1.375, z: -0.1 }, anchor: "floor" }, halfSize: { x: 0.005, y: 0.062, z: 0.085 }, material: { palette: "displayCool", value: 1, emission: 0.1 } },
  { kind: "box", id: "wall/panel", group: "wall-panel-board", tags: ["wall", "board"], place: { position: { x: -2.7720000000000002, y: 1.62, z: 0.2 }, anchor: "floor" }, halfSize: { x: 0.028, y: 0.58, z: 1.05 }, material: { palette: "bone", value: 1 } },
  { kind: "box", id: "wall/panel-mark-1", group: "wall-panel-board", tags: ["wall", "board"], place: { position: { x: -2.742, y: 1.9, z: -0.3 }, anchor: "floor" }, halfSize: { x: 0.006, y: 0.02, z: 0.38 }, material: { colorLinear: [0.075, 0.078, 0.082] } },
  { kind: "box", id: "wall/panel-mark-2", group: "wall-panel-board", tags: ["wall", "board"], place: { position: { x: -2.742, y: 1.7000000000000002, z: -0.1 }, anchor: "floor" }, halfSize: { x: 0.006, y: 0.018, z: 0.26 }, material: { colorLinear: [0.075, 0.078, 0.082] } },
  { kind: "box", id: "wall/panel-mark-3", group: "wall-panel-board", tags: ["wall", "board"], place: { position: { x: -2.742, y: 1.32, z: 0.3 }, anchor: "floor" }, halfSize: { x: 0.006, y: 0.022, z: 0.5 }, material: { colorLinear: [0.075, 0.078, 0.082] } },
  { kind: "box", id: "wall/cable-tray", group: "conduit-pipe", tags: ["wall", "conduit"], place: { position: { x: -2.71, y: 2.35, z: 0.1 }, anchor: "floor" }, halfSize: { x: 0.09, y: 0.035, z: 1.5 }, material: { colorLinear: [0.19, 0.196, 0.204] } },
  { kind: "box", id: "wall/conduit-a", group: "conduit-pipe", tags: ["wall", "conduit"], place: { position: { x: -2.7399999999999998, y: 1.175, z: -0.040000000000000015 }, anchor: "floor" }, halfSize: { x: 0.03, y: 1.175, z: 0.03 }, material: { colorLinear: [0.2, 0.206, 0.214] } },
  { kind: "box", id: "wall/conduit-b", group: "conduit-pipe", tags: ["wall", "conduit"], place: { position: { x: -2.7399999999999998, y: 1.175, z: 0.76 }, anchor: "floor" }, halfSize: { x: 0.03, y: 1.175, z: 0.03 }, material: { colorLinear: [0.2, 0.206, 0.214] } },
  { kind: "cylinder", id: "dewar/body", group: "dewar-steel", tags: ["dewar", "instrument"], place: { position: { x: 1.6700000000000002, y: 0.62, z: -1.125 }, anchor: "floor" }, radius: 0.17, halfHeight: 0.62, material: { palette: "ash", value: 1 } },
  { kind: "cylinder", id: "dewar/collar", group: "dewar-steel", tags: ["dewar", "instrument"], place: { position: { x: 1.6700000000000002, y: 1.25, z: -1.125 }, anchor: "floor" }, radius: 0.185, halfHeight: 0.035, material: { palette: "pewter", value: 1 } },
  { kind: "cylinder", id: "dewar/cap", group: "dewar-steel", tags: ["dewar", "instrument"], place: { position: { x: 1.6700000000000002, y: 1.38, z: -1.125 }, anchor: "floor" }, radius: 0.11500000000000002, halfHeight: 0.1, material: { palette: "steel", value: 1 } },
];

export const nightLabSceneryGraph: SceneryGraph = {
  palettes: {
    ink: { tint: [.042, .044, .048] }, graphite: { tint: [.085, .089, .095] },
    slate: { tint: [.150, .156, .163] }, steel: { tint: [.250, .258, .268] },
    pewter: { tint: [.355, .360, .366] }, ash: { tint: [.470, .472, .474] },
    bone: { tint: [.640, .640, .632] }, chalk: { tint: [.815, .813, .802] },
    paper: { tint: [.915, .910, .895] }, tungsten: { tint: [1.00, .585, .235] },
    coolBar: { tint: [.520, .690, 1.00] }, displayCool: { tint: [.230, .455, .720] },
    ringCool: { tint: [.760, .855, 1.00] }, sheetCool: { tint: [.560, .800, 1.00] },
  },
  nodes: NIGHT_LAB_NODES,
};

// ---- the remaining sets ---------------------------------------------------
//
// These five were re-authored rather than transcribed. Their imperative modules
// were the same shapes in a different notation, and holding a hundred recovered
// digits per set to prove that would have bought nothing: only the three paper
// figures — hose-tank, garden lighting, jet past sphere — carry composition that
// had to survive. What each set has to keep is its *reading*: a colonnade at
// noon, a luminous slot, battens striping lantern light, a pressure hull, a
// measurement studio. Each is a shell and the handful of objects that say which
// room you are standing in.

/** Chalky lime stucco, and the cooler stone it is laid on. */
const STUCCO = { tint: [1, .992, .972] as const };
const FLAG = { tint: [.985, .99, 1] as const };

/**
 * Mediterranean courtyard: a colonnade at noon.
 *
 * The columns stand across the back at a 0.85 s pitch, wide enough that the
 * shadow they cast falls between them rather than over the tank, and a rill runs
 * out of the shade toward the container — the one line in the set that points at
 * the water. The sky is the ceiling, and it is the light.
 */
function courtyardNodes(): readonly SceneryNode[] {
  const column = (index: number, x: number): SceneryNode => ({
    kind: "group", id: `column-${index}`, place: { position: { x, y: 0, z: -1.35 }, anchor: "floor" },
    children: [
      { kind: "box", id: `column-${index}/base`, group: "stone-column", tags: ["colonnade"], place: { position: { x: 0, y: .05, z: 0 } }, halfSize: { x: .105, y: .05, z: .105 }, material: { palette: "stone", value: .60 } },
      { kind: "cylinder", id: `column-${index}/shaft`, group: "stone-column", tags: ["colonnade"], place: { position: { x: 0, y: .88, z: 0 } }, radius: .072, halfHeight: .78, material: { palette: "stucco", value: .88 } },
      { kind: "box", id: `column-${index}/capital`, group: "stone-column", tags: ["colonnade"], place: { position: { x: 0, y: 1.71, z: 0 } }, halfSize: { x: .1, y: .05, z: .1 }, material: { palette: "stucco", value: .80 } },
    ],
  });
  const beam = (index: number, z: number): SceneryNode => ({
    kind: "box", id: `pergola/beam-${index}`, group: "pergola", tags: ["pergola"],
    place: { position: { x: 0, y: 1.83, z }, anchor: "floor" },
    halfSize: { x: 1.9, y: .035, z: .045 }, material: { palette: "stucco", value: .74 },
  });
  return [
    {
      kind: "room-shell", id: "shell", materialModel: "courtyard",
      floor: { palette: "stone", value: .58 },
      wall: { colorLinear: [1, 1, 1] },
      ceiling: { palette: "stucco", value: .95 },
    },
    // Noon: one hard source high and slightly behind the colonnade, so the
    // columns throw their shadows between themselves and never across the tank.
    { kind: "box", id: "sun/panel", group: "softbox", tags: ["light", "emits-negative-y"], place: { position: { x: -.35, y: 2.55, z: -.55 }, anchor: "floor" }, halfSize: { x: .85, y: .03, z: .7 }, material: { palette: "daylight", value: 1, emission: 2.6 } },
    ...[-1.7, -.85, 0, .85, 1.7].map((x, index) => column(index + 1, x)),
    ...[-1.05, -.45, .15].map((z, index) => beam(index + 1, z)),
    // The rill: a shallow channel of the same stone, running out of the shade.
    { kind: "box", id: "rill/channel", group: "stone-rill", tags: ["rill"], place: { position: { x: 0, y: .02, z: -.98 }, anchor: "floor" }, halfSize: { x: .085, y: .02, z: .34 }, material: { palette: "stone", value: .52 } },
    { kind: "box", id: "rill/lip-left", group: "stone-rill", tags: ["rill"], place: { position: { x: -.115, y: .035, z: -.98 }, anchor: "floor" }, halfSize: { x: .03, y: .035, z: .34 }, material: { palette: "stone", value: .70 } },
    { kind: "box", id: "rill/lip-right", group: "stone-rill", tags: ["rill"], place: { position: { x: .115, y: .035, z: -.98 }, anchor: "floor" }, halfSize: { x: .03, y: .035, z: .34 }, material: { palette: "stone", value: .70 } },
    // Two pots on the flagstones, off frame edge, for scale.
    { kind: "cone", id: "pot-left/body", group: "clay-pot", tags: ["pot"], place: { position: { x: -1.32, y: .17, z: .42 }, anchor: "floor" }, baseRadius: .11, topRadius: .17, halfHeight: .17, material: { palette: "stucco", value: .70 } },
    { kind: "torus", id: "pot-left/rim", group: "clay-pot", tags: ["pot"], place: { position: { x: -1.32, y: .34, z: .42 }, anchor: "floor" }, majorRadius: .17, minorRadius: .019, material: { palette: "stucco", value: .78 } },
    { kind: "cone", id: "pot-right/body", group: "clay-pot", tags: ["pot"], place: { position: { x: 1.38, y: .14, z: .30 }, anchor: "floor" }, baseRadius: .09, topRadius: .14, halfHeight: .14, material: { palette: "stucco", value: .68 } },
    { kind: "torus", id: "pot-right/rim", group: "clay-pot", tags: ["pot"], place: { position: { x: 1.38, y: .28, z: .30 }, anchor: "floor" }, majorRadius: .14, minorRadius: .016, material: { palette: "stucco", value: .76 } },
    // A stone bench along the left wall, so the room has somewhere to stand.
    { kind: "box", id: "bench/seat", group: "stone-bench", tags: ["bench"], place: { position: { x: -1.95, y: .42, z: .1 }, anchor: "floor" }, halfSize: { x: .16, y: .04, z: .78 }, material: { palette: "stone", value: .72 } },
    { kind: "box", id: "bench/leg-front", group: "stone-bench", tags: ["bench"], place: { position: { x: -1.95, y: .19, z: .78 }, anchor: "floor" }, halfSize: { x: .13, y: .19, z: .06 }, material: { palette: "stone", value: .62 } },
    { kind: "box", id: "bench/leg-back", group: "stone-bench", tags: ["bench"], place: { position: { x: -1.95, y: .19, z: -.58 }, anchor: "floor" }, halfSize: { x: .13, y: .19, z: .06 }, material: { palette: "stone", value: .62 } },
  ];
}

export const courtyardSceneryGraph: SceneryGraph = {
  palettes: { stucco: STUCCO, stone: FLAG, daylight: { tint: [1, .985, .95] } },
  nodes: courtyardNodes(),
};

/**
 * Brutalist water gallery: the room is a plinth and the dam break is the exhibit.
 *
 * One luminous slot runs the length of the back wall at eye height; everything
 * else is board-formed concrete, ribbed by the shuttering it was poured against.
 * The plinth under the container is what makes the tank read as an exhibit
 * rather than as furniture.
 */
function galleryNodes(): readonly SceneryNode[] {
  const board = (index: number, y: number): SceneryNode => ({
    kind: "box", id: `board/rib-${index}`, group: "concrete-board", tags: ["board-form"],
    place: { position: { x: 0, y, z: -2.72 }, anchor: "floor" },
    halfSize: { x: 2.7, y: .006, z: .012 }, material: { palette: "concrete", value: .46 },
  });
  return [
    {
      kind: "room-shell", id: "shell", materialModel: "gallery",
      floor: { palette: "concrete", value: .40 },
      wall: { colorLinear: [1, 1, 1] },
      ceiling: { palette: "concrete", value: .50 },
    },
    ...[.42, .78, 1.14, 1.5, 1.86, 2.22].map((y, index) => board(index + 1, y)),
    // The slot. A thin bright line at eye height is the whole lighting design:
    // it grazes the boards, so the poured texture reads across the wall.
    { kind: "box", id: "slot/light", group: "emissive-slot", tags: ["light", "emits-positive-z"], place: { position: { x: 0, y: 1.32, z: -2.7 }, anchor: "floor" }, halfSize: { x: 2.3, y: .055, z: .02 }, material: { palette: "daylight", value: 1, emission: 3.2 } },
    { kind: "box", id: "slot/reveal-lower", group: "concrete-board", tags: ["slot"], place: { position: { x: 0, y: 1.24, z: -2.66 }, anchor: "floor" }, halfSize: { x: 2.3, y: .026, z: .06 }, material: { palette: "concrete", value: .64 } },
    { kind: "box", id: "slot/reveal-upper", group: "concrete-board", tags: ["slot"], place: { position: { x: 0, y: 1.4, z: -2.66 }, anchor: "floor" }, halfSize: { x: 2.3, y: .026, z: .06 }, material: { palette: "concrete", value: .64 } },
    // The plinth is a kerb around the tank, not a slab under it: scenery must
    // leave the container volume empty, and a gallery plinth reads from its
    // edge anyway. Four boxes, each clear of the footprint on its own axis.
    { kind: "box", id: "plinth/kerb-left", group: "concrete-plinth", tags: ["plinth"], place: { position: { x: -.78, y: .09, z: 0 }, anchor: "floor" }, halfSize: { x: .27, y: .09, z: .95 }, material: { palette: "concrete", value: .52 } },
    { kind: "box", id: "plinth/kerb-right", group: "concrete-plinth", tags: ["plinth"], place: { position: { x: .78, y: .09, z: 0 }, anchor: "floor" }, halfSize: { x: .27, y: .09, z: .95 }, material: { palette: "concrete", value: .52 } },
    { kind: "box", id: "plinth/kerb-front", group: "concrete-plinth", tags: ["plinth"], place: { position: { x: 0, y: .09, z: .68 }, anchor: "floor" }, halfSize: { x: .51, y: .09, z: .27 }, material: { palette: "concrete", value: .52 } },
    { kind: "box", id: "plinth/kerb-back", group: "concrete-plinth", tags: ["plinth"], place: { position: { x: 0, y: .09, z: -.68 }, anchor: "floor" }, halfSize: { x: .51, y: .09, z: .27 }, material: { palette: "concrete", value: .52 } },
    // A bench facing the exhibit, because a gallery is somewhere you stand.
    { kind: "box", id: "bench/slab", group: "concrete-bench", tags: ["bench"], place: { position: { x: -1.75, y: .43, z: -.2 }, anchor: "floor" }, halfSize: { x: .22, y: .045, z: .82 }, material: { palette: "concrete", value: .60 } },
    { kind: "box", id: "bench/leg-front", group: "concrete-bench", tags: ["bench"], place: { position: { x: -1.75, y: .2, z: .5 }, anchor: "floor" }, halfSize: { x: .17, y: .2, z: .07 }, material: { palette: "concrete", value: .46 } },
    { kind: "box", id: "bench/leg-back", group: "concrete-bench", tags: ["bench"], place: { position: { x: -1.75, y: .2, z: -.9 }, anchor: "floor" }, halfSize: { x: .17, y: .2, z: .07 }, material: { palette: "concrete", value: .46 } },
  ];
}

export const gallerySceneryGraph: SceneryGraph = {
  palettes: { concrete: { tint: [.99, .995, 1] }, daylight: { tint: [1, .985, .95] } },
  nodes: galleryNodes(),
};

/**
 * Japanese bathhouse: bleached cedar battens striping lantern light.
 *
 * The set is one gesture — a batten screen across the back — and two paper
 * lanterns lighting it from in front, so the light arrives already striped. The
 * calmest set in the app, which means nothing in it moves and nothing in it is
 * bright except the lanterns.
 */
function bathhouseNodes(): readonly SceneryNode[] {
  const batten = (index: number, x: number): SceneryNode => ({
    kind: "box", id: `screen/batten-${index}`, group: "cedar-screen", tags: ["batten", "wood"],
    place: { position: { x, y: 1.05, z: -1.62 }, anchor: "floor" },
    halfSize: { x: .038, y: 1.05, z: .038 },
    // The value walk is what makes the screen read as depth rather than as a
    // flat stripe pattern: alternate battens sit a shade back in the light.
    material: { palette: "cedar", value: index % 2 === 0 ? .74 : .82 },
  });
  const lantern = (side: "left" | "right", x: number, z: number, height: number): SceneryNode => ({
    kind: "group", id: `lantern-${side}`, place: { position: { x, y: 0, z }, anchor: "floor" },
    children: [
      { kind: "cylinder", id: `lantern-${side}/paper`, group: "lantern-paper", tags: ["light", "lantern"], place: { position: { x: 0, y: height, z: 0 } }, radius: .155, halfHeight: .21, material: { palette: "lamp", value: 1, emission: 2.1 } },
      { kind: "cylinder", id: `lantern-${side}/cap`, group: "cedar-fixture", tags: ["lantern"], place: { position: { x: 0, y: height + .225, z: 0 } }, radius: .17, halfHeight: .018, material: { palette: "cedar", value: .55 } },
      // Thinner than a voxel: the cord is one of the surfaces the coverage
      // audit watches, because a subcell prop is where a raster set would
      // quietly drop geometry the analytic path still has to draw.
      { kind: "cylinder", id: `lantern-${side}/cord`, group: "cedar-fixture", tags: ["lantern"], place: { position: { x: 0, y: height + .72, z: 0 } }, radius: .008, halfHeight: .48, material: { palette: "cedar", value: .42 } },
    ],
  });
  return [
    {
      kind: "room-shell", id: "shell", materialModel: "bathhouse",
      floor: { palette: "cedar", value: .50 },
      wall: { colorLinear: [1, 1, 1] },
      ceiling: { palette: "cedar", value: .60 },
    },
    ...[-1.68, -1.26, -.84, -.42, 0, .42, .84, 1.26, 1.68].map((x, index) => batten(index + 1, x)),
    { kind: "box", id: "screen/rail-low", group: "cedar-screen", tags: ["batten", "wood"], place: { position: { x: 0, y: .16, z: -1.62 }, anchor: "floor" }, halfSize: { x: 1.76, y: .05, z: .046 }, material: { palette: "cedar", value: .66 } },
    { kind: "box", id: "screen/rail-high", group: "cedar-screen", tags: ["batten", "wood"], place: { position: { x: 0, y: 2.04, z: -1.62 }, anchor: "floor" }, halfSize: { x: 1.76, y: .06, z: .046 }, material: { palette: "cedar", value: .66 } },
    lantern("left", -1.02, .58, 1.42),
    lantern("right", 1.14, .34, 1.18),
    // A cedar duckboard beside the tank rather than under it — scenery has to
    // leave the container volume empty — and a low stool on it.
    { kind: "box", id: "duckboard/deck", group: "cedar-deck", tags: ["deck", "wood"], place: { position: { x: -1.24, y: .035, z: .05 }, anchor: "floor" }, halfSize: { x: .58, y: .035, z: .88 }, material: { palette: "cedar", value: .70 } },
    { kind: "box", id: "stool/seat", group: "cedar-stool", tags: ["stool", "wood"], place: { position: { x: -1.24, y: .28, z: .12 }, anchor: "floor" }, halfSize: { x: .16, y: .028, z: .13 }, material: { palette: "cedar", value: .78 } },
    { kind: "cylinder", id: "stool/leg-left", group: "cedar-stool", tags: ["stool", "wood"], place: { position: { x: -1.36, y: .13, z: .12 }, anchor: "floor" }, radius: .019, halfHeight: .13, material: { palette: "cedar", value: .62 } },
    { kind: "cylinder", id: "stool/leg-right", group: "cedar-stool", tags: ["stool", "wood"], place: { position: { x: -1.12, y: .13, z: .12 }, anchor: "floor" }, radius: .019, halfHeight: .13, material: { palette: "cedar", value: .62 } },
    { kind: "cylinder", id: "bucket/body", group: "cedar-bucket", tags: ["bucket", "wood"], place: { position: { x: 1.32, y: .105, z: .5 }, anchor: "floor" }, radius: .13, halfHeight: .105, material: { palette: "cedar", value: .84 } },
    { kind: "torus", id: "bucket/hoop", group: "cedar-bucket", tags: ["bucket"], place: { position: { x: 1.32, y: .17, z: .5 }, anchor: "floor" }, majorRadius: .132, minorRadius: .012, material: { palette: "plaster", value: .52 } },
  ];
}

export const bathhouseSceneryGraph: SceneryGraph = {
  palettes: {
    cedar: { tint: [1, .975, .93] }, plaster: { tint: [.99, .995, 1] },
    lamp: { tint: [1, .87, .66] },
  },
  nodes: bathhouseNodes(),
};

/**
 * Submerged research station: a white pressure hull built to hold water back.
 *
 * Four ribs hoop the room on the Z axis — the one set where a torus is the
 * honest primitive rather than a bead chain approximating one — with bolted
 * flanges between them and a depth gauge climbing the left wall. A single
 * porthole behind the tank is the only thing in the set that is lit from
 * outside, which is what places the room underwater.
 */
function stationNodes(): readonly SceneryNode[] {
  // A hoop standing in the XY plane: the torus sweeps about local +Y, so its
  // axis is turned a quarter turn about X to lie along Z.
  const hoopAxis = { w: .7071067811865476, x: -.7071067811865476, y: 0, z: 0 };
  /** The same quarter turn about Z: local +Y onto +X, for anything set into a side wall. */
  const wallAxis = { w: .7071067811865476, x: 0, y: 0, z: -.7071067811865476 };
  const rib = (index: number, z: number): SceneryNode => ({
    kind: "group", id: `hull/rib-${index}`, place: { position: { x: 0, y: 1.62, z }, anchor: "floor" },
    children: [
      { kind: "torus", id: `hull/rib-${index}/hoop`, group: "steel-rib", tags: ["hull", "rib"], place: { orientation: hoopAxis }, majorRadius: 1.72, minorRadius: .058, material: { palette: "hull", value: .74 } },
      { kind: "box", id: `hull/rib-${index}/flange-left`, group: "steel-rib", tags: ["hull", "flange"], place: { position: { x: -1.72, y: 0, z: 0 } }, halfSize: { x: .075, y: .1, z: .075 }, material: { palette: "hull", value: .58 } },
      { kind: "box", id: `hull/rib-${index}/flange-right`, group: "steel-rib", tags: ["hull", "flange"], place: { position: { x: 1.72, y: 0, z: 0 } }, halfSize: { x: .075, y: .1, z: .075 }, material: { palette: "hull", value: .58 } },
      { kind: "box", id: `hull/rib-${index}/crown`, group: "steel-rib", tags: ["hull", "flange"], place: { position: { x: 0, y: 1.72, z: 0 } }, halfSize: { x: .1, y: .075, z: .075 }, material: { palette: "hull", value: .58 } },
    ],
  });
  const gauge = (index: number, y: number): SceneryNode => ({
    kind: "box", id: `gauge/mark-${index}`, group: "instrument-gauge", tags: ["gauge"],
    place: { position: { x: -2.0, y, z: .35 }, anchor: "floor" },
    // Every fifth mark is a major division, which is the only thing that makes a
    // ladder of identical ticks readable as a scale.
    halfSize: { x: .012, y: .008, z: index % 5 === 0 ? .16 : .085 },
    material: { palette: "paint", value: index % 5 === 0 ? .92 : .62 },
  });
  return [
    {
      kind: "room-shell", id: "shell", materialModel: "station",
      floor: { palette: "hull", value: .48 },
      wall: { colorLinear: [1, 1, 1] },
      ceiling: { palette: "hull", value: .70 },
      // The observation port: a hole in the hull, the glass in it, and the lit
      // sea outside. One declaration, so the frame cannot drift off the pane.
      backWall: {
        halfWidth: .66, halfHeight: .39, centerY: 1.55,
        frame: "observation-port/frame", glazing: "observation-port/glazing",
        backing: {
          id: "observation-port/backing", group: "porthole-water",
          tags: ["light", "emits-positive-z", "window", "backing"],
          material: { palette: "deep", value: .62, emission: .9 },
        },
      },
    },
    ...[-1.35, -.45, .45, 1.35].map((z, index) => rib(index + 1, z)),
    ...Array.from({ length: 16 }, (_unused, index) => gauge(index, .28 + .13 * index)),
    { kind: "box", id: "gauge/rail", group: "instrument-gauge", tags: ["gauge"], place: { position: { x: -2.02, y: 1.26, z: .35 }, anchor: "floor" }, halfSize: { x: .008, y: 1.02, z: .014 }, material: { palette: "paint", value: .84 } },
    // A round porthole in the left wall, beside the rectangular observation
    // port. Circular transmission is the one glazing shape the finite-pane
    // path cannot express, so this stays a lit disc rather than real glass —
    // and lib/svo-scene-glass.ts publishes that as an explicit unsupported
    // entry rather than letting it pass as a pane.
    { kind: "cylinder", id: "porthole/glass", group: "porthole-glass", tags: ["light"], place: { position: { x: -2.02, y: 1.66, z: -.6 }, anchor: "floor", orientation: wallAxis }, radius: .42, halfHeight: .03, material: { palette: "deep", value: 1, emission: 1.6 } },
    { kind: "torus", id: "porthole/ring", group: "steel-fixture", tags: ["porthole"], place: { position: { x: -2.0, y: 1.66, z: -.6 }, anchor: "floor", orientation: wallAxis }, majorRadius: .44, minorRadius: .045, material: { palette: "hull", value: .66 } },
    { kind: "torus", id: "porthole/bolts", group: "steel-fixture", tags: ["porthole"], place: { position: { x: -1.96, y: 1.66, z: -.6 }, anchor: "floor", orientation: wallAxis }, majorRadius: .52, minorRadius: .018, material: { palette: "hull", value: .52 } },
    // A pipe run along the right wall, and the valve it terminates in.
    { kind: "capsule", id: "pipe/run", group: "steel-pipe", tags: ["pipe"], place: { anchor: "floor" }, from: { x: 2.02, y: 2.35, z: -1.9 }, to: { x: 2.02, y: 2.35, z: 1.5 }, radius: .062, material: { palette: "hull", value: .60 } },
    { kind: "capsule", id: "pipe/drop", group: "steel-pipe", tags: ["pipe"], place: { anchor: "floor" }, from: { x: 2.02, y: 2.35, z: 1.5 }, to: { x: 2.02, y: 1.05, z: 1.5 }, radius: .062, material: { palette: "hull", value: .60 } },
    { kind: "torus", id: "pipe/valve", group: "steel-pipe", tags: ["pipe", "valve"], place: { position: { x: 2.02, y: 1.02, z: 1.5 }, anchor: "floor" }, majorRadius: .14, minorRadius: .022, material: { palette: "paint", value: .70 } },
  ];
}

export const stationSceneryGraph: SceneryGraph = {
  palettes: {
    hull: { tint: [.985, 1, 1] }, paint: { tint: [1, .99, .96] },
    deep: { tint: [.72, .87, 1] },
  },
  nodes: stationNodes(),
};

/** A plain white room and the single overhead source that lights it. */
function studioNodes(): readonly SceneryNode[] {
  return [
    // The front face remains in the SVO model but the interior renderer omits
    // it, so the camera can look into the enclosure when a water scene is
    // framed tightly.
    {
      kind: "room-shell", id: "shell", materialModel: "room",
      halfSize: { x: 1.45, y: 1, z: 1.45 },
      floor: { palette: "white", value: .9 },
      wall: { colorLinear: [1, 1, 1] },
      ceiling: { palette: "white", value: .9 },
    },
    { kind: "box", id: "light/softbox", group: "softbox", tags: ["softbox", "light", "emits-negative-y"], place: { position: { x: 0, y: 1.72, z: -.15 }, anchor: "floor" }, halfSize: { x: .72, y: .03, z: .55 }, material: { palette: "daylight", value: 1, emission: 1 } },
  ];
}

export const studioSceneryGraph: SceneryGraph = {
  palettes: { white: { tint: [1, 1, 1] }, daylight: { tint: [1, 1, 1] } },
  nodes: studioNodes(),
};

/**
 * The scenery an environment seeds a new scene with.
 *
 * Total over `EnvironmentId`: every environment is described here, because this
 * is now the only description there is. A scene copies the graph at creation and
 * owns it from then on.
 */
export const SCENERY_GRAPHS: Readonly<Record<EnvironmentId, (scene: SceneDescription) => SceneryGraph>> =
  Object.freeze({
    garden: () => gardenSceneryGraph,
    conservatory: () => conservatorySceneryGraph,
    "night-lab": () => nightLabSceneryGraph,
    courtyard: () => courtyardSceneryGraph,
    "concrete-gallery": () => gallerySceneryGraph,
    bathhouse: () => bathhouseSceneryGraph,
    "research-station": () => stationSceneryGraph,
    default: () => studioSceneryGraph,
  });

/** The graph a scene starts from when it adopts `environmentId`. */
export function sceneryGraphForEnvironment(
  scene: SceneDescription,
  environmentId: EnvironmentId,
): SceneryGraph {
  return SCENERY_GRAPHS[environmentId](scene);
}

/**
 * Adopt an environment: its identity and the scenery it seeds, together.
 *
 * These are one act, not two. A scene that named a new environment while
 * keeping its old graph would render the previous set in the new room's frame —
 * a conservatory's pot bench standing in a night lab — and it would look like a
 * rendering bug rather than a stale document.
 */
export function sceneWithEnvironment<T extends SceneDescription>(
  scene: T,
  environmentId: EnvironmentId,
): T {
  return { ...scene, environment: environmentId, scenery: sceneryGraphForEnvironment(scene, environmentId) };
}
