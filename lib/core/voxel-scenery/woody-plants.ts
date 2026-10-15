import type { Vec3 } from "../model";
import type { SceneryGroupNode, SceneryMaterial, SceneryRecursiveShapeNode, SceneryTaperedSweepClusterNode } from "../scenery-graph";
import { sampleSvoPrimitive } from "../../svo/contracts/svo-primitive-abi";
import { alongAxis } from "./builder";
import { oakFoliageDescriptor } from "./oak-v2";

export const WOODY_PLANT_SPECIES = ["birch", "pine", "willow", "azalea", "juniper", "boxwood", "hebe"] as const;
export type WoodyPlantSpecies = typeof WOODY_PLANT_SPECIES[number];
export interface WoodyPlantSpec {
  readonly key: string;
  readonly species: WoodyPlantSpecies;
  readonly seed: number;
  /** Metres per unit specimen, independent of the render lattice. */
  readonly scale_m: number;
  /** Fewer shoots for miniature plantings; never selected by voxel resolution. */
  readonly detail?: "full" | "miniature";
  readonly bark: SceneryMaterial;
  readonly foliage: SceneryMaterial;
}
export interface WoodyPlantBranch {
  readonly id: string;
  readonly parent: string | null;
  readonly from: Vec3;
  readonly to: Vec3;
  readonly baseRadius: number;
  readonly tipRadius: number;
}
export interface WoodyPlantPlan {
  readonly node: SceneryGroupNode;
  readonly branches: readonly WoodyPlantBranch[];
}
const v = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
const mix = (a: Vec3, b: Vec3, t: number) => v(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);
const goldenAngle = Math.PI * (3 - Math.sqrt(5));
function hash(n: number): number {
  n = Math.imul(n ^ (n >>> 16), 0x7feb352d);
  n = Math.imul(n ^ (n >>> 15), 0x846ca68b);
  return (n ^ (n >>> 16)) >>> 0;
}

/**
 * Species-specific skeletons with the oak's tapered sweeps and porous leaf
 * fields. Birch has ascending alternate shoots, pine radial whorls, willow
 * hanging branchlets, azalea basal forks and juniper low evergreen sprays.
 * Boxwood and hebe carry overlapping foliage along low, multi-stem crowns.
 * Geometry is seeded, editable and independent of sampling resolution. The
 * neutral material is supplied by the scene; species read through their form.
 */
export function planWoodyPlant(spec: WoodyPlantSpec): WoodyPlantPlan {
  if (!WOODY_PLANT_SPECIES.includes(spec.species) || !Number.isFinite(spec.scale_m)
    || spec.scale_m <= 0 || !Number.isSafeInteger(spec.seed) || !spec.key.trim()
    || (spec.detail !== undefined && spec.detail !== "full" && spec.detail !== "miniature")) {
    throw new RangeError("Woody plants need a known species, key, positive finite scale and integer seed");
  }
  const s = spec.scale_m;
  const miniature = spec.detail === "miniature";
  let salt = 0;
  const random = () => hash(spec.seed ^ Math.imul(++salt, 0x9e3779b1)) / 4294967296;
  const p = (x: number, y: number, z: number) => v(x * s, y * s, z * s);
  const radial = (a: number, r: number, y: number) => p(Math.cos(a) * r, y, Math.sin(a) * r);
  const branches: WoodyPlantBranch[] = [];
  const wood: SceneryTaperedSweepClusterNode[] = [];
  const leaves: SceneryRecursiveShapeNode[] = [];
  const denseBush = spec.species === "boxwood" || spec.species === "hebe";
  const shrub = spec.species === "azalea" || spec.species === "juniper" || denseBush;
  const tags = [shrub ? "shrub" : "tree", spec.species, "woody-plant"];

  // Child starts are taken from the parent, never independently recomputed.
  const branch = (parent: WoodyPlantBranch | null, to: Vec3, radius: number, root = p(0, 0, 0)): WoodyPlantBranch => {
    const from = parent?.to ?? root;
    const baseRadius = Math.min(radius * s, parent?.tipRadius ?? Infinity);
    const tipRadius = baseRadius * .70;
    const id = `${spec.key}/structure/${branches.length}`;
    const segment = { id, parent: parent?.id ?? null, from, to, baseRadius, tipRadius };
    const center = mix(from, to, .5), bend = mix(from, to, .52);
    const span = Math.hypot(to.x - from.x, to.y - from.y, to.z - from.z);
    const angle = random() * Math.PI * 2, bow = span * .065;
    bend.x += Math.cos(angle) * bow;
    bend.z += Math.sin(angle) * bow;
    const points = [
      { position: v(from.x - center.x, from.y - center.y, from.z - center.z), radius: baseRadius },
      { position: v(bend.x - center.x, bend.y - center.y, bend.z - center.z), radius: (baseRadius + tipRadius) / 2 },
      { position: v(to.x - center.x, to.y - center.y, to.z - center.z), radius: tipRadius },
    ];
    const smoothRadius = .0006 * s;
    const bound = Math.max(...points.map(point => Math.hypot(...Object.values(point.position)) + point.radius)) + smoothRadius;
    wood.push({ kind: "cluster", field: "tapered-sweep", id, tags: [...tags, "structure"],
      place: { units: "metres", position: center }, lobe: v(bound, bound, bound),
      smoothRadius, seed: spec.seed >>> 0, points, material: spec.bark });
    branches.push(segment);
    return segment;
  };
  const spray = (shoot: WoodyPlantBranch, radii: readonly [number, number, number], needle = false) => {
    const size = .83 + .34 * random();
    const radii_m = radii.map(r => r * size * s) as [number, number, number];
    const period = Math.min(...radii_m) * (denseBush ? .85 : needle ? 1.25 : .85);
    let node: SceneryRecursiveShapeNode = {
      kind: "recursive-shape", family: "foliage-pad", id: `${spec.key}/foliage/${leaves.length}`,
      tags: [...tags, "foliage"], seed: hash(spec.seed + ++salt),
      place: { units: "metres", position: shoot.to, orientation: alongAxis(needle
        ? v(shoot.to.x - shoot.from.x, shoot.to.y - shoot.from.y, shoot.to.z - shoot.from.z)
        : v(.25 * (random() - .5), 1, .25 * (random() - .5))) },
      form: { radii_m, flatten: 1, edgeLobes: 6, lobeDepth: .6, topBias: .4,
        undersideCut: .65, blockJitter: .5,
        density: { clusterPeriod_m: period, dotSpacing_m: period * (denseBush ? .7 : .58),
          threshold: denseBush ? .36 : needle ? .46 : .48,
          clusterWeight: denseBush ? .40 : .32, detailWeight: denseBush ? .60 : .68, interiorBias: denseBush ? .23 : .10 } },
      split: { pattern: "cap", childCount: 3, childScale: .62, spread: .8, overlap: .2,
        verticalBias: .15, flattening: 0, jitter: .4 }, material: spec.foliage,
    };
    // Seat each leaf field on its shoot using the same analytic test as oak.
    for (let attempt = 0; attempt < 256; attempt++) {
      if (sampleSvoPrimitive(oakFoliageDescriptor(node), shoot.to).signedDistance_m < -.000015 * s) {
        leaves.push(node);
        return;
      }
      node = { ...node, seed: hash(node.seed + 1) };
    }
    throw new Error(`Cannot seat foliage on ${shoot.id}`);
  };
  const fork = (stem: WoodyPlantBranch, angle: number, reach: number, rise: number, depth: number, leafSize: number) => {
    if (!depth) { spray(stem, [leafSize, leafSize * .65, leafSize * .85]); return; }
    for (const side of [-1, 1]) {
      const a = angle + side * (.40 + random() * .25);
      const tip = v(stem.to.x + Math.cos(a) * reach * s, stem.to.y + rise * s * (.8 + .4 * random()), stem.to.z + Math.sin(a) * reach * s);
      const child = branch(stem, tip, stem.tipRadius / s * .66);
      fork(child, a, reach * .65, rise * .7, depth - 1, leafSize);
    }
  };

  if (spec.species === "birch") {
    // Unequal stems and airy ascending shoots, with a visible slender leader.
    for (let stem = 0; stem < (miniature ? 2 : 3); stem++) {
      const heading = stem * goldenAngle + random(), height = 1 - stem * .12;
      let trunk: WoodyPlantBranch | null = null;
      for (let tier = 0; tier < 6; tier++) {
        const t = (tier + 1) / 6;
        trunk = branch(trunk, radial(heading + .1 * random(), .10 * t * (stem + .5), height * t), .025 * (1 - .65 * t));
        if (!tier) continue;
        for (let side = 0; side < 2; side++) {
          const a = heading + tier * goldenAngle + side * Math.PI;
          const reach = .20 * (1 - .55 * t) * (.8 + random() * .4);
          const bough = branch(trunk, v(trunk.to.x + Math.cos(a) * reach * s, trunk.to.y + .06 * s, trunk.to.z + Math.sin(a) * reach * s), .009 * (1 - .5 * t));
          fork(bough, a, .065, .045, miniature ? 1 : 2, miniature ? .08 : .065);
        }
      }
      spray(trunk!, [.045, .075, .045]);
    }
  } else if (spec.species === "pine") {
    let trunk: WoodyPlantBranch | null = null;
    const tiers = miniature ? 6 : 8, steps = miniature ? 2 : 3;
    for (let tier = 0; tier < tiers; tier++) {
      const level = tier / (tiers - 1), y = .16 + level * .805;
      trunk = branch(trunk, p(.012 * Math.sin(tier * .7), y, .056 * level), .045 * (1 - level * 7 / 9));
      const count = tier === tiers - 1 ? 3 : miniature ? 4 : 5;
      for (let arm = 0; arm < count; arm++) {
        const a = arm * Math.PI * 2 / count + tier * 1.13 + .15 * random();
        const reach = (.38 - level * .308) * (.85 + .3 * random());
        let limb = trunk;
        for (let step = 1; step <= steps; step++) {
          const t = step / steps;
          limb = branch(limb, v(trunk.to.x + Math.cos(a) * reach * t * s, (y - .065 * Math.sin(t * Math.PI) + .035 * t) * s, trunk.to.z + Math.sin(a) * reach * t * s), .012 * (1 - t * .65));
          for (const side of [-1, 1]) {
            const heading = a + side * .75;
            const tip = branch(limb, v(limb.to.x + Math.cos(heading) * .055 * s, limb.to.y + .025 * s, limb.to.z + Math.sin(heading) * .055 * s), .003);
            spray(tip, [.045, .085, .035], true);
          }
        }
      }
    }
    spray(trunk!, [.04, .085, .04], true);
  } else if (spec.species === "willow") {
    const base = branch(null, p(-.025, .25, 0), .055);
    const trunk = branch(base, p(.015, .48, .012), .037);
    for (let arm = 0; arm < (miniature ? 7 : 9); arm++) {
      const a = arm * goldenAngle, reach = .25 + random() * .16;
      const shoulder = branch(trunk, radial(a, reach * .65, .70 + .09 * random()), .021);
      const bough = branch(shoulder, radial(a, reach, .76 + .09 * random()), .014);
      const twigs = miniature ? 3 : 4, steps = miniature ? 3 : 4;
      for (let twig = 0; twig < twigs; twig++) {
        const heading = a + (twig - (twigs - 1) / 2) * .26;
        const fall = .32 + random() * .22;
        let hanging = bough;
        for (let step = 1; step <= steps; step++) {
          const t = step / steps, outward = .14 * Math.sin(t * Math.PI * .5);
          hanging = branch(hanging, v(bough.to.x + Math.cos(heading) * outward * s,
            bough.to.y - fall * t * t * s, bough.to.z + Math.sin(heading) * outward * s), .0045 * (1 - t * .6));
          spray(hanging, [.033, .073, .028], true);
        }
      }
    }
  } else if (spec.species === "azalea") {
    for (let stem = 0; stem < 9; stem++) {
      const a = stem * goldenAngle, r = .10 + random() * .12;
      const height = .16 + random() * .28;
      const base = branch(null, radial(a, r, height * .6), .018, radial(a, .025, 0));
      const bough = branch(base, radial(a, r * 1.6, height), .011);
      spray(base, [.09, .07, .08]);
      spray(bough, [.115, .08, .10]);
      fork(bough, a, .09, .045, 2, .13);
    }
  } else if (denseBush) {
    // Overlapping, fine-grained sprays occupy the whole bush, including its
    // lower skirt. Unequal basal stems support every spray. Staggered tiers
    // and sector widths leave small seams and an irregular living silhouette.
    // Boxwood rises into a rounded crown; hebe makes a wider, lower mound.
    const boxwood = spec.species === "boxwood";
    const height = boxwood ? .56 : .40, width = boxwood ? .32 : .41;
    const stems = miniature ? 8 : 9;
    for (let stem = 0; stem < stems; stem++) {
      const a = stem * goldenAngle + (random() - .5) * .25;
      const vigor = .87 + .22 * random();
      let leader: WoodyPlantBranch | null = null;
      for (let tier = 0; tier < 3; tier++) {
        const t = (tier + .35 + random() * .65) / 3.4;
        const y = height * t * vigor;
        const reach = width * Math.sqrt(1 - (t * 1.4 - .4) ** 2) * vigor;
        leader = branch(leader, radial(a + tier * .13, reach * .38, y), .014 * (1 - t * .55), radial(a, .035, 0));
        for (const side of [-1, 1]) {
          const heading = a + side * (.20 + random() * .18);
          const shoot = branch(leader, radial(heading, reach * (.78 + .12 * random()), y + (random() - .3) * .065), .005);
          const leafRadius = boxwood ? .17 : .185;
          spray(shoot, [leafRadius, boxwood ? .14 : .13, leafRadius * .9]);
        }
        if (tier === 1) spray(leader, [width * .49, height * .28, width * .46]);
      }
    }
  } else {
    for (let stem = 0; stem < 11; stem++) {
      const a = stem * goldenAngle, reach = .25 + random() * .23;
      let limb: WoodyPlantBranch | null = null;
      for (let step = 1; step <= 3; step++) {
        const t = step / 3;
        limb = branch(limb, radial(a + .12 * Math.sin(step), reach * t, .025 + .13 * Math.sin(t * Math.PI * .8) + .04 * random()), .020 * (1 - t * .65));
        for (const side of [-1, 1]) {
          const heading = a + side * .7;
          const tip = branch(limb, v(limb.to.x + Math.cos(heading) * .085 * s, limb.to.y + .055 * s, limb.to.z + Math.sin(heading) * .085 * s), .004);
          spray(tip, [.055, .105, .045], true);
        }
      }
    }
  }
  return { branches, node: { kind: "group", id: spec.key, tags, children: [
    { kind: "group", id: `${spec.key}/structure`, children: wood },
    { kind: "group", id: `${spec.key}/foliage`, children: leaves },
  ] } };
}
