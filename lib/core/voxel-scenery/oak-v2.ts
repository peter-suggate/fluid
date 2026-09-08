import type { Vec3 } from "../model";
import type { SceneryGroupNode, SceneryMaterial, SceneryRecursiveShapeNode, SceneryTaperedSweepClusterNode } from "../scenery-graph";
import { sampleSvoPrimitive, type SvoSmoothUnionClusterPrimitive } from "../../svo/contracts/svo-primitive-abi";
import { oakParameters, OAK_V2_CONTROLS, type OakV2Parameters } from "./oak-v2-parameters";
export interface OakV2Spec extends Partial<OakV2Parameters> {
  readonly key: string;
  readonly seed: number;
  /** Overall scale. The default specimen is approximately one metre tall. */
  readonly scale_m?: number;
  /** Additional binary twig forks, 0..3. Default 3; independent of voxel depth. */
  readonly twigDepth?: number;
  readonly bark: SceneryMaterial;
  readonly foliage: SceneryMaterial;
}
export interface OakV2Branch {
  readonly id: string;
  readonly parent: string | null;
  readonly from: Vec3;
  readonly to: Vec3;
  readonly baseRadius: number;
  readonly tipRadius: number;
  readonly leaves: number;
}
export interface OakV2Plan {
  readonly branches: readonly OakV2Branch[];
  readonly node: SceneryGroupNode;
}
const v = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
const mix = (a: Vec3, b: Vec3, t: number) => v(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);
function hash(n: number): number {
  n = Math.imul(n ^ (n >>> 16), 0x7feb352d);
  n = Math.imul(n ^ (n >>> 15), 0x846ca68b);
  return (n ^ (n >>> 16)) >>> 0;
}
const random = (seed: number, salt: number) => hash(seed ^ Math.imul(salt + 1, 0x9e3779b1)) / 4294967296;
function pathSeed(path: string): number {
  let result = 2166136261;
  for (let i = 0; i < path.length; i++)
    result = Math.imul(result ^ path.charCodeAt(i), 16777619);
  return result >>> 0;
}
const distance = (a: Vec3, b: Vec3) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
/** Used to certify the shoot actually ends in occupied foliage, not just its envelope. */
export function oakFoliageDescriptor(node: SceneryRecursiveShapeNode): SvoSmoothUnionClusterPrimitive {
  const d = node.form.density;
  const [x, y, z] = node.form.radii_m;
  return {
    kind: "smooth-union-cluster", primitiveId: 0, materialId: 1, clusterReference: 0,
    center_m: node.place!.position!, lobeRadii_m: v(x, y * node.form.flatten, z),
    packing: { field: "noise-foliage", seed: node.seed, smoothRadius_m: 0,
      clusterPeriod_m: d.clusterPeriod_m, detailPeriod_m: d.dotSpacing_m,
      threshold: d.threshold, clusterWeight: d.clusterWeight, detailWeight: d.detailWeight, interiorBias: d.interiorBias },
  };
}
/**
 * Crown-guided branching: spatially partition foliage sites, route each fork
 * toward its descendants, then assign radii bottom-up with r² ∝ leaf count.
 * This is a deterministic geometric construction, not a growth simulation.
 * Every fork starts exactly at its parent's endpoint; round-cone sweeps share
 * endpoint spheres. Refinement changes sampling, never this topology.
 *
 * The output is ordinary editable document geometry, with no runtime generator
 * or hidden LOD. Macro gaps come from separate boughs; the density field only
 * supplies shoot-scale breakup, never a single canopy-sized envelope.
 */
export function planOakV2(spec: OakV2Spec): OakV2Plan {
  const parameters = oakParameters(Object.fromEntries(Object.entries(spec).filter(([key, value]) => key in OAK_V2_CONTROLS && value !== undefined)));
  const s = parameters.scale_m;
  const q = parameters;
  if (!Number.isFinite(s) || s <= 0 || !Number.isInteger(spec.seed))
    throw new RangeError("Oak v2 needs a positive finite scale and integer seed");
  const p = (x: number, y: number, z: number) => v(x * s, y * s, z * s);
  const branches: OakV2Branch[] = [];
  const branchById = new Map<string, OakV2Branch>();
  const wood: SceneryTaperedSweepClusterNode[] = [];
  const foliage: SceneryRecursiveShapeNode[] = [];
  const twigDepth = q.twigDepth;
  if (!Number.isInteger(twigDepth) || twigDepth < 0 || twigDepth > 3)
    throw new RangeError("Oak v2 twig depth must be an integer in 0..3");
  const shootsPerSite = 2 ** twigDepth;
  const shootRadius = 0.0042 * s * q.woodScale / Math.sqrt(shootsPerSite);
  const pipe = (count: number) => shootRadius * Math.sqrt(count);
  const addBranch = (id: string, parent: string | null, from: Vec3, to: Vec3, leaves: number, baseRadius: number, tipRadius: number) => {
    if (parent) {
      const upstream = branchById.get(parent)!;
      baseRadius = upstream.tipRadius * Math.sqrt(leaves / upstream.leaves);
      tipRadius = Math.min(baseRadius, Math.max(.001 * s * q.woodScale, baseRadius * (leaves === 1 ? .80 : .97)));
    }
    const branch = { id, parent, from, to, leaves, baseRadius, tipRadius };
    branches.push(branch);
    branchById.set(id, branch);
    // A slightly bowed centreline, with a common sphere at each fork. The
    // envelope includes smooth-min inflation as well as every control sphere.
    const bend = mix(from, to, 0.52);
    const span = distance(from, to);
    const bow = Math.min(0.045 * s, span * 0.22) * q.branchBend;
    const phase = random(spec.seed, pathSeed(id.slice(spec.key.length))) * Math.PI * 2;
    bend.x += Math.cos(phase) * bow;
    bend.z += Math.sin(phase) * bow;
    bend.y -= bow * 0.45;
    const center = mix(from, to, 0.5);
    const points = [
      { position: v(from.x - center.x, from.y - center.y, from.z - center.z), radius: baseRadius },
      { position: v(bend.x - center.x, bend.y - center.y, bend.z - center.z), radius: (baseRadius + tipRadius) * 0.5 },
      { position: v(to.x - center.x, to.y - center.y, to.z - center.z), radius: tipRadius },
    ];
    const smoothRadius = 0.001 * s;
    const radius = Math.max(...points.map(point => Math.hypot(point.position.x, point.position.y, point.position.z) + point.radius)) + smoothRadius;
    wood.push({ kind: "cluster", field: "tapered-sweep", id, group: id,
      tags: ["tree", "structure", "oak-v2"], place: { units: "metres", position: center },
      lobe: v(radius, radius, radius), smoothRadius, seed: spec.seed >>> 0,
      points, material: spec.bark });
  };
  const trunkNodes = [p(0, 0, 0), p(-0.012, 0.16, 0.004), p(0.006, 0.255, 0.009), p(-0.018, 0.35, 0.014), p(-0.035, 0.44, 0.008)].map(point => ({ ...point, y: point.y * q.trunkHeight }));
  const sitesPerTier = q.boughsPerTier * q.sitesPerBough;
  const siteCount = 4 * sitesPerTier;
  // Two boughs at each of four fork sites: 96 shoot sites, each carrying a recursive twig spray.
  for (let i = 0; i < 4; i++) {
    addBranch(`${spec.key}/structure/trunk-${i}`, i ? `${spec.key}/structure/trunk-${i - 1}` : null, trunkNodes[i], trunkNodes[i + 1], (siteCount - i * sitesPerTier) * shootsPerSite, i === 0 ? 0.055 * s * q.woodScale : pipe((siteCount - i * sitesPerTier) * shootsPerSite), pipe((siteCount - i * sitesPerTier) * shootsPerSite));
  }
  for (let bough = 0; bough < 4 * q.boughsPerTier; bough++) {
    const tier = Math.floor(bough / q.boughsPerTier);
    const angle = bough * Math.PI * (3 - Math.sqrt(5)) + 0.3 * (random(spec.seed, bough) - 0.5);
    const reach = (0.40 - tier * 0.065) * (0.90 + 0.20 * random(spec.seed, 40 + bough));
    const crownCenter = p((-0.035 + Math.cos(angle) * reach) * q.crownWidth, .44 * q.trunkHeight + (.04 + tier * .09) * q.crownHeight, Math.sin(angle) * reach * 0.85 * q.crownWidth);
    const sites: {
      position: Vec3;
      index: number;
    }[] = [];
    for (let i = 0; i < q.sitesPerBough; i++) {
      // Stratified sites avoid both a regular sphere lattice and coincident
      // shoots. Their broad vertical range keeps the crown volumetric.
      const a = i * 2.399963229728653 + angle;
      const radius = Math.sqrt((i + 0.5) / q.sitesPerBough);
      const site = v(crownCenter.x + Math.cos(a) * radius * 0.21 * s * q.crownWidth, crownCenter.y + (random(spec.seed, 100 + bough * 12 + i) - 0.35) * 0.21 * s * q.crownHeight, crownCenter.z + Math.sin(a) * radius * 0.17 * s * q.crownWidth);
      sites.push({ position: site, index: i });
    }
    const grow = (sites: readonly {
      position: Vec3;
      index: number;
    }[], from: Vec3, parent: string, path: string): void => {
      const centroid = v(0, 0, 0);
      for (const site of sites) {
        centroid.x += site.position.x / sites.length;
        centroid.y += site.position.y / sites.length;
        centroid.z += site.position.z / sites.length;
      }
      const to = sites.length === 1 ? centroid : mix(from, centroid, 0.58);
      const id = `${spec.key}/structure/bough-${bough}/${path}`;
      addBranch(id, parent, from, to, sites.length * shootsPerSite, pipe(sites.length * shootsPerSite), pipe(sites.length * shootsPerSite));
      if (sites.length === 1) {
        const sprout = (tip: Vec3, direction: Vec3, parentId: string, remaining: number, twigPath: string, twigSalt: number): void => {
          if (remaining > 0) {
            const norm = Math.hypot(direction.x, direction.y, direction.z);
            const axis = v(direction.x / norm, direction.y / norm, direction.z / norm);
            // Rotate the fork plane each generation, avoiding planar fans and
            // rings of lollipops. Each child inherits its parent's direction.
            const reference = Math.abs(axis.y) < .9 ? v(0, 1, 0) : v(1, 0, 0);
            const cross = (a: Vec3, b: Vec3) => v(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
            const u0 = cross(axis, reference), un = Math.hypot(u0.x, u0.y, u0.z);
            const u = v(u0.x / un, u0.y / un, u0.z / un), w = cross(axis, u);
            const angle = twigSalt * 2.399963229728653;
            const side = v(u.x * Math.cos(angle) + w.x * Math.sin(angle), u.y * Math.cos(angle) + w.y * Math.sin(angle), u.z * Math.cos(angle) + w.z * Math.sin(angle));
            for (let child = 0; child < 2; child++) {
              const sign = child ? 1 : -1;
              const lateral = .72 * Math.tan(q.forkAngle * Math.PI / 180);
              const nextDirection = v(axis.x * .72 + side.x * sign * lateral, axis.y * .72 + side.y * sign * lateral + q.upwardBias, axis.z * .72 + side.z * sign * lateral);
              const dn = Math.hypot(nextDirection.x, nextDirection.y, nextDirection.z);
              const reach = .075 * s * q.twigLength * q.twigDecay ** (twigDepth - remaining) * (.85 + .3 * random(spec.seed, twigSalt + child));
              const next = v(tip.x + nextDirection.x / dn * reach, tip.y + nextDirection.y / dn * reach, tip.z + nextDirection.z / dn * reach);
              const childPath = `${twigPath}/${child ? "b" : "a"}`;
              const childId = `${id}/twig/${childPath}`;
              const count = 2 ** (remaining - 1);
              addBranch(childId, parentId, tip, next, count, pipe(count), pipe(count));
              sprout(next, nextDirection, childId, remaining - 1, childPath, hash(twigSalt + child + 1));
            }
            return;
          }
          if (!q.showFoliage) return;
          const size = q.leafScale * 0.62 ** twigDepth * (0.90 + 0.20 * random(spec.seed, 500 + bough * 12 + sites[0].index)) * s;
          let node: SceneryRecursiveShapeNode = {
            kind: "recursive-shape", family: "foliage-pad", id: `${spec.key}/foliage/bough-${bough}/shoot-${sites[0].index}/${twigPath}`,
            tags: ["tree", "foliage", "oak-v2"], seed: hash(spec.seed + bough * 97 + sites[0].index + twigSalt),
            place: { units: "metres", position: tip },
            form: { radii_m: [0.105 * size, 0.073 * size, 0.090 * size], flatten: q.leafFlatten,
              edgeLobes: 6, lobeDepth: 0.6, topBias: 0.4, undersideCut: 0.65, blockJitter: 0.5,
              density: { clusterPeriod_m: 0.060 * size * q.clumpScale, dotSpacing_m: 0.018 * size,
                threshold: q.leafThreshold, clusterWeight: 1 - q.detailWeight, detailWeight: q.detailWeight, interiorBias: q.interiorBias } },
            split: { pattern: "cap", childCount: 3, childScale: 0.62, spread: 0.8, overlap: 0.2, verticalBias: 0.15, flattening: 0, jitter: 0.4 },
            material: spec.foliage,
          };
          // Fix the density phase, not the wood position, until the centre is
          // inside a leaf cluster. Bounded deterministic search; no floating tufts.
          let supported = false;
          for (let attempt = 0; attempt < 256; attempt++) {
            if (sampleSvoPrimitive(oakFoliageDescriptor(node), tip).signedDistance_m < -0.000015 * s) {
              supported = true;
              break;
            }
            node = { ...node, seed: hash(node.seed + 1) };
          }
          if (!supported)
            throw new Error(`Oak v2 cannot seat foliage ${node.id}`);
          foliage.push(node);
        };
        sprout(to, v(to.x - from.x, to.y - from.y + .02 * s, to.z - from.z), id, twigDepth, "tip", hash(spec.seed + bough * 97 + sites[0].index));
        return;
      }
      const axes = ["x", "y", "z"] as const;
      const axis = axes.reduce((best, axis) => {
        const range = (a: typeof axis) => Math.max(...sites.map(site => site.position[a])) - Math.min(...sites.map(site => site.position[a]));
        return range(axis) > range(best) ? axis : best;
      }, "x");
      const sorted = [...sites].sort((a, b) => a.position[axis] - b.position[axis] || a.index - b.index);
      const half = Math.floor(sorted.length / 2);
      grow(sorted.slice(0, half), to, id, `${path}/a`);
      grow(sorted.slice(half), to, id, `${path}/b`);
    };
    grow(sites, trunkNodes[tier + 1], `${spec.key}/structure/trunk-${tier}`, "stem");
  }
  return { branches, node: { kind: "group", id: spec.key, tags: ["tree", "oak-v2", "shape-lab"], oak: { version: 1, parameters, bark: spec.bark, foliage: spec.foliage }, children: [
        { kind: "group", id: `${spec.key}/structure`, children: wood },
        { kind: "group", id: `${spec.key}/foliage`, children: foliage },
      ] } };
}
