import type {
  FoliagePadForm,
  SceneryNoiseFoliageClusterNode,
  SceneryRecursiveShapeNode,
} from "../scenery-graph";

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value));

function hash32(value: number): number {
  let x = value | 0;
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
  return (x ^ (x >>> 16)) >>> 0;
}

function random01(seed: number, salt: number): number {
  return hash32((seed | 0) ^ Math.imul(salt + 1, 0x9e3779b1)) / 0xffff_ffff;
}

/** Compile one active foliage frontier leaf to one bounded density field. */
export function foliagePadClusterNode(node: SceneryRecursiveShapeNode): SceneryNoiseFoliageClusterNode {
  const [rx, ry, rz] = node.form.radii_m;
  const density = node.form.density;
  return {
    kind: "cluster",
    id: node.id,
    group: node.group ?? node.id,
    tags: node.tags ?? ["tree", "foliage", "recursive-shape"],
    field: "noise-foliage",
    lobe: { x: rx, y: ry * node.form.flatten, z: rz },
    // `smoothRadius` belongs to the shared cluster header. A density field has
    // no lobe unions to smooth, so zero is both valid and intentional.
    smoothRadius: 0,
    seed: node.seed,
    // These are document values rather than compiler tuning constants. Shape
    // Lab edits this same object, exports it in the node override, and every
    // production publication path therefore receives exactly the tuned field.
    clusterPeriod: density.clusterPeriod_m,
    // The renderer calls this its detail period; the document calls it dot
    // spacing because that is the art-direction decision it represents.
    detailPeriod: density.dotSpacing_m,
    threshold: density.threshold,
    clusterWeight: density.clusterWeight,
    detailWeight: density.detailWeight,
    interiorBias: density.interiorBias,
    material: node.material,
  };
}

function childForm(parent: FoliagePadForm, scale: number, flattening: number, jitter: number): FoliagePadForm {
  return {
    ...parent,
    radii_m: [parent.radii_m[0] * scale, parent.radii_m[1] * scale, parent.radii_m[2] * scale],
    flatten: Math.max(0.36, parent.flatten * (1 - 0.18 * flattening)),
    edgeLobes: Math.max(4, Math.min(12, Math.round(parent.edgeLobes - 1 + jitter * 3))),
    lobeDepth: clamp01(parent.lobeDepth + 0.08 + jitter * 0.12),
    blockJitter: clamp01(parent.blockJitter + jitter * 0.12),
    density: {
      ...parent.density,
      clusterPeriod_m: parent.density.clusterPeriod_m * scale,
      dotSpacing_m: parent.density.dotSpacing_m * scale,
    },
  };
}

/**
 * Materialize one recursive frontier step. Existing descendants are authority
 * and are returned untouched; regeneration is an explicit caller decision.
 */
export function refineFoliageShape(node: SceneryRecursiveShapeNode): SceneryRecursiveShapeNode {
  if (node.children?.length) return node;
  const count = Math.max(3, Math.min(5, Math.trunc(node.split.childCount)));
  const [rx, ry, rz] = node.form.radii_m;
  const patternArc = node.split.pattern === "fan" ? Math.PI * 1.15 : Math.PI * 2;
  const start = node.split.pattern === "fan" ? -patternArc * 0.5 : 0;
  const children: SceneryRecursiveShapeNode[] = [];
  for (let index = 0; index < count; index += 1) {
    const jitter = (random01(node.seed, index * 5) - 0.5) * node.split.jitter;
    const scale = node.split.childScale * (0.91 + 0.18 * random01(node.seed, index * 5 + 1));
    const angle = start + patternArc * (node.split.pattern === "fan"
      ? index / Math.max(1, count - 1)
      : index / count) + jitter * 0.45;
    // Use the available parent-minus-child radius instead of clustering every
    // child at the parent's centre. This keeps siblings joined, but carries the
    // recursive frontier far enough outward to leave a readable cloud edge.
    const radial = (1 - scale) * (0.70 + 0.50 * node.split.spread) * (1 - 0.25 * node.split.overlap);
    const capCentre = node.split.pattern === "cap" && index === 0;
    const x = capCentre ? 0 : Math.cos(angle) * rx * radial;
    const z = capCentre ? 0 : Math.sin(angle) * rz * radial;
    const top = node.form.topBias * ry * 0.18;
    const underside = Math.max(0, -Math.sin(angle)) * node.form.undersideCut * ry * 0.16;
    // Preserve the reference's calm lower shelf while letting the recursive
    // frontier build a genuinely volumetric cloud above it. A strictly
    // positive split offset made the leaf frontier collapse into a thin plate.
    const verticalScatter = (random01(node.seed, index * 5 + 3) - 0.32) * ry * 0.34 * node.split.spread;
    const y = top + node.split.verticalBias * ry * (0.10 + 0.10 * random01(node.seed, index * 5 + 2))
      + underside + verticalScatter;
    const suffix = String.fromCharCode(97 + index);
    children.push({
      kind: "recursive-shape",
      family: "foliage-pad",
      id: `${node.id}/${suffix}`,
      seed: hash32(node.seed ^ Math.imul(index + 1, 0x45d9f3b)),
      place: { units: "metres", position: { x, y, z } },
      form: childForm(node.form, scale, node.split.flattening, Math.abs(jitter)),
      split: { ...node.split, childScale: Math.max(0.46, node.split.childScale * 0.94) },
      material: node.material,
    });
  }
  return { ...node, children };
}

/** Materialize a complete deterministic subtree for authored seed data. */
export function refineFoliageShapeToDepth(
  node: SceneryRecursiveShapeNode,
  depth: number,
): SceneryRecursiveShapeNode {
  if (depth <= 0) return node;
  const refined = refineFoliageShape(node);
  return {
    ...refined,
    children: refined.children?.map((child) => refineFoliageShapeToDepth(child, depth - 1)),
  };
}

export function activeFoliageLeaves(node: SceneryRecursiveShapeNode): readonly SceneryRecursiveShapeNode[] {
  return node.children?.length ? node.children.flatMap(activeFoliageLeaves) : [node];
}
