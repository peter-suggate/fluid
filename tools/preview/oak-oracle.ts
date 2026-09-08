import type { SceneryNode } from "../../lib/core/scenery-graph";
import type { Vec3 } from "../../lib/core/model";
import { oakFoliageDescriptor } from "../../lib/core/voxel-scenery/oak-v2";
import type { SvoSmoothUnionClusterPrimitive } from "../../lib/svo/contracts/svo-primitive-abi";
/** Root-local descriptors for isolated analytic and voxel CPU comparisons. */
export function oakOracleDescriptors(tree: SceneryNode, bare = false): SvoSmoothUnionClusterPrimitive[] {
  const descriptors: SvoSmoothUnionClusterPrimitive[] = [];
  function visit(node: SceneryNode, parent: Vec3, root = false) {
    const place = root ? { x: 0, y: 0, z: 0 } : node.place?.position ?? { x: 0, y: 0, z: 0 };
    const at = { x: parent.x + place.x, y: parent.y + place.y, z: parent.z + place.z };
    if (node.kind === "group" || node.kind === "recursive-shape" && node.children?.length) {
      for (const child of node.children!)
        visit(child, at);
    }
    else if (node.kind === "recursive-shape") {
      if (bare)
        return;
      descriptors.push({ ...oakFoliageDescriptor(node), center_m: at, primitiveId: descriptors.length, materialId: 2 });
    }
    else if (node.kind === "cluster" && node.field === "tapered-sweep") {
      descriptors.push({ kind: "smooth-union-cluster", primitiveId: descriptors.length, materialId: 1,
        center_m: at, lobeRadii_m: node.lobe, clusterReference: 0,
        packing: { field: "tapered-sweep", seed: node.seed, smoothRadius_m: node.smoothRadius,
          points: node.points.map(p => ({ position_m: p.position, radius_m: p.radius })) } });
    }
    else
      throw new Error(`Unsupported oracle node ${node.id}`);
  }
  visit(tree, { x: 0, y: 0, z: 0 }, true);
  return descriptors;
}
