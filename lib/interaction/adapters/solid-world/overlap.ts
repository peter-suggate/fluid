import type { SceneDescription, Vec3 } from '../../../core/model';
import { sceneCellSizes_m } from '../../../core/scene-lattice-dimensions';
import { fluidSolidWorldForScene, sampleSolidWorld } from '../../../core/solid-world';
import { proceduralSolidField, sampleSolidNodes } from '../../../core/procedural-solid';
import { boundingRadius, quaternionInverseRotate, type RigidBodyState } from '../../../core/rigid-body';
import { sceneShape } from '../../../core/scene-shape';
/** Fixed-resolution solid predicate using the same validity/override rules as collision. */
function solidAt(scene: SceneDescription) {
  const world = fluidSolidWorldForScene(scene), field = proceduralSolidField(scene), h = sceneCellSizes_m(scene);
  const origin = [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2];
  return { world, field, h, origin, at: (p: Vec3) => {
    const c = [(p.x+scene.container.width_m/2)/h[0], p.y/h[1], (p.z+scene.container.depth_m/2)/h[2]];
    const cell = c.map(Math.floor) as [number,number,number], at=field?.at(cell);
    return at ? sampleSolidNodes(at.values, at.local.map((v,i)=>v+c[i]-cell[i]) as [number,number,number])[3] < -.005 : sampleSolidWorld(world,cell).solidFraction>0;
  } };
}
/** A remote edit cannot introduce penetration where both predicates still
 * read the same pages and sampled nodes. Check the body's current bounds,
 * rather than retaining a validation across a possible physics step. */
function unchangedNearBody(next: ReturnType<typeof solidAt>, previous: ReturnType<typeof solidAt>, body: RigidBodyState): boolean {
  const radius = boundingRadius(body.description), p = body.position_m;
  const lower = [p.x, p.y, p.z].map((v, a) => Math.floor((v - radius - next.origin[a]) / next.h[a] / 8));
  const upper = [p.x, p.y, p.z].map((v, a) => Math.floor((v + radius - next.origin[a]) / next.h[a] / 8));
  if (lower.reduce((n, v, a) => n * (upper[a] - v + 1), 1) > 4096) return false;
  for (let z = lower[2]; z <= upper[2]; z++) for (let y = lower[1]; y <= upper[1]; y++) for (let x = lower[0]; x <= upper[0]; x++) {
    const coordinate = [x, y, z] as const;
    const a = next.world.directory.lookup(coordinate), b = previous.world.directory.lookup(coordinate);
    if ((a === undefined ? undefined : next.world.pages[a]) !== (b === undefined ? undefined : previous.world.pages[b])) return false;
    const key = coordinate.join(','), field = next.field?.bricks.get(key), oldField = previous.field?.bricks.get(key);
    if (field?.values !== oldField?.values || Boolean(field) !== Boolean(oldField)) return false;
    if (field && oldField && field.valid.some((word, i) => word !== oldField.valid[i])) return false;
  }
  return true;
}
function interior(body: RigidBodyState, p: Vec3): boolean {
  const d = {x:p.x-body.position_m.x,y:p.y-body.position_m.y,z:p.z-body.position_m.z};
  return sceneShape(body.description.shape).distance_m(body.description.dimensions_m,quaternionInverseRotate(body.orientation,d)) < -.015;
}
/** Reject newly occupied interior samples; existing contact penetration is retained. */
export function assertNoSolidOverlap(scene: SceneDescription, bodies: readonly RigidBodyState[], previous?: SceneDescription, otherBodies: readonly RigidBodyState[] = []) {
  const solid=solidAt(scene), old=previous?solidAt(previous):undefined;
  const comparable = old && solid.world.ground === old.world.ground
    && solid.h.every((v, i) => v === old.h[i]) && solid.origin.every((v, i) => v === old.origin[i])
    && JSON.stringify(solid.world.regions) === JSON.stringify(old.world.regions);
  const spacing=Math.min(.1,...sceneCellSizes_m(scene).map(h=>h/2));
  let samples=0;
  for(const body of bodies) {
    if (comparable && otherBodies.length === 0 && unchangedNearBody(solid, old, body)) continue;
    const radius=boundingRadius(body.description), n=Math.ceil(radius/spacing), p=body.position_m;
    for(let z=-n;z<=n;z++)for(let y=-n;y<=n;y++)for(let x=-n;x<=n;x++) {
      if(++samples>2_000_000)throw new Error('Overlap validation exceeds supported sampling budget');
      const point={x:p.x+x*spacing,y:p.y+y*spacing,z:p.z+z*spacing};
      if(!interior(body,point))continue;
      if(solid.at(point)&&!old?.at(point) || otherBodies.some(b=>b.description.id!==body.description.id&&interior(b,point))) throw new Error(`Placement or solid edit overlaps ${body.description.name}`);
    }
  }
}
