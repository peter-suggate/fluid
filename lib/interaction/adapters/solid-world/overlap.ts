import type { SceneDescription, Vec3 } from '../../../core/model';
import { sceneCellSizes_m } from '../../../core/scene-lattice-dimensions';
import { fluidSolidWorldForScene, sampleSolidWorld } from '../../../core/solid-world';
import { proceduralSolidField, sampleSolidNodes } from '../../../core/procedural-solid';
import { boundingRadius, quaternionInverseRotate, type RigidBodyState } from '../../../core/rigid-body';
import { sceneShape } from '../../../core/scene-shape';
/** Fixed-resolution solid predicate using the same validity/override rules as collision. */
function solidAt(scene: SceneDescription) {
  const world = fluidSolidWorldForScene(scene), field = proceduralSolidField(scene), h = sceneCellSizes_m(scene);
  return (p: Vec3) => {
    const c = [(p.x+scene.container.width_m/2)/h[0], p.y/h[1], (p.z+scene.container.depth_m/2)/h[2]];
    const cell = c.map(Math.floor) as [number,number,number], at=field?.at(cell);
    return at ? sampleSolidNodes(at.values, at.local.map((v,i)=>v+c[i]-cell[i]) as [number,number,number])[3] < -.005 : sampleSolidWorld(world,cell).solidFraction>0;
  };
}
function interior(body: RigidBodyState, p: Vec3): boolean {
  const d = {x:p.x-body.position_m.x,y:p.y-body.position_m.y,z:p.z-body.position_m.z};
  return sceneShape(body.description.shape).distance_m(body.description.dimensions_m,quaternionInverseRotate(body.orientation,d)) < -.015;
}
/** Reject newly occupied interior samples; existing contact penetration is retained. */
export function assertNoSolidOverlap(scene: SceneDescription, bodies: readonly RigidBodyState[], previous?: SceneDescription, otherBodies: readonly RigidBodyState[] = []) {
  const solid=solidAt(scene), old=previous?solidAt(previous):undefined;
  const spacing=Math.min(.1,...sceneCellSizes_m(scene).map(h=>h/2));
  let samples=0;
  for(const body of bodies) {
    const radius=boundingRadius(body.description), n=Math.ceil(radius/spacing), p=body.position_m;
    for(let z=-n;z<=n;z++)for(let y=-n;y<=n;y++)for(let x=-n;x<=n;x++) {
      if(++samples>2_000_000)throw new Error('Overlap validation exceeds supported sampling budget');
      const point={x:p.x+x*spacing,y:p.y+y*spacing,z:p.z+z*spacing};
      if(!interior(body,point))continue;
      if(solid(point)&&!old?.(point) || otherBodies.some(b=>b.description.id!==body.description.id&&interior(b,point))) throw new Error(`Placement or solid edit overlaps ${body.description.name}`);
    }
  }
}
