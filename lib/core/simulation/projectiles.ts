import type { SceneDescription, Vec3 } from "../model";
import type { RigidBodyState } from "../rigid-body";
import { add, length, scale } from "../math";
import { latticePoint } from "../voxel-editor/geometry";
import { visibleToolSurface } from "../voxel-editor/surface";
import { sceneCellSizes_m } from "../scene-lattice";
import { subcellPatch } from "../solid-patch/ellipsoid-bounds";
import type { SolidWorldCoordinate, SolidWorldVoxelPatch } from "../solid-world";

interface Flight { cycle: number; position: Vec3; velocity: Vec3; stopped: boolean }

/** One instance per running pane. Launch and flight use the accepted fixed
 * clock. Only a swept contact with a surviving surface produces an edit. */
export class ProjectileSimulation {
  private readonly flights = new Map<string, Flight>();

  step(scene: SceneDescription, bodies: RigidBodyState[], time_s: number, dt_s: number): SolidWorldVoxelPatch[] {
    const patches: SolidWorldVoxelPatch[] = [];
    for (const body of bodies) {
      const { id, launcher, position_m: muzzle, dimensions_m } = body.description;
      if (!launcher || time_s < launcher.start_s) continue;
      const elapsed = time_s - launcher.start_s;
      const cycle = Math.floor(elapsed / launcher.period_s);
      let flight = this.flights.get(id);
      let dt = dt_s;
      if (!flight || flight.cycle !== cycle) {
        dt = Math.min(dt, elapsed - cycle * launcher.period_s);
        flight = { cycle, position: muzzle, velocity: launcher.velocity_m_s, stopped: false };
        this.flights.set(id, flight);
      }
      if (!flight.stopped && dt > 0) {
        const acceleration = scene.fluid.gravity_m_s2;
        const travel = add(scale(flight.velocity, dt), scale(acceleration, .5 * dt * dt));
        const distance = length(travel);
        const direction = distance > 0 ? scale(travel, 1 / distance) : { x: 0, y: -1, z: 0 };
        const hit = distance > 0 && visibleToolSurface(scene, { origin: flight.position, direction });
        const radius = dimensions_m.x;
        if (hit && hit.distance_m <= distance + radius) {
          const center = latticePoint(scene, hit.point);
          const extent = sceneCellSizes_m(scene).map(h => launcher.cutRadius_m / h);
          const corner = (sign: number) => center.map((v, i) => v + sign * extent[i]!) as unknown as SolidWorldCoordinate;
          patches.push({ ...subcellPatch(corner(-1), corner(1), "clear", "ellipsoid"), affectsScenery: true });
          flight.position = add(flight.position, scale(direction, Math.max(0, hit.distance_m - radius)));
          flight.stopped = true;
        } else {
          flight.position = add(flight.position, travel);
          flight.velocity = add(flight.velocity, scale(acceleration, dt));
          if (flight.position.y <= radius) {
            flight.position = { ...flight.position, y: radius };
            flight.stopped = true;
          }
        }
      }
      body.position_m = flight.position;
      body.linearVelocity_m_s = flight.stopped ? { x: 0, y: 0, z: 0 } : flight.velocity;
    }
    return patches;
  }
}
