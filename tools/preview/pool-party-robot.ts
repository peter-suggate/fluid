/** Bub beside the ladder, through the production dry-scene renderer. */
import { createScene as pool } from './pool-party-live';
import { interactionScene } from '../../lib/interaction/adapters/scene';
export const createScene = () => interactionScene(pool());
export const camera = { azimuth_rad: .7, elevation_rad: .22, distance_m: 4.2, tanHalfFov: .4,
  target_m: { x: -7.2, y: 2.65, z: .85 } };
