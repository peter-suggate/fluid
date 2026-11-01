import type { RigidBodyDescription } from "./model";
import { seaweedSkin } from "./seaweed-render";

/** Appearance bounds do not change the collision shape or displaced volume. */
export function rigidAppearanceSkin(body: RigidBodyDescription) {
  return seaweedSkin(body);
}
