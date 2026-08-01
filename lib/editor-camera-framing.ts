import { defaultCamera, type CameraState } from "./model";

/**
 * Standard camera framings, on the keyboard rather than in a toolbar.
 *
 * These were four permanent buttons in the viewport's bottom edge. The buttons
 * are gone — the frame was ringed with chrome on all four sides — but the
 * capability could not go with them: `reset` is the only way back from a camera
 * orbited into empty space, and losing it would mean reloading the page.
 */

export type CameraFraming = "reset" | "front" | "side" | "top";

export interface CameraFramingSpec {
  readonly id: CameraFraming;
  readonly label: string;
  /** Single unmodified key, chosen clear of the tool shortcuts. */
  readonly key: string;
}

export const CAMERA_FRAMINGS: readonly CameraFramingSpec[] = Object.freeze([
  { id: "reset", label: "Reset", key: "0" },
  { id: "front", label: "Front", key: "1" },
  { id: "side", label: "Side", key: "2" },
  { id: "top", label: "Top", key: "3" },
] as const satisfies readonly CameraFramingSpec[]);

export function cameraFramingForKey(key: string): CameraFraming | undefined {
  return CAMERA_FRAMINGS.find((framing) => framing.key === key)?.id;
}

export function cameraForFraming(framing: CameraFraming): CameraState {
  switch (framing) {
    case "front": return { ...defaultCamera, azimuth_rad: 0, elevation_rad: 0.08 };
    case "side": return { ...defaultCamera, azimuth_rad: Math.PI / 2, elevation_rad: 0.08 };
    case "top": return { ...defaultCamera, azimuth_rad: 0, elevation_rad: 1.34, distance_m: 2.25 };
    case "reset": return defaultCamera;
  }
}
