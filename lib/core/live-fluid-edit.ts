import type { InitialLiquidVolume, Vec3 } from "./model";

/** Transient current-field operation. This never authors or rewinds scene history. */
export interface LiveFluidEdit {
  readonly operation: "add" | "remove";
  readonly shape: "ball" | "cube" | "torus";
  readonly center_m: Vec3;
  /** Outer radius; a cube uses this as its half extent on every axis. */
  readonly radius_m: number;
  /** Horizontal XZ torus tube radius, strictly below half its outer radius. */
  readonly tubeRadius_m?: number;
}
export interface LiveFluidEditResult { readonly accepted: boolean; readonly reason?: string }

export function validateLiveFluidEdit(edit: LiveFluidEdit): LiveFluidEdit {
  if (!edit || !["add", "remove"].includes(edit.operation) || !["ball", "cube", "torus"].includes(edit.shape)
    || !edit.center_m || ![edit.center_m.x, edit.center_m.y, edit.center_m.z].every(Number.isFinite)
    || !(edit.radius_m > 0) || !Number.isFinite(edit.radius_m)) throw new RangeError("Fluid edits require a finite center and positive radius.");
  const tube = edit.tubeRadius_m ?? edit.radius_m / 3;
  if (edit.shape === "torus" && (!(tube > 0) || !Number.isFinite(tube) || tube >= edit.radius_m / 2)) {
    throw new RangeError("The torus tube must be positive and smaller than half its outer radius.");
  }
  return Object.freeze({ ...edit, center_m: Object.freeze({ ...edit.center_m }),
    ...(edit.shape === "torus" ? { tubeRadius_m: tube } : {}) });
}

/** The same shape as an authored volume, so a live drop and a scene edit share one world edit. */
export function liveFluidEditVolume(input: LiveFluidEdit): InitialLiquidVolume {
  const edit = validateLiveFluidEdit(input), c = edit.center_m, r = edit.radius_m;
  if (edit.shape === "ball") return { shape: "sphere", center_m: { ...c }, radius_m: r };
  if (edit.shape === "cube") return { shape: "box", min_m: { x: c.x - r, y: c.y - r, z: c.z - r },
    max_m: { x: c.x + r, y: c.y + r, z: c.z + r } };
  // The authored torus measures to the tube centreline; a live edit measures its outer radius.
  return { shape: "torus", center_m: { ...c }, radius_m: r - edit.tubeRadius_m!, tubeRadius_m: edit.tubeRadius_m! };
}

/** The most cells one live shape's bounding box may span: a 32-cube, four 8-cell pages a side. */
export const LIVE_FLUID_EDIT_CELL_BUDGET = 32768;

/** Why this world cannot take the shape, before anything is encoded: a refused drop must change nothing. */
export function liveFluidEditRefusal(input: LiveFluidEdit, domain: {
  readonly origin_m: readonly [number, number, number];
  readonly cellSize_m: readonly [number, number, number];
  readonly dimensions: readonly [number, number, number];
}): string | undefined {
  const edit = validateLiveFluidEdit(input), c = [edit.center_m.x, edit.center_m.y, edit.center_m.z];
  const half = [edit.radius_m, edit.shape === "torus" ? edit.tubeRadius_m! : edit.radius_m, edit.radius_m];
  let cells = 1;
  for (let axis = 0; axis < 3; axis++) {
    const low = domain.origin_m[axis]!, high = low + domain.dimensions[axis]! * domain.cellSize_m[axis]!;
    if (c[axis]! + half[axis]! <= low || c[axis]! - half[axis]! >= high) return "The shape lies outside the fluid domain bounds.";
    cells *= Math.ceil(2 * half[axis]! / domain.cellSize_m[axis]! - 1e-9);
  }
  return cells > LIVE_FLUID_EDIT_CELL_BUDGET
    ? `The shape spans ${cells} cells, over the ${LIVE_FLUID_EDIT_CELL_BUDGET}-cell live edit budget; use a smaller shape.` : undefined;
}

export function liveFluidEditMode(edit: Pick<LiveFluidEdit, "operation" | "shape">): 1 | 3 | 4 | 5 | 6 | 7 {
  const shape = edit.shape === "ball" ? 1 : edit.shape === "cube" ? 3 : 4;
  return (edit.operation === "remove" ? shape === 1 ? 5 : shape + 3 : shape) as 1 | 3 | 4 | 5 | 6 | 7;
}

/** Same signed-distance convention as GPU coverage; positive means outside. */
export function liveFluidEditDistance(edit: LiveFluidEdit, point: Vec3): number {
  const x = point.x - edit.center_m.x, y = point.y - edit.center_m.y, z = point.z - edit.center_m.z;
  if (edit.shape === "ball") return Math.hypot(x, y, z) - edit.radius_m;
  if (edit.shape === "torus") return Math.hypot(Math.hypot(x, z) - (edit.radius_m - (edit.tubeRadius_m ?? edit.radius_m / 3)), y)
    - (edit.tubeRadius_m ?? edit.radius_m / 3);
  const q = [Math.abs(x) - edit.radius_m, Math.abs(y) - edit.radius_m, Math.abs(z) - edit.radius_m];
  return Math.hypot(...q.map(v => Math.max(0, v))) + Math.min(Math.max(...q), 0);
}
