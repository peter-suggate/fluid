/**
 * The drawing vocabulary every pass shares to decorate a frame.
 *
 * A decoration is a world-space segment: two points, a colour, a width in
 * device pixels, an optional dash and an optional arrowhead. That is the whole
 * primitive, and it is enough for boxes, stencils, cones, rings and crosses —
 * which means one instanced draw can carry every visualization on screen no
 * matter how many passes contributed to it.
 *
 * Two conventions travel with the vocabulary so a picture cannot claim more
 * than its source knows:
 *
 * - **Solid** is measured — state a frame published, read back and drawn.
 * - **Dashed** is derived — a bound, a schedule, or anything a residual gate
 *   may have cut short. See `visualization-registry.ts`, where every decorator
 *   also declares which of the two its notes are.
 *
 * Coordinates are grid coordinates, not metres. A pass reasons in the lattice it
 * owns; `DecorationSpace` is the single place that lattice becomes world, so a
 * decorator can never disagree with the shader whose work it explains.
 */

/** Floats per instance, matching the overlay pipeline's vertex layout. */
export const DECORATION_SEGMENT_FLOATS = 16;

export type DecorationVec3 = readonly [number, number, number];

/**
 * The affine map from one pass's lattice to world metres.
 *
 * `origin_m` is where lattice coordinate zero sits and `extent_m` is the world
 * span of the whole lattice, so a pass working in a coarser or finer grid than
 * another simply declares a different space and both land correctly.
 */
export interface DecorationSpace {
  readonly dimensions: DecorationVec3;
  readonly origin_m: DecorationVec3;
  readonly extent_m: DecorationVec3;
}

/**
 * The convention the fluid container uses: the box spans the finest grid
 * exactly, floor on y = 0, centred on x and z. This is the forward transform of
 * the `worldToFine` every technique shader inverts, so anything placed through
 * it lands on the pixels it describes.
 */
export function containerDecorationSpace(
  dimensions: DecorationVec3, extent_m: DecorationVec3,
): DecorationSpace {
  return {
    dimensions,
    origin_m: [-0.5 * extent_m[0], 0, -0.5 * extent_m[2]],
    extent_m,
  };
}

export function decorationSpaceValid(space: DecorationSpace): boolean {
  return space.dimensions.every((extent) => Number.isFinite(extent) && extent > 0)
    && space.extent_m.every((extent) => Number.isFinite(extent) && extent > 0)
    && space.origin_m.every((value) => Number.isFinite(value));
}

export function decorationToWorld(space: DecorationSpace, point: DecorationVec3): DecorationVec3 {
  return [
    space.origin_m[0] + (point[0] / space.dimensions[0]) * space.extent_m[0],
    space.origin_m[1] + (point[1] / space.dimensions[1]) * space.extent_m[1],
    space.origin_m[2] + (point[2] / space.dimensions[2]) * space.extent_m[2],
  ];
}

/** Smallest world span of one lattice cell; the unit dash lengths are set in. */
export function decorationCellSize_m(space: DecorationSpace): number {
  return Math.min(
    space.extent_m[0] / space.dimensions[0],
    space.extent_m[1] / space.dimensions[1],
    space.extent_m[2] / space.dimensions[2],
  );
}

export interface DecorationStyle {
  /** Linear RGB; the overlay composites additively. Use `linearFromHex`. */
  readonly colorLinear: DecorationVec3;
  readonly width_px: number;
  /** 0..1 alpha before depth ghosting. */
  readonly intensity?: number;
  /** Dash period in metres. Present means derived rather than measured. */
  readonly dash_m?: number;
  /** Screen-space arrowhead at the segment's end. */
  readonly arrow?: boolean;
  /**
   * 0..1 position in a reveal sweep. Only a decorator whose work really is a
   * sequence should use it; anything else leaves it at zero, because a non-zero
   * order brightens a segment as though it were the leading edge of one.
   */
  readonly order?: number;
}

/** sRGB hex to the linear triple the overlay shader consumes. */
export function linearFromHex(hex: string): DecorationVec3 {
  const value = Number.parseInt(hex.slice(1), 16);
  return [(value >> 16) & 0xff, (value >> 8) & 0xff, value & 0xff].map((channel) => {
    const normalized = channel / 255;
    return normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
  }) as unknown as DecorationVec3;
}

/** Arrowhead size in device pixels, expanded in screen space by the shader. */
const ARROW_HEAD_SCALE = 3.4;
const ARROW_HEAD_MINIMUM_PX = 4.5;
const ARROW_HEAD_MAXIMUM_PX = 13;

const BOX_EDGES: readonly (readonly [number, number])[] = Object.freeze([
  [0, 1], [1, 3], [3, 2], [2, 0],
  [4, 5], [5, 7], [7, 6], [6, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
]);

function boxCorner(minimum: DecorationVec3, maximum: DecorationVec3, index: number): DecorationVec3 {
  return [
    (index & 1) === 0 ? minimum[0] : maximum[0],
    (index & 2) === 0 ? minimum[1] : maximum[1],
    (index & 4) === 0 ? minimum[2] : maximum[2],
  ];
}

function orthonormalBasis(axis: DecorationVec3): readonly [DecorationVec3, DecorationVec3] {
  const length = Math.hypot(...axis) || 1;
  const forward: DecorationVec3 = [axis[0] / length, axis[1] / length, axis[2] / length];
  const helper: DecorationVec3 = Math.abs(forward[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
  const cross = (a: DecorationVec3, b: DecorationVec3): DecorationVec3 => [
    a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0],
  ];
  const tangent = cross(helper, forward);
  const tangentLength = Math.hypot(...tangent) || 1;
  const unit: DecorationVec3 = [
    tangent[0] / tangentLength, tangent[1] / tangentLength, tangent[2] / tangentLength,
  ];
  return [unit, cross(forward, unit)];
}

/**
 * Accumulates one frame's decorations into a single instance buffer.
 *
 * Every decorator writes into the same builder, so the assembled overlay is one
 * upload and one draw regardless of how many passes contributed. A builder is
 * bound to one space; a pass that reasons in a different lattice calls
 * `inSpace()` for a view of the same buffer under its own transform.
 */
export class DecorationBuilder {
  private readonly values: number[];
  private readonly countAt: () => number;

  constructor(
    public readonly space: DecorationSpace,
    values?: number[],
  ) {
    this.values = values ?? [];
    this.countAt = () => this.values.length / DECORATION_SEGMENT_FLOATS;
  }

  /** A view of this same buffer under another lattice. */
  inSpace(space: DecorationSpace): DecorationBuilder {
    return new DecorationBuilder(space, this.values);
  }

  get segmentCount(): number { return this.countAt(); }

  /** World span of one lattice cell in the builder's own space. */
  get cellSize_m(): number { return decorationCellSize_m(this.space); }

  toWorld(point: DecorationVec3): DecorationVec3 {
    return decorationToWorld(this.space, point);
  }

  /** One straight segment between two lattice points. */
  segment(start: DecorationVec3, end: DecorationVec3, style: DecorationStyle): void {
    this.worldSegment(this.toWorld(start), this.toWorld(end), style);
  }

  /** The same, for a decorator that already holds world metres. */
  worldSegment(start: DecorationVec3, end: DecorationVec3, style: DecorationStyle): void {
    const width_px = Math.max(0.35, style.width_px);
    const intensity = Math.max(0, Math.min(1, style.intensity ?? 1));
    const order = Math.max(0, Math.min(1, style.order ?? 0));
    const emit = (head_px: number) => {
      this.values.push(
        start[0], start[1], start[2], width_px,
        end[0], end[1], end[2], order,
        style.colorLinear[0], style.colorLinear[1], style.colorLinear[2], intensity,
        0, style.dash_m ?? 0, head_px, 0,
      );
    };
    emit(0);
    // The head is its own instance built from the same endpoints: the shaft keeps
    // its dash and thickness, and the shader turns the head into a solid
    // screen-space triangle at `end` so it never folds away edge-on.
    if (style.arrow) {
      emit(Math.max(ARROW_HEAD_MINIMUM_PX,
        Math.min(ARROW_HEAD_MAXIMUM_PX, width_px * ARROW_HEAD_SCALE)));
    }
  }

  /** Twelve edges of a lattice-aligned box; `high` is the exclusive corner. */
  box(low: DecorationVec3, high: DecorationVec3, style: DecorationStyle): void {
    const a = this.toWorld(low), b = this.toWorld(high);
    for (const [from, to] of BOX_EDGES) {
      this.worldSegment(boxCorner(a, b, from), boxCorner(a, b, to), style);
    }
  }

  /** The box of one lattice cell at `cell`, sized `size` cells on a side. */
  cellBox(cell: DecorationVec3, size: number, style: DecorationStyle): void {
    this.box(cell, [cell[0] + size, cell[1] + size, cell[2] + size], style);
  }

  /** A closed ring of `facets` segments, normal to `axis`, in lattice units. */
  ring(centre: DecorationVec3, axis: DecorationVec3, radius: number, style: DecorationStyle, facets = 12): void {
    const count = Math.max(3, Math.min(48, Math.round(facets)));
    const [tangent, bitangent] = orthonormalBasis(axis);
    const at = (index: number): DecorationVec3 => {
      const angle = (index / count) * Math.PI * 2;
      return [
        centre[0] + (tangent[0] * Math.cos(angle) + bitangent[0] * Math.sin(angle)) * radius,
        centre[1] + (tangent[1] * Math.cos(angle) + bitangent[1] * Math.sin(angle)) * radius,
        centre[2] + (tangent[2] * Math.cos(angle) + bitangent[2] * Math.sin(angle)) * radius,
      ];
    };
    for (let index = 0; index < count; index += 1) this.segment(at(index), at(index + 1), style);
  }

  /** Three axis-aligned bars through a point; marks a location without a box. */
  cross(centre: DecorationVec3, radius: number, style: DecorationStyle): void {
    for (let axis = 0; axis < 3; axis += 1) {
      const low = [...centre] as [number, number, number];
      const high = [...centre] as [number, number, number];
      low[axis] -= radius;
      high[axis] += radius;
      this.segment(low, high, style);
    }
  }

  /**
   * Absorb instances a decorator built itself.
   *
   * This is what lets an existing producer join the framework by wrapping rather
   * than by being rewritten: any buffer already in the instance layout — the SVO
   * pixel trace's, for one — concatenates straight in and draws in the same call
   * as everything else.
   */
  append(segments: Float32Array, segmentCount: number): void {
    const floats = Math.max(0, Math.min(segmentCount, Math.floor(segments.length / DECORATION_SEGMENT_FLOATS)))
      * DECORATION_SEGMENT_FLOATS;
    for (let index = 0; index < floats; index += 1) this.values.push(segments[index]);
  }

  /** Freeze what has accumulated into the buffer the overlay pipeline uploads. */
  finish(): { segments: Float32Array<ArrayBuffer>; segmentCount: number } {
    return { segments: Float32Array.from(this.values), segmentCount: this.countAt() };
  }
}
