import type { BoxExtent, EditorEntityTone, EditorFrame } from "./editor-entity";
import type { EditorSelection } from "./editor-tools";
import type { Vec3 } from "./model";

/**
 * What the cursor is over.
 *
 * The editor's second plugin boundary, one level *below* the entity. An
 * `EditorEntity` is a thing in the document that can be selected and dragged as
 * a whole — a body, the tank, a region. A target is finer: it is whatever the
 * ray actually met, which is usually a *part* of one of those, and sometimes
 * something the document does not name at all.
 *
 * The distinction earns its keep because a scene is not made of entities. It is
 * made of voxels, walls, cells and surfaces, and every interesting contextual
 * question — what is this cell's pressure, clear these voxels, what is this
 * wall's boundary condition — is asked of one of those, not of the object that
 * happens to contain it. Without this layer each such question has to be
 * smuggled in as an armed tool, which is exactly the design this replaces.
 *
 * Three properties are load-bearing:
 *
 * - **A target always exists.** `targetAtRay` returns one, not one-or-nothing.
 *   The room is a declared fallback probe, so a ray that meets no object still
 *   meets the room, and "there is nothing under the cursor" stops being a state
 *   the interface has to render. See `editor-probe-catalog.ts`.
 * - **`selection` is optional and separate from `id`.** `id` addresses the
 *   target — a voxel coordinate, a wall face — while `selection` names the
 *   document object a left-click should select. A fluid cell has no selection:
 *   it can be inspected and its ring is real, but there is nothing in the
 *   document to select.
 * - **`highlight` is declarative.** A probe says what shape to light up and
 *   never how. One layer draws every shape, so a new probe costs no drawing
 *   code and cannot invent a fifth way of outlining a box.
 */
export type EditorTargetKind =
  | "entity"
  | "solid-voxel"
  | "tank-wall"
  | "fluid-cell"
  | "terrain"
  | "room";

export interface EditorTarget {
  /** Which probe answered. Also the tie-break key when two agree on distance. */
  readonly probeId: string;
  readonly kind: EditorTargetKind;
  /** Addresses the target within its probe: "12,3,44", "+y", an entity id. */
  readonly id: string;
  /** Shown on the hover chip. */
  readonly label: string;
  /** Palette token, shared with entities so one vocabulary colours everything. */
  readonly tone: EditorEntityTone;
  readonly distance_m: number;
  readonly point_m: Vec3;
  readonly normal: Vec3;
  /**
   * What a left-click selects, when the target belongs to something the
   * document names. Absent for targets that are real but not selectable — the
   * room, and a fluid cell, which exist only in the solve.
   */
  readonly selection?: EditorSelection;
  readonly highlight: EditorHighlight;
  /**
   * Free-form detail the probe's own actions and gestures read back.
   *
   * Deliberately unstructured and deliberately never read by anything generic:
   * the face axis a voxel drag locks to, the cell's row index. A probe owns
   * both ends, so a shared shape here would only be a place for unrelated
   * probes to collide.
   */
  readonly detail?: Readonly<Record<string, number | string | boolean>>;
}

/**
 * What to light up, as a shape rather than as drawing.
 *
 * `instance-range` is the one member that goes to the GPU: it names a run of
 * scenery instances for the renderer's own rim pass, which already exists
 * (`setHoverHighlight`). Everything else is projected to the screen and stroked
 * by `EditorHighlightLayer`, which is what the four hand-rolled box overlays in
 * the viewport were each doing separately.
 */
export type EditorHighlight =
  | { readonly kind: "box"; readonly box: BoxExtent; readonly frame?: EditorFrame }
  | {
    readonly kind: "boxes";
    readonly boxes: readonly BoxExtent[];
    /** Set when more boxes exist than are drawn, so the caption can say so. */
    readonly truncated?: boolean;
  }
  | { readonly kind: "quad"; readonly corners: readonly [Vec3, Vec3, Vec3, Vec3] }
  | { readonly kind: "point"; readonly position_m: Vec3; readonly radius_m: number }
  | { readonly kind: "instance-range"; readonly first: number; readonly last: number };

/** The instance range a highlight names, or undefined when it is not a GPU one. */
export function highlightInstanceRange(
  highlight: EditorHighlight | undefined,
): { readonly first: number; readonly last: number } | undefined {
  return highlight?.kind === "instance-range" ? { first: highlight.first, last: highlight.last } : undefined;
}

/** True when two targets name the same thing, so a redraw can be skipped. */
export function sameTarget(a: EditorTarget | undefined, b: EditorTarget | undefined): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return a.probeId === b.probeId && a.kind === b.kind && a.id === b.id;
}

/** The box a target lights up, when it lights up exactly one. */
export function targetBox(target: EditorTarget): BoxExtent | undefined {
  return target.highlight.kind === "box" ? target.highlight.box : undefined;
}
