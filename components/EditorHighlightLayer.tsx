"use client";

import { memo, useMemo } from "react";
import {
  BOX_EDGES, boxCorners, frameToWorld,
  type BoxExtent, type EditorEntityTone, type EditorFrame,
} from "../lib/core/editor-entity";
import type { EditorHighlight, EditorTarget } from "../lib/core/editor-target";
import type { CameraState, Vec3 } from "../lib/core/model";
import { viewportProjector } from "../lib/core/webgpu-camera";

interface EditorHighlightLayerProps {
  /** What is under the cursor, or undefined in LOOK and while a gesture runs. */
  target: EditorTarget | undefined;
  /** An idle tool ghost may share the view with faint object bounds; active drags may not. */
  showHoverWithHeld?: boolean;
  camera: CameraState;
  width: number;
  height: number;
  /** A gesture's own live highlight, drawn over the hover one. */
  held?: {
    readonly highlight: EditorHighlight;
    readonly caption?: string;
    /** A gesture may be about something other than what it started on. */
    readonly tone?: EditorEntityTone;
  };
  /** A selection that stands while a gesture runs over it, drawn quietly under the gesture's own. */
  standing?: { readonly highlight: EditorHighlight };
}

/**
 * The one place a highlight becomes lines on the screen.
 *
 * Every probe says what to light up as an `EditorHighlight` and none of them
 * says how, so this file is the whole drawing vocabulary: box, boxes, quad,
 * point and paths. That split is what makes a new probe free to add — it inherits the
 * outline, the clipping rule, the tone and the caption — and it is what stops
 * the viewport accumulating a fifth hand-rolled projected-box overlay next to
 * the four it already had, each with its own idea of when an edge is behind the
 * camera.
 *
 * `instance-range` remains renderer-owned. Selectable objects can additionally
 * declare hover bounds, including instanced trees, so the pointer can preview
 * the same box that selection will expose.
 *
 * Clipping is per *edge*, not per shape. A box the camera is standing inside has
 * corners behind it; dropping the whole box there makes a highlight blink out
 * the moment you lean into it, and projecting the negative-depth corner anyway
 * throws a line across the screen. Dropping the edge is the only reading that
 * degrades gracefully.
 *
 * Drawn by what it shows, not by how often it is asked. The viewport renders on
 * every pointer sample, and a gesture hands over a freshly built highlight each
 * time whether or not anything about it moved; a hundred-box stroke was eleven
 * hundred line elements rebuilt per sample to say the same thing. So the layer
 * compares what it is given by *shape* and stands down when nothing drawn has
 * changed, and inside a `boxes` highlight each box is its own memoized piece: a
 * stroke that grew by a cell draws that cell and compares the rest.
 */
export const EditorHighlightLayer = memo(function EditorHighlightLayer(
  { target, camera, width, height, held, standing, showHoverWithHeld = false }: EditorHighlightLayerProps,
) {
  // One basis per view, shared by every corner of every layer.
  const project = useMemo(() => viewportProjector(camera, width, height), [camera, width, height]);
  const drawn: HighlightLayer[] = [];
  // Idle previews can sit alongside quiet object bounds. Once the pointer is
  // held, the gesture owns the overlay and stale object outlines disappear.
  if (target && (!held || (showHoverWithHeld && target.hoverHighlight))) drawn.push({
    highlight: target.hoverHighlight ?? target.highlight, tone: target.tone,
    live: false, hover: Boolean(target.hoverHighlight),
  });
  if (standing) drawn.push({ highlight: standing.highlight, tone: "region", live: false, hover: false });
  if (held) drawn.push({
    highlight: held.highlight, tone: held.tone ?? target?.tone ?? "prop",
    caption: held.caption, live: true, hover: false,
  });
  const layers = drawn.filter((layer) => layer.highlight.kind !== "instance-range");
  if (layers.length === 0 || width <= 0 || height <= 0) return null;
  return <svg
    className="editor-highlight-layer"
    data-testid="editor-highlight-layer"
    width={width}
    height={height}
    aria-hidden="true"
  >
    {layers.map((layer, index) => <g
      key={index}
      className="editor-highlight"
      data-tone={layer.tone}
      data-live={layer.live || undefined}
      data-hover={layer.hover || undefined}
      data-highlight-kind={layer.highlight.kind}
    >
      <HighlightShape highlight={layer.highlight} project={project} width={width} height={height}
        caption={layer.caption} />
    </g>)}
  </svg>;
}, sameLayerProps);

interface HighlightLayer {
  readonly highlight: EditorHighlight;
  readonly tone: EditorEntityTone;
  readonly caption?: string;
  readonly live: boolean;
  readonly hover: boolean;
}

type Project = (point_m: Vec3) => { leftFraction: number; topFraction: number; depth_m: number };

const sameVec3 = (a: Vec3, b: Vec3): boolean => a === b || (a.x === b.x && a.y === b.y && a.z === b.z);

const sameBox = (a: BoxExtent, b: BoxExtent): boolean =>
  a === b || (sameVec3(a.min, b.min) && sameVec3(a.max, b.max));

const sameFrame = (a: EditorFrame | undefined, b: EditorFrame | undefined): boolean =>
  a === b || (a !== undefined && b !== undefined && sameVec3(a.origin_m, b.origin_m)
    && a.orientation.w === b.orientation.w && a.orientation.x === b.orientation.x
    && a.orientation.y === b.orientation.y && a.orientation.z === b.orientation.z);

/**
 * Whether two highlights draw the same lines.
 *
 * By shape rather than by identity, because the producers are not written to
 * keep one: a tool's `update` builds its highlight afresh for every sample.
 * The comparison is a number per coordinate; the drawing it spares is an
 * element per edge.
 */
export function sameHighlight(a: EditorHighlight | undefined, b: EditorHighlight | undefined): boolean {
  if (a === b) return true;
  if (!a || !b || a.kind !== b.kind) return false;
  switch (a.kind) {
    case "box": {
      const other = b as typeof a;
      return sameBox(a.box, other.box) && sameFrame(a.frame, other.frame);
    }
    case "boxes": {
      const other = b as typeof a;
      return a.boxes.length === other.boxes.length
        && a.boxes.every((box, index) => sameBox(box, other.boxes[index]!));
    }
    case "quad": {
      const other = b as typeof a;
      return a.corners.every((corner, index) => sameVec3(corner, other.corners[index]!));
    }
    case "point": {
      const other = b as typeof a;
      return a.radius_m === other.radius_m && sameVec3(a.position_m, other.position_m);
    }
    case "paths": {
      const other = b as typeof a;
      return a.paths.length === other.paths.length
        && (a.anchor_m === other.anchor_m
          || (a.anchor_m !== undefined && other.anchor_m !== undefined && sameVec3(a.anchor_m, other.anchor_m)))
        && a.paths.every((path, index) => {
          const against = other.paths[index]!;
          return path === against
            || (path.length === against.length && path.every((point, at) => sameVec3(point, against[at]!)));
        });
    }
    case "instance-range": {
      const other = b as typeof a;
      return a.first === other.first && a.last === other.last;
    }
  }
}

/** The layer reads a target's tone and the one highlight it draws for it, and nothing else. */
function sameDrawnTarget(a: EditorTarget | undefined, b: EditorTarget | undefined): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return a.tone === b.tone && Boolean(a.hoverHighlight) === Boolean(b.hoverHighlight)
    && sameHighlight(a.hoverHighlight ?? a.highlight, b.hoverHighlight ?? b.highlight);
}

function sameLayerProps(a: EditorHighlightLayerProps, b: EditorHighlightLayerProps): boolean {
  return a.camera === b.camera && a.width === b.width && a.height === b.height
    && Boolean(a.showHoverWithHeld) === Boolean(b.showHoverWithHeld)
    && sameDrawnTarget(a.target, b.target)
    && Boolean(a.held) === Boolean(b.held)
    && a.held?.caption === b.held?.caption && a.held?.tone === b.held?.tone
    && sameHighlight(a.held?.highlight, b.held?.highlight)
    && sameHighlight(a.standing?.highlight, b.standing?.highlight);
}

interface HighlightShapeProps {
  readonly highlight: EditorHighlight;
  readonly project: Project;
  readonly width: number;
  readonly height: number;
  readonly caption: string | undefined;
}

/** One layer's shape. A layer the sample left alone is not redrawn because its neighbour moved. */
const HighlightShape = memo(function HighlightShape({ highlight, project, width, height, caption }: HighlightShapeProps) {
  return <>{drawHighlight(highlight, project, width, height, caption)}</>;
}, (a, b) => a.project === b.project && a.width === b.width && a.height === b.height
  && a.caption === b.caption && sameHighlight(a.highlight, b.highlight));

function drawHighlight(
  highlight: EditorHighlight,
  project: Project,
  width: number,
  height: number,
  caption: string | undefined,
) {
  const px = (point: { leftFraction: number; topFraction: number }) =>
    ({ x: point.leftFraction * width, y: point.topFraction * height });
  switch (highlight.kind) {
    case "paths": {
      const lines = highlight.paths.flatMap((path, pathIndex) => {
        const points = path.map((point) => {
          const projected = project(point);
          return { ...px(projected), depth_m: projected.depth_m };
        });
        return points.slice(1).map((to, index) => {
          const from = points[index]!;
          if (!(from.depth_m > 1e-6 && to.depth_m > 1e-6)) return null;
          return <line key={`path-${pathIndex}-${index}`} className="highlight-edge"
            x1={from.x} y1={from.y} x2={to.x} y2={to.y} />;
        });
      });
      const anchor = highlight.anchor_m ? project(highlight.anchor_m) : undefined;
      return [...lines, ...captionAt(anchor && anchor.depth_m > 1e-6 ? px(anchor) : undefined, caption)];
    }
    case "box":
      return [
        ...boxLines(highlight.box, highlight.frame, project, px, "box"),
        ...captionAt(boxAnchor(highlight.box, highlight.frame, project, px), caption),
      ];
    // The first box leads and the rest are its members — a swept region and the
    // cells inside it. Drawn at the same weight, five hundred cell outlines bury
    // the box they are inside; the lead/member split is what keeps the shape the
    // gesture is actually making readable through them.
    case "boxes": {
      const anchor = highlight.boxes[0]
        ? boxAnchor(highlight.boxes[0], undefined, project, px) : undefined;
      return [
        ...highlight.boxes.map((box, index) => <HighlightBox key={`boxes-${index}`} box={box} project={project}
          width={width} height={height} name={`boxes-${index}`} member={index > 0} />),
        ...captionAt(anchor, caption),
      ];
    }
    case "quad": {
      const corners = highlight.corners.map((corner) => {
        const projected = project(corner);
        return { ...px(projected), depth_m: projected.depth_m };
      });
      const visible = corners.every((corner) => corner.depth_m > 1e-6);
      return [
        visible ? <polygon
          key="quad"
          className="highlight-face"
          points={corners.map((corner) => `${corner.x},${corner.y}`).join(" ")}
        /> : null,
        ...captionAt(visible ? corners[0] : undefined, caption),
      ];
    }
    case "point": {
      const centre = project(highlight.position_m);
      if (!(centre.depth_m > 1e-6)) return null;
      const edge = project({ ...highlight.position_m, x: highlight.position_m.x + highlight.radius_m });
      const origin = px(centre);
      const radius_px = Math.max(3, Math.hypot(px(edge).x - origin.x, px(edge).y - origin.y));
      return [
        <circle key="mark" className="highlight-mark" cx={origin.x} cy={origin.y} r={radius_px} />,
        <circle key="pip" className="highlight-pip" cx={origin.x} cy={origin.y} r={1.5} />,
        ...captionAt(origin, caption),
      ];
    }
    // `instance-range` is filtered out before this call; the case keeps the
    // switch exhaustive so a new highlight kind is a compile error here.
    case "instance-range":
      return null;
  }
}

interface HighlightBoxProps {
  readonly box: BoxExtent;
  readonly project: Project;
  readonly width: number;
  readonly height: number;
  readonly name: string;
  readonly member: boolean;
}

/**
 * One box of a `boxes` highlight, redrawn only when it or the view moves.
 *
 * A stroke's highlight grows by a box per cell and is otherwise the boxes it
 * was, so comparing each by its six numbers leaves a sample that added one cell
 * with one cell's twelve edges to draw.
 */
const HighlightBox = memo(function HighlightBox({ box, project, width, height, name, member }: HighlightBoxProps) {
  const px = (point: { leftFraction: number; topFraction: number }) =>
    ({ x: point.leftFraction * width, y: point.topFraction * height });
  return <>{boxLines(box, undefined, project, px, name, member ? "highlight-edge highlight-member" : "highlight-edge")}</>;
}, (a, b) => a.project === b.project && a.width === b.width && a.height === b.height
  && a.member === b.member && a.name === b.name && sameBox(a.box, b.box));

function boxLines(
  box: BoxExtent,
  frame: EditorFrame | undefined,
  project: Project,
  px: (point: { leftFraction: number; topFraction: number }) => { x: number; y: number },
  key: string,
  className = "highlight-edge",
) {
  const corners = boxCorners(box)
    .map((corner) => {
      const projected = project(frame ? frameToWorld(frame, corner) : corner);
      return { ...px(projected), depth_m: projected.depth_m };
    });
  return BOX_EDGES
    .filter(([from, to]) => corners[from]!.depth_m > 1e-6 && corners[to]!.depth_m > 1e-6)
    .map(([from, to]) => <line
      key={`${key}-${from}-${to}`}
      className={className}
      x1={corners[from]!.x} y1={corners[from]!.y}
      x2={corners[to]!.x} y2={corners[to]!.y}
    />);
}

/** The top-far corner, so a caption sits above the shape rather than inside it. */
function boxAnchor(
  box: BoxExtent,
  frame: EditorFrame | undefined,
  project: Project,
  px: (point: { leftFraction: number; topFraction: number }) => { x: number; y: number },
): { x: number; y: number } | undefined {
  const corner = { x: box.min.x, y: box.max.y, z: box.max.z };
  const projection = project(frame ? frameToWorld(frame, corner) : corner);
  return projection.depth_m > 1e-6 ? px(projection) : undefined;
}

function captionAt(anchor: { x: number; y: number } | undefined, caption: string | undefined) {
  if (!anchor || !caption) return [];
  return [<text key="caption" className="highlight-caption" x={anchor.x} y={anchor.y - 9}>{caption}</text>];
}
