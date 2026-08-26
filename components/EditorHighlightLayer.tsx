"use client";

import {
  BOX_EDGES, boxCorners, frameToWorld,
  type BoxExtent, type EditorEntityTone, type EditorFrame,
} from "../lib/core/editor-entity";
import type { EditorHighlight, EditorTarget } from "../lib/core/editor-target";
import type { CameraState, Vec3 } from "../lib/core/model";
import { projectToViewport } from "../lib/core/webgpu-camera";

/**
 * The one place a highlight becomes lines on the screen.
 *
 * Every probe says what to light up as an `EditorHighlight` and none of them
 * says how, so this file is the whole drawing vocabulary: box, boxes, quad,
 * point. That split is what makes a new probe free to add — it inherits the
 * outline, the clipping rule, the tone and the caption — and it is what stops
 * the viewport accumulating a fifth hand-rolled projected-box overlay next to
 * the four it already had, each with its own idea of when an edge is behind the
 * camera.
 *
 * `instance-range` is absent by design: that highlight is drawn by the renderer
 * as a rim on the actual instanced geometry, which is the only way to outline a
 * tree without drawing a crate around it. It never reaches this layer.
 *
 * Clipping is per *edge*, not per shape. A box the camera is standing inside has
 * corners behind it; dropping the whole box there makes a highlight blink out
 * the moment you lean into it, and projecting the negative-depth corner anyway
 * throws a line across the screen. Dropping the edge is the only reading that
 * degrades gracefully.
 */
export function EditorHighlightLayer({ target, camera, width, height, held }: {
  /** What is under the cursor, or undefined in LOOK and while a gesture runs. */
  target: EditorTarget | undefined;
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
}) {
  const drawn: HighlightLayer[] = [];
  // A gesture's own shape replaces the hover one rather than joining it: while a
  // drag is running the hover target is whatever the press started on, and
  // drawing both would leave a stale outline pinned under a live one.
  if (held) drawn.push({
    highlight: held.highlight, tone: held.tone ?? target?.tone ?? "prop",
    caption: held.caption, live: true,
  });
  else if (target) drawn.push({ highlight: target.highlight, tone: target.tone, live: false });
  const layers = drawn.filter((layer) => layer.highlight.kind !== "instance-range");
  if (layers.length === 0 || width <= 0 || height <= 0) return null;
  const project = (point_m: Vec3) => projectToViewport(point_m, camera, width, height);
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
      data-highlight-kind={layer.highlight.kind}
    >
      {drawHighlight(layer.highlight, project, width, height, layer.caption)}
    </g>)}
  </svg>;
}

interface HighlightLayer {
  readonly highlight: EditorHighlight;
  readonly tone: EditorEntityTone;
  readonly caption?: string;
  readonly live: boolean;
}

type Project = (point_m: Vec3) => { leftFraction: number; topFraction: number; depth_m: number };

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
        ...highlight.boxes.flatMap((box, index) => boxLines(box, undefined, project, px,
          `boxes-${index}`, index === 0 ? "highlight-edge" : "highlight-edge highlight-member")),
        ...captionAt(anchor, caption),
      ];
    }
    case "quad": {
      const corners = highlight.corners.map((corner) => ({ ...px(project(corner)), depth_m: project(corner).depth_m }));
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

function boxLines(
  box: BoxExtent,
  frame: EditorFrame | undefined,
  project: Project,
  px: (point: { leftFraction: number; topFraction: number }) => { x: number; y: number },
  key: string,
  className = "highlight-edge",
) {
  const corners = boxCorners(box)
    .map((corner) => (frame ? frameToWorld(frame, corner) : corner))
    .map((corner) => ({ ...px(project(corner)), depth_m: project(corner).depth_m }));
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
