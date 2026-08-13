"use client";

import { useId } from "react";

import type { SceneDescription } from "@/lib/model";
import {
  sceneIsoGlyph,
  sceneIsoGlyphLabel,
  type IsoBox,
  type IsoVec,
  type SceneIsoGlyph as SceneIsoGlyphModel,
} from "@/lib/scene-glyph-iso";

/**
 * The card's thumbnail: `sceneIsoGlyph` drawn as a small room, in inline SVG.
 *
 * Inline rather than an image for the same reason the elevation was — the mark
 * is already a pure function of the document, and a fetch or a canvas would only
 * add a way for the picture to disagree with the card it sits on. Every colour
 * lives in the stylesheet, so one rule restyles every thumbnail in the product.
 *
 * Nothing here decides what a scene contains; it decides where the corners land.
 */

const VIEW_WIDTH = 100;
const VIEW_HEIGHT = 72;
/** Room for the near-bottom corner's stroke and for a body sitting proud of the rim. */
const MARGIN = 6;

const ISO_COS = Math.cos(Math.PI / 6);
const ISO_SIN = Math.sin(Math.PI / 6);
/**
 * The projection is orthogonal with equal row norms, so it is √1.5 times an
 * orthonormal one — which is exactly the radius a unit sphere comes out at.
 */
const ISO_SPHERE = Math.sqrt(1.5);

/** Enough to read as a hose at 90 px without crossing the room it enters. */
const INFLOW_LENGTH = 0.28;

interface Projected { readonly x: number; readonly y: number }

/** The screen-space box a room of these proportions occupies, before fitting. */
function projectedBounds(extent: IsoVec) {
  return {
    x0: -extent.z * ISO_COS, x1: extent.x * ISO_COS,
    y0: -extent.y, y1: (extent.x + extent.z) * ISO_SIN,
  };
}

export interface SceneIsoGlyphProps {
  readonly scene: SceneDescription;
  /** Precomputed mark, when the caller already cached one for this document. */
  readonly glyph?: SceneIsoGlyphModel;
  /**
   * Draw at a shared scale instead of filling the frame.
   *
   * A card shows one scene, so it fits. A row that offers three room sizes has
   * to draw them against each other or the choice it presents is a lie: pass
   * `size_m / (largest size_m in the row)` and every room in the row lands on
   * one floor at its true size.
   */
  readonly scale?: number;
  readonly className?: string;
}

export function SceneIsoGlyph({ scene, glyph = sceneIsoGlyph(scene), scale, className }: SceneIsoGlyphProps) {
  const sphereClipId = useId().replace(/:/g, "");
  const extent = glyph.extent;
  const shared = scale !== undefined;
  // A shared scale measures every room against the same cube, so a hall and a
  // cube are not each independently blown up to fill their own frame.
  const frame = projectedBounds(shared ? { x: 1, y: 1, z: 1 } : extent);
  const own = projectedBounds(extent);
  const fit = Math.min(
    (VIEW_WIDTH - 2 * MARGIN) / (frame.x1 - frame.x0),
    (VIEW_HEIGHT - 2 * MARGIN) / (frame.y1 - frame.y0),
  ) * (scale ?? 1);
  const originX = 0.5 * VIEW_WIDTH - fit * 0.5 * (own.x0 + own.x1);
  // Shared rooms stand on one floor; a lone room is centred in its own frame.
  const originY = shared
    ? VIEW_HEIGHT - MARGIN - fit * own.y1
    : 0.5 * VIEW_HEIGHT - fit * 0.5 * (own.y0 + own.y1);

  const put = (point: IsoVec): Projected => ({
    x: originX + fit * (point.x - point.z) * ISO_COS,
    y: originY + fit * ((point.x + point.z) * ISO_SIN - point.y),
  });
  const face = (points: readonly IsoVec[]) =>
    `M${points.map(put).map(({ x, y }) => `${x.toFixed(2)},${y.toFixed(2)}`).join("L")}Z`;
  const edge = (from: IsoVec, to: IsoVec) => {
    const a = put(from), b = put(to);
    return `M${a.x.toFixed(2)},${a.y.toFixed(2)}L${b.x.toFixed(2)},${b.y.toFixed(2)}`;
  };

  // The three faces a corner view sees of any box: the top, and the two that
  // face the viewer along +x and +z.
  const litFaces = ({ min, max }: IsoBox) => ({
    top: [{ x: min.x, y: max.y, z: min.z }, { x: max.x, y: max.y, z: min.z }, { x: max.x, y: max.y, z: max.z }, { x: min.x, y: max.y, z: max.z }],
    right: [{ x: max.x, y: min.y, z: min.z }, { x: max.x, y: max.y, z: min.z }, { x: max.x, y: max.y, z: max.z }, { x: max.x, y: min.y, z: max.z }],
    front: [{ x: min.x, y: min.y, z: max.z }, { x: min.x, y: max.y, z: max.z }, { x: max.x, y: max.y, z: max.z }, { x: max.x, y: min.y, z: max.z }],
  });

  const corner = (x: number, y: number, z: number): IsoVec => ({ x, y, z });
  const { x: ex, y: ey, z: ez } = extent;
  const room = {
    floor: [corner(0, 0, 0), corner(ex, 0, 0), corner(ex, 0, ez), corner(0, 0, ez)],
    backLeft: [corner(0, 0, 0), corner(0, ey, 0), corner(0, ey, ez), corner(0, 0, ez)],
    backRight: [corner(0, 0, 0), corner(0, ey, 0), corner(ex, ey, 0), corner(ex, 0, 0)],
    lid: [corner(0, ey, 0), corner(ex, ey, 0), corner(ex, ey, ez), corner(0, ey, ez)],
  };
  // The three edges meeting the nearest corner carry the depth cue, so they are
  // the ones drawn a shade heavier. Everything else is one hairline.
  const nearEdges = [
    edge(corner(ex, 0, 0), corner(ex, 0, ez)),
    edge(corner(0, 0, ez), corner(ex, 0, ez)),
    edge(corner(ex, 0, ez), corner(ex, ey, ez)),
  ];
  const farEdges = [
    edge(corner(0, 0, 0), corner(ex, 0, 0)), edge(corner(0, 0, 0), corner(0, 0, ez)),
    edge(corner(0, ey, 0), corner(ex, ey, 0)), edge(corner(0, ey, 0), corner(0, ey, ez)),
    edge(corner(ex, ey, 0), corner(ex, ey, ez)), edge(corner(0, ey, ez), corner(ex, ey, ez)),
    edge(corner(0, 0, 0), corner(0, ey, 0)), edge(corner(ex, 0, 0), corner(ex, ey, 0)),
    edge(corner(0, 0, ez), corner(0, ey, ez)),
  ];

  // Painter's order, and the one place the drawing is opinionated: water goes
  // down before the ground, so a bank in front of the pond hides the water
  // behind it rather than the other way round.
  const depth = (volume: IsoBox) => volume.min.x + volume.min.y + volume.min.z;
  const water = [...(glyph.water ?? [])].sort((left, right) => depth(left) - depth(right));
  const bodies = [...glyph.bodies].sort((left, right) => depth(left) - depth(right));
  const sphere = glyph.tank.shape === "sphere" ? (() => {
    const centre = put({ x: 0.5 * ex, y: 0.5 * ey, z: 0.5 * ez });
    return { centre, radius: ISO_SPHERE * fit * Math.min(ex, ey, ez) * 0.5 };
  })() : undefined;
  const vesselClip = sphere ? `url(#${sphereClipId})` : undefined;

  return (
    <svg
      className={className ? `scene-iso ${className}` : "scene-iso"}
      viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={sceneIsoGlyphLabel(glyph)}
    >
      {sphere && <defs>
        <clipPath id={sphereClipId}>
          <circle cx={sphere.centre.x} cy={sphere.centre.y} r={sphere.radius} />
        </clipPath>
      </defs>}
      {sphere
        ? <circle className="iso-sphere-vessel-fill" cx={sphere.centre.x} cy={sphere.centre.y} r={sphere.radius} />
        : <>
          <path className="iso-room-floor" d={face(room.floor)} />
          <path className="iso-room-wall" d={face(room.backLeft)} />
          <path className="iso-room-wall" d={face(room.backRight)} />
        </>}

      {water.map((volume, index) => {
        const faces = litFaces(volume);
        return <g className="iso-water" key={`water-${index}`} clipPath={vesselClip}>
          <path className="iso-water-side" d={face(faces.right)} />
          <path className="iso-water-side" d={face(faces.front)} />
          <path className="iso-water-top" d={face(faces.top)} />
        </g>;
      })}

      {/* Ground rows, back to front. Each is translucent, so the overlap does
          the depth shading without a per-row colour anywhere in this file. */}
      {glyph.terrain?.rows.map((row, index) => (
        <path
          className="iso-terrain-row"
          key={`terrain-${index}`}
          d={face([
            corner(0, 0, row.z),
            ...row.heights.map((height, sample) =>
              corner((sample / (row.heights.length - 1)) * ex, height, row.z)),
            corner(ex, 0, row.z),
          ])}
        />
      ))}

      {bodies.map((body, index) => {
        if (body.round) {
          const centre = put({
            x: 0.5 * (body.min.x + body.max.x),
            y: 0.5 * (body.min.y + body.max.y),
            z: 0.5 * (body.min.z + body.max.z),
          });
          const radius = ISO_SPHERE * fit * (body.max.x - body.min.x + body.max.y - body.min.y + body.max.z - body.min.z) / 6;
          return <circle className="iso-body-round" key={`body-${index}`} cx={centre.x} cy={centre.y} r={Math.max(0.6, radius)} />;
        }
        const faces = litFaces(body);
        return <g className="iso-body" key={`body-${index}`}>
          <path className="iso-body-side" d={face(faces.right)} />
          <path className="iso-body-side" d={face(faces.front)} />
          <path className="iso-body-top" d={face(faces.top)} />
        </g>;
      })}

      {glyph.inflow && (() => {
        const { origin, direction } = glyph.inflow;
        const nozzle = put(origin);
        const jet = put({
          x: origin.x + direction.x * INFLOW_LENGTH,
          y: origin.y + direction.y * INFLOW_LENGTH,
          z: origin.z + direction.z * INFLOW_LENGTH,
        });
        return <g className="iso-inflow">
          <line x1={nozzle.x} y1={nozzle.y} x2={jet.x} y2={jet.y} />
          <circle cx={nozzle.x} cy={nozzle.y} r={1.6} />
        </g>;
      })()}

      {/* The room last, so nothing drawn inside it spills over its walls. A lid
          is a face you look through; an open room simply has none. */}
      {sphere
        ? <circle className="iso-sphere-vessel-edge" cx={sphere.centre.x} cy={sphere.centre.y} r={sphere.radius} />
        : <>
          {glyph.tank.top === "closed" && <path className="iso-room-lid" d={face(room.lid)} />}
          <path className="iso-room-edge" d={farEdges.join("")} />
          <path className="iso-room-edge iso-room-edge-near" d={nearEdges.join("")} />
        </>}
    </svg>
  );
}
