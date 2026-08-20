"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { actionIsChoosable, type EditorAction } from "../lib/core/editor-action";
import { EditorActionIconMark } from "./EditorActionIcon";
import { performEditorAction } from "../lib/core/editor-action-runtime";
import { useUIStore } from "../lib/core/stores/ui-store";

/**
 * The contextual ring.
 *
 * A pie rather than a list for the reason every editor that has one gives: the
 * wedges are in fixed directions from where the pointer already is, so choosing
 * becomes a flick in a remembered direction instead of a read. That only works
 * if the ring's shape is stable, which is why an action that does not apply
 * right now is drawn disabled rather than omitted — see `EditorAction.enabled`.
 *
 * What is *in* the ring is not this component's business. The viewport resolves
 * the pointer to an entity and asks the catalog what that entity offers; this
 * draws whatever comes back and hands the chosen effect to the one performer.
 * Adding a verb to an object never touches this file.
 *
 * Keyboard-reachable throughout: the ring takes focus when it opens, the arrow
 * keys walk it, Enter chooses, and Escape backs out one level and then closes.
 * A menu that only a mouse can reach is a menu that disappears for anyone
 * driving from the keyboard, and this one now owns modes that had buttons.
 */

const OUTER_RADIUS_PX = 124;
const INNER_RADIUS_PX = 46;
const LABEL_RADIUS_PX = 88;
/** Icon and caption straddle the label radius, so both stay clear of both edges. */
const ICON_OFFSET_PX = 13;
const LABEL_OFFSET_PX = 14;
/** Half the gap drawn between neighbouring wedges, in radians at the outer edge. */
const WEDGE_GAP_RAD = 0.022;
/** The SVG's own box: the ring plus the stroke room its viewBox reserves. */
const RING_BOX_PX = 2 * OUTER_RADIUS_PX + 8;
/** Vertical room the hint readout takes under the ring, so clamping can keep it on screen. */
const READOUT_RESERVE_PX = 58;
const RING_MARGIN_PX = 8;
/** The gap the readout hangs off the ring by — the CSS `top`/`bottom` offset. */
const READOUT_GAP_PX = 11;
/** Two lines of readout plus its padding: what the flip below has to clear. */
const READOUT_HEIGHT_PX = 64;
/** The band along the bottom edge the transport cluster floats in. */
const TRANSPORT_KEEPOUT_PX = 90;
/** From this many wedges the caption steps down a size — the arc has halved. */
const DENSE_WEDGE_COUNT = 8;
/** How close a caption may come to its wedge's own edges. */
const LABEL_EDGE_MARGIN_PX = 3;
/** A caption squeezed below this stops being a word. No shipped ring comes near it. */
const LABEL_MINIMUM_PX = 24;

/**
 * Keep the whole ring on screen.
 *
 * A pie opened under the pointer is the one menu shape that cannot simply flip
 * to the other side of the cursor when it runs out of room — its whole premise
 * is that every wedge is a fixed direction away. So it slides instead: the ring
 * moves just far enough to fit, and the wedge directions survive even though
 * the centre is no longer exactly where the click was.
 */
function clampToViewport(x: number, y: number): { x: number; y: number } {
  const width = typeof window === "undefined" ? Infinity : window.innerWidth;
  const height = typeof window === "undefined" ? Infinity : window.innerHeight;
  const radius = RING_BOX_PX / 2 + RING_MARGIN_PX;
  return {
    x: Math.min(Math.max(x, radius), Math.max(radius, width - radius)),
    y: Math.min(Math.max(y, radius), Math.max(radius, height - radius - READOUT_RESERVE_PX)),
  };
}

function polar(radius: number, angle_rad: number): [number, number] {
  return [radius * Math.cos(angle_rad), radius * Math.sin(angle_rad)];
}

/**
 * How far left and right of (x, y) a horizontal caption may run before it
 * crosses one of its own wedge's edges.
 *
 * The obvious bound — the arc the wedge subtends at the caption's radius — is
 * wrong in both directions, and the eleven-wedge ring shows both errors at once:
 * it is far too tight for the wedge pointing left, where the caption runs *along*
 * the radius and has the whole 78 px depth of the ring to use, and too loose for
 * the wedge at the top, whose caption sits inboard of the label radius where the
 * wedge is narrower than the arc says. So the four edges a wedge actually has —
 * two radial lines and two arcs — are solved directly.
 *
 * The two ends are returned separately rather than as one symmetric width
 * because a wedge is not symmetric about the point its caption is anchored at:
 * "Inspect cell" has 50 px of room on the ring's left flank and only 39 of it
 * centred, and it is a whole word either way.
 */
function labelWedgeRoom(x: number, y: number, from_rad: number, to_rad: number): { low: number; high: number } {
  const radius = Math.hypot(x, y);
  if (radius <= 0) return { low: 0, high: 0 };
  // The margin is applied to the wedge rather than to the caption, so a wedge
  // that is tight everywhere gives ground on every side at once.
  const inset_rad = Math.min(LABEL_EDGE_MARGIN_PX / radius, (to_rad - from_rad) / 2);
  const from = from_rad + inset_rad;
  const to = to_rad - inset_rad;
  const outer = OUTER_RADIUS_PX - LABEL_EDGE_MARGIN_PX;
  const inner = INNER_RADIUS_PX + LABEL_EDGE_MARGIN_PX;
  let low = -Infinity;
  let high = Infinity;
  // Every constraint reads `offset + slope * w >= 0` for a caption end at x + w.
  const keep = (offset: number, slope: number) => {
    if (Math.abs(slope) < 1e-9) return;
    if (slope > 0) low = Math.max(low, -offset / slope);
    else high = Math.min(high, -offset / slope);
  };
  // A wedge wider than a half-turn is not the intersection of two half-planes,
  // and a ring of two has room to spare anyway: only the arcs bound it.
  if (to - from < Math.PI) {
    keep(Math.cos(from) * y - Math.sin(from) * x, -Math.sin(from));
    keep(x * Math.sin(to) - y * Math.cos(to), Math.sin(to));
  }
  const reach = Math.sqrt(Math.max(0, outer * outer - y * y));
  low = Math.max(low, -reach - x);
  high = Math.min(high, reach - x);
  // The hub: a caption may not cross it, so it stays on the side it started.
  if (Math.abs(y) < inner) {
    const clearance = Math.sqrt(inner * inner - y * y);
    if (x >= 0) low = Math.max(low, clearance - x);
    else high = Math.min(high, -clearance - x);
  }
  return { low: Math.min(low, 0), high: Math.max(high, 0) };
}

/** One wedge as an SVG path, in a coordinate system centred on the ring. */
function wedgePath(from_rad: number, to_rad: number): string {
  const [outerFromX, outerFromY] = polar(OUTER_RADIUS_PX, from_rad);
  const [outerToX, outerToY] = polar(OUTER_RADIUS_PX, to_rad);
  const [innerToX, innerToY] = polar(INNER_RADIUS_PX, to_rad);
  const [innerFromX, innerFromY] = polar(INNER_RADIUS_PX, from_rad);
  const sweep = to_rad - from_rad > Math.PI ? 1 : 0;
  return [
    `M ${outerFromX} ${outerFromY}`,
    `A ${OUTER_RADIUS_PX} ${OUTER_RADIUS_PX} 0 ${sweep} 1 ${outerToX} ${outerToY}`,
    `L ${innerToX} ${innerToY}`,
    `A ${INNER_RADIUS_PX} ${INNER_RADIUS_PX} 0 ${sweep} 0 ${innerFromX} ${innerFromY}`,
    "Z",
  ].join(" ");
}

export function RadialMenu() {
  const menu = useUIStore((state) => state.radialMenu);
  const closeRadialMenu = useUIStore((state) => state.closeRadialMenu);
  // The ring can now leave the studio — Library is a wedge. The performer is
  // framework-free, so the one surface that has a router hands it over.
  const router = useRouter();
  // The ring currently drawn: the root, or a wedge's children after opening it.
  const [path, setPath] = useState<readonly number[]>([]);
  const [focused, setFocused] = useState(0);
  const ringRef = useRef<SVGSVGElement>(null);

  // A fresh menu always opens at its root, whatever the last one was left on.
  // Closing does not unmount this — it renders null and keeps its state — so
  // the reset is compared against the menu that was drawn rather than run from
  // an effect, which would render the stale ring once before correcting it.
  const [drawnMenu, setDrawnMenu] = useState(menu);
  if (menu !== drawnMenu) {
    setDrawnMenu(menu);
    setPath([]);
    setFocused(0);
  }
  useEffect(() => { ringRef.current?.focus(); }, [menu, path]);

  const actions = useMemo(() => {
    let level: readonly EditorAction[] = menu?.actions ?? [];
    for (const index of path) level = level[index]?.children ?? [];
    return level;
  }, [menu, path]);

  // Whatever the type step above could not save, the geometry finishes: a
  // caption is measured against the wedge it is actually in, slid along its own
  // line to whatever room that wedge has, and condensed only if it still does
  // not fit. Measured rather than guessed because the wedge count, the word and
  // the face all vary — a caption that crosses its own border is the one thing
  // that makes a drawn menu look unfinished, and every ring has to pass.
  useEffect(() => {
    const ring = ringRef.current;
    if (!ring) return;
    const labels = Array.from(ring.querySelectorAll<SVGTextElement>("text.radial-label"));
    const span = (2 * Math.PI) / labels.length;
    const start = -Math.PI / 2 - span / 2;
    for (const [index, label] of labels.entries()) {
      // Recomputed rather than read back, so a second pass over a caption this
      // already moved measures the same wedge it did the first time.
      const middle = start + (index + 0.5) * span;
      const [anchorX, anchorY] = polar(LABEL_RADIUS_PX, middle);
      const [x, y] = [anchorX, anchorY + LABEL_OFFSET_PX];
      label.setAttribute("x", `${x}`);
      label.removeAttribute("textLength");
      label.removeAttribute("lengthAdjust");
      const { low, high } = labelWedgeRoom(x, y,
        start + index * span + WEDGE_GAP_RAD,
        start + (index + 1) * span - WEDGE_GAP_RAD);
      const half = label.getComputedTextLength() / 2;
      if (high - low >= 2 * half) {
        // The least movement that brings both ends inside: a caption that
        // already fits does not budge.
        const shift = Math.min(Math.max(0, low + half), high - half);
        if (shift !== 0) label.setAttribute("x", `${x + shift}`);
        continue;
      }
      label.setAttribute("x", `${x + (low + high) / 2}`);
      label.setAttribute("textLength", `${Math.max(high - low, LABEL_MINIMUM_PX)}`);
      label.setAttribute("lengthAdjust", "spacingAndGlyphs");
    }
  }, [actions]);

  if (!menu || actions.length === 0) return null;

  const title = path.reduce<{ label: string; level: readonly EditorAction[] }>(
    (carried, index) => ({
      label: carried.level[index]?.label ?? carried.label,
      level: carried.level[index]?.children ?? [],
    }),
    { label: menu.title, level: menu.actions }).label;

  const choose = (action: EditorAction, index: number) => {
    if (!actionIsChoosable(action)) return;
    if (action.children?.length) { setPath([...path, index]); setFocused(0); return; }
    if (!action.effect) return;
    closeRadialMenu();
    performEditorAction(action.effect, { navigate: (href) => router.push(href) });
  };

  const back = () => {
    if (path.length === 0) { closeRadialMenu(); return; }
    setPath(path.slice(0, -1));
    setFocused(0);
  };

  const span = (2 * Math.PI) / actions.length;
  // The first wedge is centred at the top, so a ring of two is left/right and a
  // ring of four is up/right/down/left — the directions a flick already means.
  const start = -Math.PI / 2 - span / 2;
  const focusedAction = actions[focused];
  const centre = clampToViewport(menu.x, menu.y);
  // Opened low, the hint would land on the transport cluster. The ring itself
  // may not move — its wedges are directions — so the prose goes above it
  // instead, which is the one part of this menu a reader is not aiming at.
  const viewportHeight = typeof window === "undefined" ? Infinity : window.innerHeight;
  const readoutFlipped = centre.y + RING_BOX_PX / 2 + READOUT_GAP_PX + READOUT_HEIGHT_PX
    > viewportHeight - TRANSPORT_KEEPOUT_PX;

  return <div
    className="radial-menu-layer"
    data-testid="radial-menu"
    onPointerDown={(event) => { if (event.target === event.currentTarget) closeRadialMenu(); }}
    onContextMenu={(event) => event.preventDefault()}
  >
    {/* Sized to the ring alone, so translating it by half its own box puts the
        hub exactly under the pointer. The hint hangs off it out of flow — a
        readout that could widen or heighten this box would drag the wedge
        directions off the click, which is the one thing a pie must not do. */}
    <div className="radial-menu" style={{
      left: `${centre.x}px`,
      top: `${centre.y}px`,
      width: `${RING_BOX_PX}px`,
      height: `${RING_BOX_PX}px`,
    }}>
      <svg
        ref={ringRef}
        className="radial-ring"
        data-dense={actions.length >= DENSE_WEDGE_COUNT}
        viewBox={`${-RING_BOX_PX / 2} ${-RING_BOX_PX / 2} ${RING_BOX_PX} ${RING_BOX_PX}`}
        width={RING_BOX_PX}
        height={RING_BOX_PX}
        role="menu"
        aria-label={title}
        tabIndex={-1}
        onKeyDown={(event) => {
          if (event.key === "Escape") { event.preventDefault(); back(); return; }
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            if (focusedAction) choose(focusedAction, focused);
            return;
          }
          const forward = event.key === "ArrowRight" || event.key === "ArrowDown" || (event.key === "Tab" && !event.shiftKey);
          const backward = event.key === "ArrowLeft" || event.key === "ArrowUp" || (event.key === "Tab" && event.shiftKey);
          if (!forward && !backward) return;
          event.preventDefault();
          setFocused((current) => (current + (forward ? 1 : -1) + actions.length) % actions.length);
        }}
      >
        {actions.map((action, index) => {
          const from = start + index * span + WEDGE_GAP_RAD;
          const to = start + (index + 1) * span - WEDGE_GAP_RAD;
          const middle = (from + to) / 2;
          // Icon above caption in *screen* space, not along the radius: stacking
          // radially puts the picture beside the word on the side wedges and
          // under it on the bottom one, so the same menu reads as three
          // different layouts as the eye goes round it.
          const [anchorX, anchorY] = polar(LABEL_RADIUS_PX, middle);
          const [iconX, iconY] = [anchorX, anchorY - ICON_OFFSET_PX];
          const [labelX, labelY] = [anchorX, anchorY + LABEL_OFFSET_PX];
          const choosable = actionIsChoosable(action);
          return <g
            key={action.id}
            className={`radial-wedge tone-${action.tone}`}
            role="menuitem"
            aria-disabled={!choosable}
            aria-haspopup={action.children?.length ? "menu" : undefined}
            data-focused={index === focused}
            data-choosable={choosable}
            onPointerEnter={() => setFocused(index)}
            onClick={() => choose(action, index)}
          >
            <path d={wedgePath(from, to)} />
            {action.icon && <EditorActionIconMark name={action.icon} x={iconX} y={iconY} />}
            <text className="radial-label" x={labelX} y={labelY}>{action.label}</text>
            {(action.children?.length ?? 0) > 0 && <path
              className="radial-more"
              d="M -2.5 -3.5 L 2.5 0 L -2.5 3.5"
              transform={`translate(${polar(OUTER_RADIUS_PX - 8, middle).join(" ")}) rotate(${(middle * 180) / Math.PI})`}
            />}
          </g>;
        })}
        {/* The hub is the way back, which is what the centre of a weapon wheel
            means everywhere it appears: no wedge is under the pointer, so
            nothing is chosen. One level up from a sub-ring, closed from the
            root — the pointer twin of Escape. */}
        <g
          className="radial-hub-group"
          role="menuitem"
          aria-label={path.length > 0 ? "Back" : "Close"}
          onClick={back}
        >
          <circle className="radial-hub" r={INNER_RADIUS_PX - 8} />
          <text className="radial-title" y={4}>{path.length > 0 ? "‹" : "·"}</text>
        </g>
      </svg>
      <div className="radial-readout" role="status" data-flip={readoutFlipped ? "above" : undefined}>
        <strong>{title}</strong>
        <span>{focusedAction?.hint ?? (path.length > 0 ? "Esc goes back" : "Esc closes")}</span>
      </div>
    </div>
  </div>;
}
