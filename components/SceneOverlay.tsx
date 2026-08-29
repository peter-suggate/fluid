"use client";

import { useEffect, useState } from "react";
import { SceneOverridesChip } from "./SceneOverridesChip";
import { findSceneDefinition } from "../lib/core/scenes";
import { simulation } from "../lib/core/simulation/controller";
import { planSceneRuntime } from "../lib/core/scene-runtime";
import { useSession } from "../lib/core/session/session-context";

/** How far into the corner the pointer has to come before the chip appears. */
const REVEAL_WIDTH_PX = 420;
const REVEAL_HEIGHT_PX = 200;

/**
 * Whether the pointer is in the top-left corner of the window.
 *
 * A rectangle rather than the element's own `:hover`, because the element is
 * invisible until this is true: hovering something with zero opacity is a
 * gesture nobody can aim, so the region that reveals it has to be larger than
 * the thing revealed. It is deliberately generous — the reader is heading for a
 * corner, not for a 43 px pill.
 */
function usePointerNearTopLeft() {
  const [near, setNear] = useState(false);
  useEffect(() => {
    const onPointerMove = (event: PointerEvent) => {
      setNear(event.clientX <= REVEAL_WIDTH_PX && event.clientY <= REVEAL_HEIGHT_PX);
    };
    window.addEventListener("pointermove", onPointerMove, { passive: true });
    return () => window.removeEventListener("pointermove", onPointerMove);
  }, []);
  return near;
}

/**
 * Which scene is loaded, the way back to the library, and the way to keep what
 * you have made of it — none of which is worth a permanent mark on the image.
 *
 * This was a dropdown over every preset — twenty-five options, of which
 * thirteen were analytic oracles, in a 214 px pill. Choosing what to explore is
 * not a form control, so it moved to the library and this became a chip that
 * names where you are and opens it. That also retires a real hazard: a focused
 * dropdown reads as text editing to the shortcut chassis, so choosing a scene
 * used to silently disable every single-key shortcut until something else took
 * focus, and the fix was a `blur()` someone had to remember. A button cannot
 * hold that state at all.
 *
 * It now answers only when asked. The cluster stays mounted and focusable at all
 * times — hiding it from the tab order would hide the only route back to the
 * library from a keyboard — and CSS fades it in when the pointer nears the
 * corner or anything inside it takes focus.
 */
export function SceneOverlay() {
  const session = useSession();
  const presetId = session.scene((state) => state.presetId);
  const scene = session.scene((state) => state.scene);
  const selectorOpen = session.ui((state) => state.sceneSelectorOpen);
  const setSelectorOpen = session.ui((state) => state.setSceneSelectorOpen);
  const pointerNear = usePointerNearTopLeft();
  const definition = findSceneDefinition(presetId);
  const runtime = planSceneRuntime(scene).fluidSolver ? `seed ${scene.randomSeed}` : "live SVO · no fluid";
  return (
    <div
      className="scene-overlay"
      data-testid="scene-panel"
      data-revealed={pointerNear ? "true" : "false"}
    >
      <span className="brand-mark" title="Fluid Lab · WebGPU CFD workbench">FL</span>
      {/* The name is the switch. It used to be the way to the library, which
          meant that changing scene was a route change: the viewport unmounted,
          the device went away, and coming back rebuilt everything — for a
          gesture that is usually "try the next one". It now raises this pane's
          selector, which swaps the document in place and keeps the library one
          row away inside it. */}
      <button
        type="button"
        className="scene-overlay-chip"
        onClick={() => setSelectorOpen(!selectorOpen)}
        data-scene-selector-toggle=""
        data-testid="open-scene-library"
        aria-haspopup="dialog"
        aria-expanded={selectorOpen}
        title={definition ? `${definition.blurb}\n\nChoose another scene` : "Choose another scene"}
      >
        <strong>{definition?.name ?? scene.sceneId}</strong>
        <small>{scene.sceneId} · {runtime}</small>
      </button>
      {/* Between the name and SAVE deliberately: it qualifies the scene you are
          looking at, and it is part of what saving would keep. Renders nothing
          at all when the link is clean. */}
      <SceneOverridesChip />
      {/* Saving is the one thing the library route cannot do for you: it lists
          what is in this browser's storage, and only the studio knows what the
          open document has become. Configuration is not here any more — it is on
          the tank, reached by selecting it or through its ring. */}
      <button
        type="button"
        className="scene-overlay-action"
        onClick={() => simulation.saveNamedScene(scene.sceneId, session.id)}
        data-testid="save-scene"
        title={`Save this document to the library as “${scene.sceneId}”, replacing an earlier save of the same name`}
      >
        <i aria-hidden="true" />SAVE
      </button>
    </div>
  );
}
