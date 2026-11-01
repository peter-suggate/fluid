"use client";

import { requestManualGPUStart } from "../lib/core/gpu-startup";
import { resourceActivities } from "../lib/core/resource-readiness";
import type { PaneId } from "../lib/core/session/session";
import { useSession } from "../lib/core/session/session-context";
import { useSafeBrowserGPUBringup } from "../lib/core/use-safe-browser-gpu-bringup";
import { EditorModeChip } from "./EditorModeChip";
import { PipelineOverlay } from "./PipelineOverlay";
import { RadialMenu } from "./RadialMenu";
import { SceneScaleOverlay } from "./SceneScaleOverlay";
import { SceneSelector } from "./SceneSelector";
import { ViewRail } from "./ViewRail";
import { VoxelToolRail } from "./VoxelToolRail";
import { WebGPUViewport } from "./WebGPUViewport";
import { InteractionLayer } from "../lib/features/interactive/ui";
import { PaneControlWidgets } from "../lib/features/ui/FeatureSlot";

/**
 * One pane of scene: a canvas and everything drawn over it.
 *
 * Every child here reads the session this component is mounted under, so a
 * second pane is a second `<SessionProvider>` around a second one of these and
 * nothing below learns a new code path — which is the whole bargain of the
 * session realm. What is deliberately *not* here is the chrome that belongs to
 * the page rather than to a pane: the transport, the scene chip and the
 * recording modal are the host's, because there is one clock, one document
 * being authored and one recorder. What a pane is waiting on is said there
 * too, under the controls it holds (see `usePaneActivity`).
 */

export interface ScenePaneProps {
  readonly paneId: PaneId;
  /** Draw the A / B tag. Absent in single-pane mode: there is nothing to tell apart. */
  readonly tagged?: boolean;
  /** The pane the keyboard and the ring belong to. */
  readonly focused?: boolean;
  readonly onFocus?: () => void;
}

export function ScenePane({ paneId, tagged = false, focused = false, onFocus }: ScenePaneProps) {
  const safeBringup = useSafeBrowserGPUBringup() === true;
  const session = useSession();
  const selectorOpen = session.ui((state) => state.sceneSelectorOpen);
  const setSelectorOpen = session.ui((state) => state.setSceneSelectorOpen);
  const gpuStatus = session.diagnostics((state) => state.gpuStatus);
  const resourceReadiness = session.diagnostics((state) => state.resourceReadiness);
  const stopped = ["unavailable", "lost", "stopping", "blocked"].includes(gpuStatus.state);
  // What the pane is waiting on is stated on the transport card, under the
  // controls it holds; the pane only marks itself as busy for its own styling.
  const activities = stopped ? [] : resourceActivities(resourceReadiness);

  return (
    <section
      className="viewport-shell"
      data-pane={paneId}
      data-pane-focused={focused}
      data-resource-active={activities.length > 0}
      data-gpu-transition={activities.at(-1)?.lane ?? resourceReadiness.platform.state}
      // Capture, so focus follows a press that a child stops: a right-click that
      // opens the ring has to focus its own pane before the ring is composed.
      onPointerDownCapture={onFocus}
      onPointerEnter={onFocus}
    >
      <WebGPUViewport paneId={paneId} />
      {/* Inert unless this pane has an interaction binding (`?interactive=demo`). */}
      <InteractionLayer paneId={paneId} focused={focused} />
      {/* Throttles, sticks and buttons for whatever in the scene can be driven; nothing when nothing can. */}
      <PaneControlWidgets paneId={paneId} />
      <EditorModeChip />
      <VoxelToolRail />
      <ViewRail />
      <RadialMenu />
      <SceneScaleOverlay />
      <PipelineOverlay />
      {/* The tag names the pane and is also its scene switch. A compare is only
          worth opening once the two panes can differ, and the coarsest way they
          differ is by running different scenes — so the affordance sits on the
          one mark that already says "this pane", rather than as a second badge
          beside it. */}
      {tagged && <button
        type="button"
        className="pane-tag"
        data-pane={paneId}
        data-pane-focused={focused}
        data-scene-selector-toggle=""
        data-testid={`pane-tag-${paneId}`}
        aria-haspopup="dialog"
        aria-expanded={selectorOpen}
        title={`Pane ${paneId.toUpperCase()} — choose the scene this pane runs`}
        onClick={() => setSelectorOpen(!selectorOpen)}
      >{paneId.toUpperCase()}</button>}
      {selectorOpen && <SceneSelector />}
      {gpuStatus.state === "manual" && <div className="gpu-fallback gpu-manual-start" role="status">
        <strong>WebGPU startup paused</strong>
        <p>{safeBringup
          ? "Bounded bring-up permits the authored 384-column dam break, one STEP, then an explicit STOP GPU. Close every Dawn process first."
          : gpuStatus.label}</p>
        <button type="button" onClick={requestManualGPUStart}>START WEBGPU</button>
        <small>{safeBringup
          ? "This browser can exclude other Fluid Lab tabs, but cannot observe Dawn's local filesystem lease."
          : <>Use <code>gpu=off</code> for UI-only inspection or <code>gpu=on</code> to restore automatic startup.</>}</small>
      </div>}
    </section>
  );
}
