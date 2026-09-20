"use client";
import { VisualLayerRows } from "../lib/features/field-view/layers-ui";
import { legacyVisualLayers } from "../lib/core/visual-layers";

import { getMethod } from "../lib/core/method-registry";
import { VISUALIZATION_FIELDS, VISUALIZATION_QUICK_FIELDS } from "../lib/core/visualization-catalog";
import { FieldViewRows as SharedFieldViewRows } from "../lib/features/field-view/ui";
import { useSession } from "../lib/core/session/session-context";
import { DEFAULT_GRID_OVERLAY_AXIS } from "../lib/core/stores/ui-store";
import type { GridOverlayMode } from "../lib/core/webgpu-renderer";

/**
 * The studio's binding of the shared field-view row.
 *
 * Everything that used to be in this file is now
 * `lib/features/field-view/ui.tsx`, rendered by both hosts — the mark, the
 * chevron onto everything the solver publishes, the plane and the scrub. What
 * is left here is the studio's two arguments, which is all this file ever
 * genuinely owned:
 *
 *   - **which views** — the catalog narrowed by the running method's
 *     `supportedFieldModes`, so a method that draws no volume never offers a
 *     volume view;
 *   - **where the reading lives** — the six ui-store members that carry the
 *     mode, the plane and the scrub for this pane.
 *
 * The lab supplies its own two and gets the same row. That is the whole of the
 * plugin claim on this capability.
 */

/**
 * A catalog field narrowed to the mode union the overlay store speaks. The
 * narrowing is an assertion, as the flyout's own list is: `mode` is declared as
 * a string beside the pass that publishes it, because core cannot depend on the
 * renderer's union.
 */
type Field = (typeof VISUALIZATION_FIELDS)[number] & { mode: GridOverlayMode };

const FIELDS = VISUALIZATION_FIELDS as readonly Field[];

/** Whether this method publishes anything the row can offer. */
export function methodHasQuickFields(methodId: string): boolean {
  const supported = new Set(getMethod(methodId).supportedFieldModes ?? []);
  return VISUALIZATION_QUICK_FIELDS.some((view) => supported.has(view.mode as GridOverlayMode));
}

export function FieldViewRows() {
  const session = useSession();
  const methodId = session.method((state) => state.methodId);
  const overlayMode = session.ui((state) => state.gridOverlayMode);
  const overlayAxis = session.ui((state) => state.gridOverlayAxis);
  const overlaySlice = session.ui((state) => state.gridOverlaySlice);
  const setOverlayMode = session.ui((state) => state.setGridOverlayMode);
  const setOverlayAxis = session.ui((state) => state.setGridOverlayAxis);
  const setOverlaySlice = session.ui((state) => state.setGridOverlaySlice);

  const layers = session.ui(state => state.visualLayers);
  const method = getMethod(methodId);
  const volumeCapable = method.capabilities?.volumeRendering === true;
  const supported = new Set(method.supportedFieldModes ?? []);
  if (methodId === "uniform-volume") return <VisualLayerRows
    state={layers ?? { ...legacyVisualLayers(overlayMode), visible: overlayAxis !== "off" }}
    onChange={visualLayers => session.ui.setState({
      visualLayers,
      gridOverlayMode: "structure",
      gridOverlayAxis: overlayAxis === "off" || overlayAxis === "volume" ? "z" : overlayAxis,
    })}
    plane={{ axis: overlayAxis, slice: overlaySlice, setAxis: setOverlayAxis, setSlice: setOverlaySlice }}
  />;
  return <SharedFieldViewRows
    // Catalog order, narrowed to this solver. The shared row splits it into the
    // short list and the rest on the `icon` each pass declared, which is the
    // same split `VISUALIZATION_QUICK_FIELDS` is derived by.
    views={FIELDS.filter((view) => supported.has(view.mode))}
    // What is drawing, whether or not this strip could have chosen it: the full
    // catalog and a link can both leave a view on that has no glyph here.
    catalog={FIELDS}
    volumeCapable={volumeCapable}
    state={{
      mode: overlayMode,
      axis: overlayAxis,
      slice: overlaySlice,
      setMode: (mode) => setOverlayMode(mode as GridOverlayMode),
      setAxis: setOverlayAxis,
      setSlice: setOverlaySlice,
      defaultAxis: DEFAULT_GRID_OVERLAY_AXIS,
    }}
  />;
}
