"use client";

import {
  SVO_PIXEL_TRACE_LAYERS,
  SVO_PIXEL_TRACE_LAYER_DEFINITIONS,
  SVO_PIXEL_TRACE_STATUS,
  svoPixelTraceLayerForKind,
  svoPixelTraceMipLadder,
  svoPixelTraceNarrative,
  svoPixelTraceTotalWork,
  type SvoPixelTrace,
  type SvoPixelTraceLayer,
} from "@/lib/svo-pixel-trace";
import type { PixelTraceStatus } from "@/lib/webgpu-renderer";

interface PixelTraceHudProps {
  readonly trace: SvoPixelTrace | undefined;
  readonly enabledLayers: readonly SvoPixelTraceLayer[];
  readonly pinned: boolean;
  /** Renderer-side reason the diagnostic is or is not producing traces. */
  readonly probeStatus: PixelTraceStatus;
  /** False until the pointer has been over the viewport at least once. */
  readonly pointerSeen: boolean;
  /**
   * The pinned trace describes a scene that has since changed, and its aim cannot
   * be re-traced because the camera has moved. Saying so beats both silently
   * showing old numbers and silently swapping in a different ray's.
   */
  readonly stale: boolean;
  readonly onToggleLayer: (layer: SvoPixelTraceLayer) => void;
  readonly onTogglePinned: () => void;
  readonly onClose: () => void;
}

/**
 * What to say when there is no trace. Each case is a different thing for the
 * user to do, which is why they are not one message.
 */
function blockedReason(probeStatus: PixelTraceStatus, pointerSeen: boolean): { headline: string; detail: string } | undefined {
  if (probeStatus === "unsupported") {
    return {
      headline: "probe unavailable",
      detail: "This device has no storage-binding slot left for the trace record buffer. The console has the exact limit.",
    };
  }
  if (probeStatus === "path-inactive") {
    return {
      headline: "sparse path inactive",
      detail: "This frame is on the raster fallback, so there is no octree traversal to record. The Presentation group above reports why.",
    };
  }
  if (probeStatus === "compiling") {
    return { headline: "compiling the probe", detail: "The probe shader builds once per session, on first use." };
  }
  if (!pointerSeen) {
    return { headline: "move the pointer", detail: "Hover anywhere over the viewport to trace the ray behind that pixel." };
  }
  return probeStatus === "waiting"
    ? { headline: "waiting for the first readback", detail: "The probe traces one pixel per frame and reads it back a frame later." }
    : undefined;
}

/** Voxel widths span millimetres to metres, so the unit has to follow. */
function widthLabel(width_m: number | undefined): string {
  if (width_m === undefined || !Number.isFinite(width_m)) return "";
  if (width_m >= 1) return `${width_m.toFixed(2)} m`;
  return width_m >= 0.01 ? `${(width_m * 100).toFixed(1)} cm` : `${(width_m * 1000).toFixed(1)} mm`;
}

const STATUS_LABEL: Readonly<Record<number, string>> = {
  [SVO_PIXEL_TRACE_STATUS.pending]: "waiting for the probe",
  [SVO_PIXEL_TRACE_STATUS.hit]: "surface found",
  [SVO_PIXEL_TRACE_STATUS.miss]: "ray left the scene",
  [SVO_PIXEL_TRACE_STATUS.unavailable]: "publication incomplete",
  [SVO_PIXEL_TRACE_STATUS.exhausted]: "bounded work exhausted",
  [SVO_PIXEL_TRACE_STATUS.invalid]: "invalid topology · failed closed",
};

/**
 * Readout for the live ray-work diagnostic.
 *
 * The 3D overlay shows where the work happened; this shows what it was, in the
 * order the shader did it, and which layers of it are currently drawn. Every
 * number comes from the probe's own counters rather than from an estimate.
 */
export function PixelTraceHud({
  trace, enabledLayers, pinned, probeStatus, pointerSeen, stale, onToggleLayer, onTogglePinned, onClose,
}: PixelTraceHudProps) {
  const blocked = trace ? undefined : blockedReason(probeStatus, pointerSeen);
  const counts = new Map<SvoPixelTraceLayer, number>();
  for (const record of trace?.records ?? []) {
    const layer = svoPixelTraceLayerForKind(record.kind);
    counts.set(layer, (counts.get(layer) ?? 0) + 1);
  }
  const narrative = trace ? svoPixelTraceNarrative(trace) : [];
  const mipLadder = trace ? svoPixelTraceMipLadder(trace) : [];
  const totalWork = trace ? svoPixelTraceTotalWork(trace) : 0;
  const status = trace ? STATUS_LABEL[trace.status] ?? "unknown" : blocked?.headline ?? "waiting for the probe";
  const failed = trace?.status === SVO_PIXEL_TRACE_STATUS.invalid
    || trace?.status === SVO_PIXEL_TRACE_STATUS.exhausted
    || probeStatus === "unsupported";

  return (
    <div
      className="pixel-trace-hud"
      data-testid="pixel-trace-hud"
      data-pinned={pinned ? "true" : "false"}
      data-stale={stale ? "true" : "false"}
      data-probe-status={probeStatus}
    >
      <header>
        <div>
          <small>Ray work · {stale ? "pinned, scene changed" : pinned ? "pinned" : "live"}</small>
          <h3>{trace ? `pixel ${trace.pixel[0]}, ${trace.pixel[1]}` : blocked?.headline ?? "no trace"}</h3>
        </div>
        <div className="pixel-trace-actions">
          <button
            type="button"
            aria-pressed={pinned}
            onClick={onTogglePinned}
            title={pinned ? "Follow the pointer again" : "Freeze this ray and orbit around it — clicking the viewport does the same"}
          >
            {pinned ? "Unpin" : "Pin ray"}
          </button>
          <button type="button" onClick={onClose} title="Close the ray-work diagnostic">Close</button>
        </div>
      </header>

      {trace ? (
        <div className="pixel-trace-headline">
          <strong>{totalWork.toLocaleString()}</strong>
          <span>units of work for this one pixel</span>
          <em data-failed={failed ? "true" : "false"}>{status}</em>
        </div>
      ) : (
        <p className="pixel-trace-blocked" data-failed={failed ? "true" : "false"}>{blocked?.detail}</p>
      )}

      {trace && <ol className="pixel-trace-steps">
        {narrative.map((step) => (
          <li key={step.id} data-layer={step.layer ?? "none"}>
            <i style={step.layer ? { background: SVO_PIXEL_TRACE_LAYER_DEFINITIONS[step.layer].swatch } : undefined} />
            <b>{step.label}</b>
            <output>{step.value}</output>
            <small>{step.detail}</small>
          </li>
        ))}
      </ol>}

      {stale && (
        <p className="pixel-trace-note pixel-trace-restale">
          The scene changed after this ray was pinned, and the camera has moved since, so the pinned pixel no longer names
          this ray. These counters describe the earlier frame. Unpin to follow the pointer again.
        </p>
      )}

      {mipLadder.length > 0 && (
        <div className="pixel-trace-mips" data-testid="pixel-trace-mips">
          <small>Mip levels read by the cones — each level is a voxel twice as wide as the one below</small>
          <div>
            {mipLadder.map((rung) => (
              <span key={rung.level} style={{ borderColor: rung.swatch }}>
                <i style={{ background: rung.swatch }} />
                <b>L{rung.level}</b>
                <output>{rung.taps}</output>
                {rung.footprint_m !== undefined && <em>{widthLabel(rung.footprint_m)}</em>}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="pixel-trace-layers">
        {SVO_PIXEL_TRACE_LAYERS.map((layer) => {
          const definition = SVO_PIXEL_TRACE_LAYER_DEFINITIONS[layer];
          const active = enabledLayers.includes(layer);
          return (
            <button
              key={layer}
              type="button"
              aria-pressed={active}
              onClick={() => onToggleLayer(layer)}
              title={definition.description}
            >
              <i style={{ background: definition.swatch, opacity: active ? 1 : 0.3 }} />
              <span>{definition.label}</span>
              {/* The ray is one path the overlay derives, not a count of work
                  items, so it carries no tally to report. */}
              {layer !== "primary-ray" && <small>{counts.get(layer) ?? 0}</small>}
            </button>
          );
        })}
      </div>

      {trace && trace.droppedRecords > 0 && (
        <p className="pixel-trace-note">
          {trace.droppedRecords.toLocaleString()} records past the capture buffer are not drawn. The prefix shown is exact.
        </p>
      )}
      <p className="pixel-trace-note pixel-trace-footnote">
        Recorded by a probe that mirrors the shipping traversal, brick walk, and cone step law.
        {pinned
          ? " Pinned: orbit to inspect the frozen ray, or click the viewport to follow the pointer again."
          : " Click the viewport to freeze the ray under the pointer and orbit around its work."}
      </p>
    </div>
  );
}
