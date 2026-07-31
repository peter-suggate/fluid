"use client";

import { visualizationsForGroups } from "@/lib/visualization-catalog";
import {
  fineBandCensus,
  fineBandFreeSurfaceCrossings,
  fineBandLadderAttribution,
  fineBandLadderPassColor,
  fineBandMembership,
  fineBandNarrative,
  fineBandPhiSpan,
  type FineBandCellContext,
  type FineBandPhiSpan,
} from "@/lib/fine-band-cell-model";
import {
  FLUID_CELL_TRACE_LAYERS,
  FLUID_CELL_TRACE_LAYER_DEFINITIONS,
  FLUID_CELL_TRACE_STATUS,
  fluidCellTraceDirectionLabel,
  fluidCellTraceHasInterfaceHit,
  fluidCellTraceKeyFigures,
  fluidCellTraceLayerForNeighbor,
  fluidCellTraceNarrative,
  type FluidCellTrace,
  type FluidCellTraceLayer,
  type FluidCellTraceSchedule,
} from "@/lib/fluid-cell-trace";
import { FLUID_CELL_TRACE_FINE_PROBES } from "@/lib/webgpu-fluid-cell-trace";
import { VisualizationLegend } from "./VisualizationLegend";

export type FluidCellTraceStatusHint =
  | "unavailable" | "compiling" | "waiting" | "ready";

interface FluidCellTraceHudProps {
  /**
   * Whether the account below the legend is unfolded.
   *
   * Held by the caller rather than here so it survives the HUD unmounting when
   * pick mode is left and re-entered — a reader who has asked for the detail
   * has asked once, not once per session with the panel.
   */
  readonly expanded: boolean;
  readonly onToggleExpanded: () => void;
  readonly trace: FluidCellTrace | undefined;
  readonly schedule: FluidCellTraceSchedule | undefined;
  /**
   * Band widths and the ladder that ran. Absent on a scene with no fine band,
   * which is why every fine-band panel below is conditional rather than
   * rendering an empty frame.
   */
  readonly fineBand: FineBandCellContext | undefined;
  readonly enabledLayers: readonly FluidCellTraceLayer[];
  readonly pinned: boolean;
  readonly probeStatus: FluidCellTraceStatusHint;
  readonly pointerSeen: boolean;
  readonly onToggleLayer: (layer: FluidCellTraceLayer) => void;
  readonly onTogglePinned: () => void;
  /** Walk the selection along the pointer ray; the gather wraps at the ends. */
  readonly onStepHit: (delta: number) => void;
  /** Jump the selection to the next leaf on the run that holds the interface. */
  readonly onJumpToInterface: () => void;
  readonly onClose: () => void;
}

/**
 * What to say when there is no trace. Each case is a different thing for the
 * user to do, which is why they are not one message.
 */
function blockedReason(
  probeStatus: FluidCellTraceStatusHint, pointerSeen: boolean,
): { headline: string; detail: string } | undefined {
  if (probeStatus === "unavailable") {
    return {
      headline: "no pressure publication",
      detail: "This frame has no compact octree topology to read, so no cell can be named. The Presentation group above reports why.",
    };
  }
  if (probeStatus === "compiling") {
    return { headline: "compiling the gather", detail: "The gather shader builds once per session, on first use." };
  }
  if (!pointerSeen) {
    return { headline: "move the pointer", detail: "Hover over the fluid to inspect the pressure cell behind that pixel." };
  }
  return probeStatus === "waiting"
    ? { headline: "waiting for the first readback", detail: "The gather runs one cell per frame and reads it back a frame later." }
    : undefined;
}

/**
 * Phi across the selected cell, on one axis.
 *
 * The orienting graphic, and the reason it comes before the narrative: every
 * other fine-band figure is a detail of what this bar shows at a glance —
 * whether this one unknown sits in liquid, in air, or across the surface.
 *
 * Two spans are drawn rather than one because they can disagree, and their
 * disagreement is the finding. The **record** is what the pressure row will
 * read; the **probes** are what the fine band actually holds. Full scale is the
 * leaf's own half-diagonal, since phi cannot vary by more than that inside the
 * cell without failing the Lipschitz condition redistancing exists to restore —
 * so a span filling the bar is a band in trouble, not merely a big cell.
 */
function PhiSpanBar({ span }: { span: FineBandPhiSpan }) {
  const scale = Math.max(span.scale, 1e-6);
  const at = (value: number) => 50 + 50 * Math.max(-1, Math.min(1, value / scale));
  const bracket = (low: number, high: number) => ({
    left: `${at(low)}%`, width: `${Math.max(0.6, at(high) - at(low))}%`,
  });
  return (
    <div className="fine-band-span" data-testid="fine-band-span">
      <small>
        φ across this cell — the one number Section 5 hands the free-surface condition
      </small>
      <div className="fine-band-span-track">
        {span.published && <span className="record" style={bracket(span.minimum, span.maximum)} />}
        {span.probeMinimum !== span.probeMaximum && (
          <span className="probes" style={bracket(span.probeMinimum, span.probeMaximum)} />
        )}
        <i className="zero" />
        {span.published && (
          <b className="centre" style={{ left: `${at(span.centre)}%` }} data-testid="fine-band-centre" />
        )}
      </div>
      <div className="fine-band-span-scale">
        <em>liquid −{span.scale.toFixed(1)}</em>
        <output>
          {span.published ? `φ = ${span.centre.toFixed(3)} cells` : "no coarse record"}
        </output>
        <em>+{span.scale.toFixed(1)} air</em>
      </div>
      {span.unresolvedInterface && (
        <p className="fine-band-warning" data-failed="true" data-testid="fine-band-subgrid">
          The band holds samples on both sides of the surface and this row&rsquo;s record does not.
          The solve is about to run as though the interface were not here — the paper&rsquo;s own
          sub-grid case (§8).
        </p>
      )}
      {!span.unresolvedInterface && span.interfaceOffCentre && (
        <p className="fine-band-warning">
          The surface is inside this cell but {Math.abs(span.centre).toFixed(2)} cells from its site,
          so accuracy near the surface rests entirely on the crossings below.
        </p>
      )}
    </div>
  );
}

/**
 * Where this cell sits in the nest of bands.
 *
 * Four bars on one axis rather than four numbers: the authored widths arrive in
 * finest cells and the derived ones in fine cells, and a reader should not have
 * to do that conversion to see that the transport reach is inside the
 * redistance support. The marker is the cell's own distance to the surface, so
 * the question "why is this cell refined at all" is answered by which bars it
 * falls inside.
 */
function BandNest({
  distance, context, fineFactor,
}: {
  readonly distance: number;
  readonly context: FineBandCellContext;
  readonly fineFactor: number;
}) {
  const membership = fineBandMembership(distance, context.widths, fineFactor);
  if (membership.rings.length === 0) return null;
  const full = Math.max(
    membership.rings[membership.rings.length - 1].radius,
    Number.isFinite(membership.distance) ? membership.distance : 0,
    1e-6,
  );
  return (
    <div className="fine-band-nest" data-testid="fine-band-nest">
      <small>Bands — why this cell is refined</small>
      {/* Bars only. At this column width four inline labels do not fit, and the
          colours are the `band-residency` field's own — so a reader who has
          seen the volumetric view already knows them, the tooltip names each
          one, and the line below names the one that matters. */}
      <div className="fine-band-nest-rows">
        {membership.rings.map((ring) => (
          <span
            key={ring.id}
            title={`${ring.label}: ${ring.radius.toFixed(1)} finest cells`}
            aria-label={`${ring.label} ${ring.radius.toFixed(1)} finest cells`}
          >
            <i
              style={{
                background: ring.swatch,
                width: `${(ring.radius / full) * 100}%`,
                opacity: ring.contains ? 1 : 0.35,
              }}
            />
          </span>
        ))}
        {Number.isFinite(membership.distance) && (
          <b style={{ left: `${Math.min(100, (membership.distance / full) * 100)}%` }} />
        )}
      </div>
      <output>
        {membership.innermost
          ? `${membership.distance.toFixed(2)} cells from the surface — inside the ${membership.innermost.label.toLowerCase()}`
          : `${membership.distance.toFixed(2)} cells from the surface — outside every band`}
      </output>
    </div>
  );
}

/**
 * This leaf's own share of the redistance ladder.
 *
 * The global provenance report says roughly half the field closes on the first
 * stride-8 pass. That is a property of the domain, not of any cell, and the
 * only way to know which side of it a given cell falls on is to bin that cell's
 * own probes against the ladder that ran — which is what this is. A leaf whose
 * bar is all late passes is one the schedule is genuinely paying for.
 */
function FloodLadder({
  trace, context,
}: { readonly trace: FluidCellTrace; readonly context: FineBandCellContext }) {
  const attribution = fineBandLadderAttribution(trace.fineProbeRecords, context.ladderStrides);
  if (!attribution || attribution.resolved === 0) return null;
  const passes = attribution.plan.strides.length;
  return (
    <div className="fine-band-ladder" data-testid="fine-band-ladder">
      <small>Flood ladder — which pass closed this leaf</small>
      <div className="fine-band-ladder-track">
        {attribution.bins.map((value, bin) => value === 0 ? null : (
          <i
            key={bin}
            style={{
              background: fineBandLadderPassColor(bin, passes),
              width: `${(value / attribution.resolved) * 100}%`,
            }}
            title={bin === 0
              ? `${value} held their own crossing`
              : `${value} closed by pass ${bin} of ${passes}`}
          />
        ))}
      </div>
      <output>
        {`pass ${attribution.requiredPasses} of ${passes} covers this leaf · deepest hop `
          + `${attribution.deepestHop} fine cells`}
        {!attribution.coveredByEncodedLadder && " · carried from an earlier frame"}
      </output>
    </div>
  );
}

/**
 * The legend, read from the catalog.
 *
 * Each row's glyph, colour, label and description come from the pass that draws
 * that layer, so a row cannot go on describing a mark its decorator stopped
 * using. The store still speaks the reader's vocabulary — "stencil", "cone" —
 * which is the `group` tag those declarations carry.
 */
const LAYER_LEGEND = visualizationsForGroups(FLUID_CELL_TRACE_LAYERS);

const STATUS_LABEL: Readonly<Record<number, string>> = {
  [FLUID_CELL_TRACE_STATUS.pending]: "waiting for the gather",
  [FLUID_CELL_TRACE_STATUS.resolved]: "cell found",
  [FLUID_CELL_TRACE_STATUS.miss]: "no pressure cell under the pointer",
  [FLUID_CELL_TRACE_STATUS.unavailable]: "publication incomplete",
  [FLUID_CELL_TRACE_STATUS.invalid]: "invalid row · failed closed",
};

/**
 * Readout for the per-cell work diagnostic.
 *
 * The ray probe's HUD can say "this is what the shader did, in order", because
 * a ray is one invocation. A cell is not, so this separates the two things it
 * can honestly say: what the frame published about this cell, and what the
 * encoded command graph does to a row. Every line is badged with which it is,
 * and the two are never added together into one figure.
 *
 * Laid out for the way it is actually used: sweep the pointer over the fluid
 * and read four figures and a legend, then unfold the account for the one cell
 * that turns out to be worth it. The unfolded half is unchanged — it was always
 * the right content and the wrong first impression.
 */
export function FluidCellTraceHud({
  trace, schedule, fineBand, enabledLayers, pinned, probeStatus, pointerSeen, expanded,
  onToggleLayer, onTogglePinned, onStepHit, onJumpToInterface, onToggleExpanded, onClose,
}: FluidCellTraceHudProps) {
  const resolved = trace?.status === FLUID_CELL_TRACE_STATUS.resolved ? trace : undefined;
  const blocked = resolved ? undefined : blockedReason(probeStatus, pointerSeen);
  // The fine band's account is appended rather than interleaved: the pressure
  // solve's story has to stay readable on a scene that carries no band at all.
  const narrative = resolved && schedule
    ? [
      ...fluidCellTraceNarrative(resolved, schedule),
      ...(fineBand && resolved.fineFactor >= 1 ? fineBandNarrative(resolved, fineBand) : []),
    ]
    : [];
  const figures = resolved && schedule ? fluidCellTraceKeyFigures(resolved, schedule) : [];
  const span = resolved ? fineBandPhiSpan(resolved) : undefined;
  const crossings = resolved ? fineBandFreeSurfaceCrossings(resolved) : [];
  const census = resolved ? fineBandCensus(resolved) : undefined;
  const status = trace ? STATUS_LABEL[trace.status] ?? "unknown" : blocked?.headline ?? "waiting for the gather";
  const failed = trace?.status === FLUID_CELL_TRACE_STATUS.invalid
    || trace?.status === FLUID_CELL_TRACE_STATUS.unavailable
    || probeStatus === "unavailable";

  const counts = new Map<FluidCellTraceLayer, number>();
  // Seeded so a stencil layer with nothing in it reports zero rather than going
  // uncounted: the collapsed legend hides a measured zero — there is no mark on
  // screen to explain — and keeps a layer whose count was never published.
  if (resolved) { counts.set("stencil", 0); counts.set("transition", 0); }
  for (const neighbor of resolved?.neighbors ?? []) {
    if (!neighbor.present) continue;
    const layer = fluidCellTraceLayerForNeighbor(neighbor);
    counts.set(layer, (counts.get(layer) ?? 0) + 1);
  }
  if (resolved) counts.set("cell", 1);
  if (resolved && resolved.fineSamples > 0) counts.set("fine", resolved.fineSamples);
  // Each fine layer counts what it actually draws, not what the leaf holds, so
  // a chip reading zero means "nothing to see" rather than "nothing is there".
  if (resolved) {
    const records = resolved.fineProbeRecords;
    counts.set("surface", crossings.length);
    counts.set("patch", records.filter((record) => record.valid && record.interface).length);
    counts.set("links", records.filter((record) => record.valid && record.resolved && record.hop > 0).length);
    counts.set("gaps", records.filter((record) => record.missing || record.stale).length);
  }

  const transitions = (resolved?.neighbors ?? []).filter(
    (neighbor) => neighbor.present && (neighbor.coarser || neighbor.finer));

  // Shown whenever the ray met anything, including a miss on the selected step,
  // because that is exactly when the reader needs to know a step exists.
  const depth = trace && trace.hits.length > 0
    ? {
      index: trace.hitIndex,
      total: trace.hits.length,
      overflow: trace.hitOverflow,
      // A pixel names the nearest leaf, which on a liquid is a surface cell,
      // and stepping inward one at a time crosses many interior cells before
      // reaching anything interesting. This is the shortcut past them.
      hasInterface: fluidCellTraceHasInterfaceHit(trace.hits),
    }
    : undefined;

  return (
    <div
      className="pixel-trace-hud"
      data-testid="fluid-cell-trace-hud"
      data-trace="fluid-cell"
      data-pinned={pinned ? "true" : "false"}
      data-probe-status={probeStatus}
    >
      <header>
        <div>
          <small>Cell work · {pinned ? "pinned" : "live"}</small>
          <h3>{resolved ? `row ${resolved.row} · leaf ${resolved.leafSize}³` : blocked?.headline ?? "no cell"}</h3>
        </div>
        <div className="pixel-trace-actions">
          {/* The pointer alone can only ever name the nearest leaf, which on a
              liquid is a surface cell. This is how an interior unknown gets
              selected, so it sits beside the pin rather than in the layer strip. */}
          {depth && <span className="fluid-cell-depth" data-testid="fluid-cell-depth">
            <button
              type="button"
              onClick={() => onStepHit(-1)}
              disabled={depth.total < 2}
              title="Step back out along the ray ( [ )"
            >[</button>
            <b>{depth.index + 1}/{depth.total}{depth.overflow > 0 ? "+" : ""}</b>
            <button
              type="button"
              onClick={() => onStepHit(1)}
              disabled={depth.total < 2}
              title="Step further in along the ray ( ] )"
            >]</button>
            <button
              type="button"
              className="fluid-cell-jump"
              onClick={onJumpToInterface}
              disabled={!depth.hasInterface}
              data-testid="fluid-cell-interface-jump"
              title={depth.hasInterface
                ? "Jump to the next leaf the surface passes through — the outlined ones on the ray"
                : "No leaf along this ray holds the interface"}
            >≈</button>
          </span>}
          <button
            type="button"
            aria-pressed={pinned}
            onClick={onTogglePinned}
            title={pinned ? "Follow the pointer again" : "Freeze this cell and orbit around it — clicking the viewport does the same"}
          >
            {pinned ? "Unpin" : "Pin cell"}
          </button>
          <button type="button" onClick={onClose} title="Close the cell-work diagnostic">Close</button>
        </div>
      </header>

      {/* The first page: four figures that answer "is this cell interesting",
          not an account that assumes the answer is yes. Each is badged with
          whether the frame published it or the command graph encodes it, which
          is the one distinction this diagnostic cannot afford to blur. */}
      {resolved ? (
        <dl className="fluid-cell-figures" data-testid="fluid-cell-figures">
          {figures.map((figure) => (
            <div key={figure.id} data-evidence={figure.evidence} data-alert={figure.alert ? "true" : "false"}>
              <dt>
                {figure.layer && (
                  <i style={{ background: FLUID_CELL_TRACE_LAYER_DEFINITIONS[figure.layer].swatch }} />
                )}
                {figure.label}
              </dt>
              <dd title={figure.detail}>{figure.value}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="pixel-trace-blocked" data-failed={failed ? "true" : "false"}>
          {blocked?.detail ?? STATUS_LABEL[trace?.status ?? FLUID_CELL_TRACE_STATUS.pending]}
        </p>
      )}

      {/* The legend is the switchboard: what each mark on screen means, and
          what to switch off. Every layer stays listed whether it is on or has
          anything to draw, because the row is also the switch — and always
          listed means it never moves under the pointer either. */}
      <VisualizationLegend
        definitions={LAYER_LEGEND}
        enabled={enabledLayers}
        counts={counts}
        expanded={expanded}
        onToggle={(group) => onToggleLayer(group as FluidCellTraceLayer)}
      />

      <button
        type="button"
        className="pixel-trace-disclosure"
        aria-expanded={expanded}
        onClick={onToggleExpanded}
        data-testid="fluid-cell-disclosure"
      >
        {expanded ? "Less" : "Learn more"}
        <em data-failed={failed ? "true" : "false"}>{status}</em>
      </button>

      {expanded && <>
        {/* Before the narrative: what this cell *is*, in the fluid. Every step
            below is a detail of it, and the sub-grid warning has to be the first
            thing a reader meets rather than the last. */}
        {resolved && span && <PhiSpanBar span={span} />}

        {narrative.length > 0 && <ol className="pixel-trace-steps">
          {narrative.map((step) => (
            <li key={step.id} data-layer={step.layer ?? "none"} data-evidence={step.evidence}>
              <i style={step.layer ? { background: FLUID_CELL_TRACE_LAYER_DEFINITIONS[step.layer].swatch } : undefined} />
              <b>{step.label}</b>
              <output>{step.value}</output>
              <small>{step.detail}</small>
            </li>
          ))}
        </ol>}

        {/* The operator's boundary story, in the same per-direction shape the
            resolution strip already uses. Sorted tightest first by the model, so
            the face that explains a large diagonal is the one you read first. */}
        {crossings.length > 0 && (
          <div className="pixel-trace-mips fine-band-crossings" data-testid="fine-band-crossings">
            <small>
              Free surface — θ is where the surface cuts each dual edge; the ghost-fluid condition
              scales that face&rsquo;s coefficient by 1/θ
            </small>
            <div>
              {crossings.map((crossing) => {
                const swatch = crossing.outward
                  ? (crossing.fraction < 0.25 ? "#ff1738" : "#f5ba1a")
                  : "#8a7ad6";
                return (
                  <span
                    key={crossing.direction}
                    style={{ borderColor: swatch }}
                    title={crossing.outward
                      ? `phi ${crossing.ownPhi.toFixed(3)} here, ${crossing.neighborPhi.toFixed(3)} across`
                      : "the neighbour is the liquid side, so this condition belongs to its row"}
                  >
                    <i style={{ background: swatch }} />
                    <b>{crossing.label}</b>
                    <output>θ {(crossing.fraction * 100).toFixed(0)}%</output>
                    <em>
                      {crossing.coefficientScale === undefined
                        ? "inward"
                        : `×${crossing.coefficientScale.toFixed(1)}`}
                    </em>
                  </span>
                );
              })}
            </div>
          </div>
        )}

        {transitions.length > 0 && (
          <div className="pixel-trace-mips" data-testid="fluid-cell-trace-transitions">
            <small>Resolution transitions — one hop crosses a whole neighbour leaf, so reach is uneven</small>
            <div>
              {transitions.map((neighbor) => (
                <span key={neighbor.direction} style={{ borderColor: neighbor.coarser ? "#fa9e14" : "#0fc7cc" }}>
                  <i style={{ background: neighbor.coarser ? "#fa9e14" : "#0fc7cc" }} />
                  <b>{fluidCellTraceDirectionLabel(neighbor.direction)}</b>
                  <output>{neighbor.leafSize}³</output>
                  <em>{neighbor.coarser ? "coarser" : "finer"}</em>
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Context for the band, two-up because neither is worth a row of its
            own: where this cell sits in the nest, and which flood pass paid for
            it. */}
        {resolved && fineBand && (
          <div className="fine-band-context">
            <BandNest
              distance={span?.published ? span.centre : resolved.probeNearestPhi}
              context={fineBand}
              fineFactor={resolved.fineFactor}
            />
            <FloodLadder trace={resolved} context={fineBand} />
          </div>
        )}

        {resolved && census && census.probes > 0 && (
          <p className="pixel-trace-note">
            Fine-band counts are over {census.probes.toLocaleString()} probes spread through the leaf,
            not a census — a 32³ leaf at factor four holds over two million samples. The line work
            draws the {census.recorded.toLocaleString()}-probe sub-lattice of those, so the picture is
            a strict subset of what the counts measured.
          </p>
        )}
        <p className="pixel-trace-note pixel-trace-footnote">
          Gathered lines are what this frame published about the cell. Scheduled lines are what the encoded
          command graph does to a row, which a residual gate may cut short — the two are never added together.
          {pinned
            ? " Pinned: orbit to inspect the frozen cell, or click the viewport to follow the pointer again."
            : " Click the viewport to freeze the cell under the pointer."}
        </p>
      </>}
    </div>
  );
}
