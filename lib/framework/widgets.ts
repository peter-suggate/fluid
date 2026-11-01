/**
 * Control widgets: live handles on whatever in a scene can be driven.
 *
 * A provider looks at its domain (rigid motors, interaction entities, …) and
 * says which widgets exist right now and what each one's channels read; the
 * host draws them and hands input back to the provider that declared them.
 * Where a widget stands is data in three layers — the provider's default, the
 * scene's authored placement, the user's own arrangement — resolved here, so
 * no host or provider ever decides another's layout.
 */

export const WIDGET_ANCHORS = [
  "top-left", "top", "top-right",
  "left", "center", "right",
  "bottom-left", "bottom", "bottom-right",
] as const;
export type WidgetAnchor = typeof WIDGET_ANCHORS[number];

export interface WidgetPlacement {
  readonly anchor: WidgetAnchor;
  /**
   * From the anchor to the widget box. At an edge it is the gap between that
   * edge and the box, positive inward; on a centre line it is the signed
   * displacement of the box centre (+x right, +y down). A placed widget stands
   * alone; those without an offset stack at their anchor.
   */
  readonly offset_px?: { readonly x: number; readonly y: number };
  /** Within an anchor, lower first; ties keep declaration order. */
  readonly order?: number;
  readonly hidden?: boolean;
}

/**
 * A lever. `rest` is where it springs back to on release; without one it
 * latches. A range spanning zero is two-way: forward above the line, reverse
 * below it, with a detent at zero.
 */
export interface WidgetAxis1 {
  readonly kind: "axis1";
  readonly id: string;
  readonly label: string;
  readonly min: number;
  readonly max: number;
  readonly step?: number;
  readonly value: number;
  readonly unit?: string;
  readonly rest?: number;
  /** What zero reads as on a two-way lever, e.g. "Stop". */
  readonly zeroLabel?: string;
}
/** A stick in [-1, 1]², screen-oriented (+y down), springing back to the centre. */
export interface WidgetAxis2 {
  readonly kind: "axis2";
  readonly id: string;
  readonly label: string;
  readonly value: readonly [number, number];
}
/** Held while pressed. */
export interface WidgetButton {
  readonly kind: "button";
  readonly id: string;
  readonly label: string;
  readonly pressed: boolean;
  /** The key that does the same, for the label. */
  readonly shortcut?: string;
}
/** One shot, e.g. take or hand back control. */
export interface WidgetAction {
  readonly kind: "action";
  readonly id: string;
  readonly label: string;
  readonly hint?: string;
  readonly tone?: "default" | "accent";
  readonly disabled?: boolean;
}
export type WidgetChannel = WidgetAxis1 | WidgetAxis2 | WidgetButton | WidgetAction;
export type WidgetValue = number | readonly [number, number] | boolean;

export interface ControlWidget {
  /** Stable across frames and sessions: overrides are keyed by it. */
  readonly id: string;
  /** What it drives, e.g. "body:paddleboat-0". */
  readonly target: string;
  readonly label: string;
  /** CSS colour tying the widget to what it drives on screen. */
  readonly accent?: string;
  /** Renderer id; absent draws each channel with the generic renderer. */
  readonly view?: string;
  readonly channels: readonly WidgetChannel[];
  readonly placement: WidgetPlacement;
}

export interface WidgetProvider<C> {
  readonly id: string;
  widgets(context: C): readonly ControlWidget[];
  /** `true` presses and `false` releases a button; an action receives `true`. */
  input(context: C, widgetId: string, channelId: string, value: WidgetValue): void;
}

export type WidgetPlacementOverrides = Readonly<Record<string, Partial<WidgetPlacement>>>;

export interface ResolvedWidget extends ControlWidget {
  readonly provider: string;
  readonly placement: WidgetPlacement;
}
export interface WidgetGroup {
  readonly anchor: WidgetAnchor;
  /** Present for a single widget the user (or scene) placed exactly. */
  readonly offset_px?: { readonly x: number; readonly y: number };
  readonly widgets: readonly ResolvedWidget[];
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function placementCheck(where: string, placement: Partial<WidgetPlacement>) {
  if (placement.anchor !== undefined && !WIDGET_ANCHORS.includes(placement.anchor)) throw new Error(`Unknown anchor ${where}: ${String(placement.anchor)}`);
  if (placement.offset_px !== undefined && !(finite(placement.offset_px.x) && finite(placement.offset_px.y))) throw new Error(`Non-finite offset ${where}`);
  if (placement.order !== undefined && !finite(placement.order)) throw new Error(`Non-finite order ${where}`);
}

function channelCheck(widget: string, channel: WidgetChannel) {
  const where = `${widget}/${channel.id}`;
  if (!channel.id || !channel.label.trim()) throw new Error(`Incomplete channel ${where}`);
  if (channel.kind === "axis1") {
    if (![channel.min, channel.max, channel.value].every(finite) || channel.min >= channel.max) throw new Error(`Invalid range ${where}`);
    if (channel.step !== undefined && !(finite(channel.step) && channel.step > 0)) throw new Error(`Invalid step ${where}`);
    if (channel.rest !== undefined && !(finite(channel.rest) && channel.rest >= channel.min && channel.rest <= channel.max)) throw new Error(`Rest outside range ${where}`);
  } else if (channel.kind === "axis2") {
    if (channel.value.length !== 2 || !channel.value.every(finite)) throw new Error(`Invalid stick value ${where}`);
  } else if (channel.kind !== "button" && channel.kind !== "action") {
    throw new Error(`Unknown channel kind ${where}`);
  }
}

/** Validate every provider's widgets and gather them; throws on bad data rather than dropping it. */
export function collectWidgets<C>(providers: readonly WidgetProvider<C>[], context: C): readonly ResolvedWidget[] {
  const ids = new Set<string>();
  const out: ResolvedWidget[] = [];
  for (const provider of providers) {
    for (const widget of provider.widgets(context)) {
      if (!widget.id || ids.has(widget.id)) throw new Error(`Duplicate or empty widget ID: ${widget.id} (${provider.id})`);
      ids.add(widget.id);
      if (!widget.target || !widget.label.trim()) throw new Error(`Incomplete widget ${widget.id}`);
      const channels = new Set<string>();
      for (const channel of widget.channels) {
        if (channels.has(channel.id)) throw new Error(`Duplicate channel ${widget.id}/${channel.id}`);
        channels.add(channel.id);
        channelCheck(widget.id, channel);
      }
      placementCheck(widget.id, widget.placement);
      if (widget.placement.anchor === undefined) throw new Error(`Widget ${widget.id} needs an anchor`);
      out.push({ ...widget, provider: provider.id });
    }
  }
  return out;
}

/**
 * Layer the overrides (scene, then user, …) over each widget's default and
 * group by anchor in `WIDGET_ANCHORS` order. A layer's `offset_px` replaces
 * the one below rather than adding to it: offsets are measured from the
 * anchor the same layer may have moved the widget to.
 */
export function resolveWidgetLayout(widgets: readonly ResolvedWidget[], ...layers: readonly (WidgetPlacementOverrides | undefined)[]): readonly WidgetGroup[] {
  for (const layer of layers) for (const [id, placement] of Object.entries(layer ?? {})) placementCheck(id, placement);
  const placed = widgets.map((widget, index) => {
    let placement: WidgetPlacement = widget.placement;
    for (const layer of layers) {
      const override = layer?.[widget.id];
      if (!override) continue;
      placement = { ...placement, ...override, offset_px: "offset_px" in override ? override.offset_px : override.anchor ? undefined : placement.offset_px };
    }
    return { widget: { ...widget, placement }, index };
  }).filter(({ widget }) => !widget.placement.hidden);
  return WIDGET_ANCHORS.flatMap((anchor): WidgetGroup[] => {
    const here = placed.filter(({ widget }) => widget.placement.anchor === anchor)
      .sort((a, b) => (a.widget.placement.order ?? 0) - (b.widget.placement.order ?? 0) || a.index - b.index)
      .map(({ widget }) => widget);
    const stacked = here.filter(widget => !widget.placement.offset_px);
    return [
      ...(stacked.length ? [{ anchor, widgets: stacked }] : []),
      ...here.filter(widget => widget.placement.offset_px).map(widget => ({ anchor, offset_px: widget.placement.offset_px, widgets: [widget] })),
    ];
  });
}

/**
 * Where a widget dropped at `point` (pane-relative px) lands: the nearest of
 * the nine regions and the offset from that anchor, so it keeps its place when
 * the pane is resized.
 */
export function placementAtPoint(point: { readonly x: number; readonly y: number }, size: { readonly width: number; readonly height: number },
  box: { readonly width: number; readonly height: number }): Pick<WidgetPlacement, "anchor" | "offset_px"> {
  const column = point.x < size.width / 3 ? 0 : point.x < 2 * size.width / 3 ? 1 : 2;
  const row = point.y < size.height / 3 ? 0 : point.y < 2 * size.height / 3 ? 1 : 2;
  const anchor = WIDGET_ANCHORS[row * 3 + column]!;
  // Measured from the edge (or centre line) the anchor names to the box edge facing it.
  const left = point.x - box.width / 2, top = point.y - box.height / 2;
  const x = column === 0 ? left : column === 2 ? size.width - (left + box.width) : left + box.width / 2 - size.width / 2;
  const y = row === 0 ? top : row === 2 ? size.height - (top + box.height) : top + box.height / 2 - size.height / 2;
  return { anchor, offset_px: { x: Math.round(x), y: Math.round(y) } };
}
