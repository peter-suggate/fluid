"use client";

import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type ComponentType, type CSSProperties, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { useStore } from "zustand";
import { Button } from "../../../components/ui";
import { findSceneDefinition } from "../../core/scenes";
import type { PaneId } from "../../core/session/session";
import { useSession } from "../../core/session/session-context";
import {
  collectWidgets, placementAtPoint, resolveWidgetLayout,
  type ControlWidget, type ResolvedWidget, type WidgetAnchor, type WidgetAxis1, type WidgetAxis2, type WidgetButton,
  type WidgetChannel, type WidgetGroup, type WidgetProvider, type WidgetValue,
} from "../../framework/widgets";
import { useInteractionPane } from "../interactive/ui";
import type { InteractionPane } from "../interactive/pane";
import type { WidgetContext } from "./context";
import { widgetLayoutStore } from "./layout-store";

/**
 * Control widgets over the pane: one card per thing that can be driven, from
 * whichever providers the application installed, at the place the scene or the
 * user put it.
 *
 * The layer itself never learns what a widget drives. It renders each card
 * with the view the widget names (or channel by channel), hands every input
 * back to the provider that declared it, and owns only the arrangement: drag a
 * card by its grip, double-click the grip to put it back, right-click for the
 * layout menu.
 */

export interface WidgetViewProps {
  readonly widget: ResolvedWidget;
  readonly onInput: (channelId: string, value: WidgetValue) => void;
}
export type WidgetViews = Readonly<Record<string, ComponentType<WidgetViewProps>>>;

const NO_INTERACTION = { subscribe: () => () => {}, get: () => undefined };

/** The interaction parts a provider reads, re-published only when one of them changes: the view moves every tick while running. */
function useInteractionContext(pane: InteractionPane | undefined): WidgetContext["interaction"] {
  const cache = useRef<WidgetContext["interaction"]>(undefined);
  const source = pane ? { subscribe: (listener: () => void) => pane.store.subscribe(listener), get: () => pane.state } : NO_INTERACTION;
  const read = () => {
    const state = source.get();
    if (!pane || !state) return undefined;
    const previous = cache.current;
    if (previous && previous.pane === pane && previous.busy === state.busy && previous.view.document === state.view.document
      && previous.view.controlCandidates === state.view.controlCandidates && previous.view.controlledEntityId === state.view.controlledEntityId) return previous;
    cache.current = { pane, view: state.view, busy: state.busy };
    return cache.current;
  };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const subscribe = useMemo(() => source.subscribe, [pane]);
  return useSyncExternalStore(subscribe, read, () => undefined);
}

export function ControlWidgetLayer({ paneId, providers, views }: {
  readonly paneId: PaneId;
  readonly providers: readonly WidgetProvider<WidgetContext>[];
  readonly views: WidgetViews;
}) {
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  const interaction = useInteractionContext(useInteractionPane(paneId));
  const layoutStore = widgetLayoutStore(paneId);
  const layouts = useStore(layoutStore, (state) => state.layouts);
  const layer = useRef<HTMLDivElement>(null);
  const [menu, setMenu] = useState<{ widgetId: string; x: number; y: number }>();

  const context = useMemo<WidgetContext>(() => ({ paneId, scene, interaction }), [paneId, scene, interaction]);
  const widgets = useMemo(() => collectWidgets(providers, context), [providers, context]);
  const sceneLayout = findSceneDefinition(scene.sceneId)?.widgets;
  const userLayout = layouts[scene.sceneId];
  const groups = useMemo(() => {
    try { return resolveWidgetLayout(widgets, sceneLayout, userLayout); }
    // Storage is the one layer nobody validated on the way in: drop it, keep the scene's.
    catch { return resolveWidgetLayout(widgets, sceneLayout); }
  }, [widgets, sceneLayout, userLayout]);

  useEffect(() => {
    if (!menu) return;
    const close = (event: PointerEvent) => { if (!(event.target instanceof Element && event.target.closest(".control-widget-menu"))) setMenu(undefined); };
    window.addEventListener("pointerdown", close, true);
    return () => window.removeEventListener("pointerdown", close, true);
  }, [menu]);

  if (groups.length === 0) return null;
  const input = (widget: ResolvedWidget) => (channelId: string, value: WidgetValue) =>
    providers.find((provider) => provider.id === widget.provider)?.input(context, widget.id, channelId, value);
  const drop = (widget: ControlWidget, centre: { x: number; y: number }, box: { width: number; height: number }) => {
    const bounds = layer.current?.getBoundingClientRect();
    if (!bounds) return;
    layoutStore.getState().place(scene.sceneId, widget.id, placementAtPoint(
      { x: centre.x - bounds.left, y: centre.y - bounds.top }, { width: bounds.width, height: bounds.height }, box));
  };

  return <div ref={layer} className="control-widget-layer" data-testid="control-widget-layer">
    {groups.map((group) => <div key={`${group.anchor}:${group.widgets[0]!.id}:${group.offset_px ? "placed" : "stack"}`}
      className="control-widget-group" data-anchor={group.anchor} style={groupStyle(group)}>
      {group.widgets.map((widget) => {
        const View = (widget.view && views[widget.view]) || ChannelsView;
        return <WidgetFrame key={widget.id} widget={widget} onDrop={drop}
          onReset={() => layoutStore.getState().reset(scene.sceneId, widget.id)}
          onMenu={(x, y) => setMenu({ widgetId: widget.id, x, y })}>
          <View widget={widget} onInput={input(widget)} />
        </WidgetFrame>;
      })}
    </div>)}
    {menu && <div className="control-widget-menu" role="menu" style={{ left: menu.x - (layer.current?.getBoundingClientRect().left ?? 0), top: menu.y - (layer.current?.getBoundingClientRect().top ?? 0) }}>
      <button type="button" role="menuitem" onClick={() => { layoutStore.getState().reset(scene.sceneId, menu.widgetId); setMenu(undefined); }}>Reset this widget</button>
      <button type="button" role="menuitem" onClick={() => { layoutStore.getState().reset(scene.sceneId); setMenu(undefined); }}>Reset widget layout</button>
    </div>}
  </div>;
}

const COLUMN: Readonly<Record<WidgetAnchor, 0 | 1 | 2>> = {
  "top-left": 0, top: 1, "top-right": 2, left: 0, center: 1, right: 2, "bottom-left": 0, bottom: 1, "bottom-right": 2,
};
const ROW: Readonly<Record<WidgetAnchor, 0 | 1 | 2>> = {
  "top-left": 0, top: 0, "top-right": 0, left: 1, center: 1, right: 1, "bottom-left": 2, bottom: 2, "bottom-right": 2,
};

/** A stacked group takes the stylesheet's insets; a placed one stands exactly where it was dropped. */
function groupStyle(group: WidgetGroup): CSSProperties | undefined {
  const offset = group.offset_px;
  if (!offset) return undefined;
  const column = COLUMN[group.anchor], row = ROW[group.anchor];
  return {
    left: column === 0 ? offset.x : column === 1 ? `calc(50% + ${offset.x}px)` : "auto",
    right: column === 2 ? offset.x : "auto",
    top: row === 0 ? offset.y : row === 1 ? `calc(50% + ${offset.y}px)` : "auto",
    bottom: row === 2 ? offset.y : "auto",
    transform: `translate(${column === 1 ? "-50%" : "0"}, ${row === 1 ? "-50%" : "0"})`,
  };
}

function WidgetFrame({ widget, onDrop, onReset, onMenu, children }: {
  readonly widget: ResolvedWidget;
  readonly onDrop: (widget: ControlWidget, centre: { x: number; y: number }, box: { width: number; height: number }) => void;
  readonly onReset: () => void;
  readonly onMenu: (x: number, y: number) => void;
  readonly children: ReactNode;
}) {
  const card = useRef<HTMLElement>(null);
  const drag = useRef<{ pointerId: number; x: number; y: number; box: DOMRect }>(undefined);
  const [delta, setDelta] = useState<{ x: number; y: number }>();

  const down = (event: ReactPointerEvent) => {
    if (event.button !== 0) return;
    const box = card.current?.getBoundingClientRect();
    if (!box) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, box };
  };
  const move = (event: ReactPointerEvent) => {
    const held = drag.current;
    if (held?.pointerId === event.pointerId) setDelta({ x: event.clientX - held.x, y: event.clientY - held.y });
  };
  const up = (event: ReactPointerEvent) => {
    const held = drag.current;
    if (held?.pointerId !== event.pointerId) return;
    drag.current = undefined;
    const { box } = held, moved = { x: event.clientX - held.x, y: event.clientY - held.y };
    setDelta(undefined);
    // From the box as it was pressed: the last move may lag the release.
    if (Math.hypot(moved.x, moved.y) > 3) {
      onDrop(widget, { x: box.left + box.width / 2 + moved.x, y: box.top + box.height / 2 + moved.y }, { width: box.width, height: box.height });
    }
  };

  // The canvas is a sibling, so presses here never reach it; what is stopped is
  // the window's listeners (editor keys, outside-click dismissals) behind React's root.
  const contain = (event: { stopPropagation(): void }) => event.stopPropagation();
  return <section ref={card} className="control-widget" onPointerDown={contain} onKeyDown={contain} onKeyUp={contain} data-widget={widget.id} data-dragging={delta ? "" : undefined}
    aria-label={`${widget.label} controls`} data-testid={`control-widget-${widget.id}`}
    style={{ ...(widget.accent ? { "--widget-accent": widget.accent } : {}), ...(delta ? { translate: `${delta.x}px ${delta.y}px` } : {}) } as CSSProperties}>
    <header className="control-widget-head">
      <span className="control-widget-grip" role="button" tabIndex={-1} aria-label={`Move ${widget.label}`}
        title="Drag to move · double-click to put back · right-click for layout"
        onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={() => { drag.current = undefined; setDelta(undefined); }}
        onDoubleClick={onReset} onContextMenu={(event) => { event.preventDefault(); onMenu(event.clientX, event.clientY); }}>
        <svg width="8" height="12" viewBox="0 0 8 12" aria-hidden><g fill="currentColor">
          <circle cx="2" cy="2" r="1" /><circle cx="6" cy="2" r="1" /><circle cx="2" cy="6" r="1" />
          <circle cx="6" cy="6" r="1" /><circle cx="2" cy="10" r="1" /><circle cx="6" cy="10" r="1" /></g></svg>
      </span>
      {widget.accent && <span className="control-widget-marker" aria-hidden />}
      <strong>{widget.label}</strong>
    </header>
    {children}
  </section>;
}

/** The fallback view: every channel with its generic control, axes beside buttons. */
export function ChannelsView({ widget, onInput }: WidgetViewProps) {
  const axes = widget.channels.filter((channel) => channel.kind === "axis1" || channel.kind === "axis2");
  const buttons = widget.channels.filter((channel) => channel.kind === "button" || channel.kind === "action");
  return <div className="control-widget-body">
    {axes.map((channel) => <Channel key={channel.id} channel={channel} label={widget.label} onInput={(value) => onInput(channel.id, value)} />)}
    {buttons.length > 0 && <div className="control-widget-buttons">
      {buttons.map((channel) => <Channel key={channel.id} channel={channel} label={widget.label} onInput={(value) => onInput(channel.id, value)} />)}
    </div>}
  </div>;
}

function Channel({ channel, label, onInput }: { readonly channel: WidgetChannel; readonly label: string; readonly onInput: (value: WidgetValue) => void }) {
  if (channel.kind === "axis1") return <Lever channel={channel} label={label} onInput={onInput} />;
  if (channel.kind === "axis2") return <Stick channel={channel} label={label} onInput={onInput} />;
  if (channel.kind === "button") return <HoldButton channel={channel} onInput={onInput} />;
  return <Button tone={channel.tone} hint={channel.hint} disabled={channel.disabled} className="control-widget-action"
    testId={`control-widget-action-${channel.id}`} onClick={() => onInput(true)}>{channel.label}</Button>;
}

/**
 * A held control's one release: the pointer coming up, a cancel, the window
 * losing focus or the control unmounting mid-press, whichever comes first. The
 * rest is sent once per press however many of those arrive.
 */
function useHold(release: () => void): { readonly grab: () => void; readonly letGo: () => void } {
  const latest = useRef(release);
  latest.current = release;
  const [hold] = useState(() => {
    let holding = false;
    const letGo = () => {
      if (!holding) return;
      holding = false;
      window.removeEventListener("blur", letGo);
      latest.current();
    };
    const grab = () => {
      if (holding) return;
      holding = true;
      window.addEventListener("blur", letGo);
    };
    return { grab, letGo };
  });
  useEffect(() => hold.letGo, [hold]);
  return hold;
}

const LEVER_TRAVEL_PX = 96;

function Lever({ channel, label, onInput }: { readonly channel: WidgetAxis1; readonly label: string; readonly onInput: (value: number) => void }) {
  const track = useRef<HTMLDivElement>(null);
  const [held, setHeld] = useState<number>();
  const value = held ?? channel.value;
  const step = channel.step ?? (channel.max - channel.min) / 100;
  const snap = (raw: number) => {
    const stepped = channel.min + Math.round((raw - channel.min) / step) * step;
    return Math.max(channel.min, Math.min(channel.max, Number(stepped.toFixed(6))));
  };
  const set = (next: number) => {
    const snapped = snap(next);
    if (snapped !== value) onInput(snapped);
    return snapped;
  };
  const fromPointer = (clientY: number) => {
    const box = track.current?.getBoundingClientRect();
    if (!box) return value;
    const fraction = 1 - Math.max(0, Math.min(1, (clientY - box.top) / box.height));
    return channel.min + fraction * (channel.max - channel.min);
  };
  const { grab, letGo } = useHold(() => {
    setHeld(undefined);
    if (channel.rest !== undefined) onInput(channel.rest);
  });
  // A range across zero is one two-way lever: the fill grows from the zero line,
  // up forward and down in reverse, and a drag catches on zero within a step.
  const twoWay = channel.min < 0 && channel.max > 0;
  const detent = (raw: number) => twoWay && Math.abs(raw) <= step ? 0 : raw;
  const at = (v: number) => (v - channel.min) / (channel.max - channel.min);
  const fraction = at(value), zero = twoWay ? at(0) : 0;
  const unit = channel.unit ? ` ${channel.unit}` : "";
  const magnitude = Number(Math.abs(value).toFixed(2));
  const shown = !twoWay ? `${Number(value.toFixed(2))}${unit}`
    : value === 0 ? channel.zeroLabel ?? `0${unit}` : `${value > 0 ? "+" : "\u2212"}${magnitude}${unit}`;
  return <div className="control-widget-lever">
    <div ref={track} className="control-widget-lever-track" role="slider" tabIndex={0}
      aria-label={`${label} ${channel.label}`} aria-valuemin={channel.min} aria-valuemax={channel.max} aria-valuenow={value}
      aria-valuetext={shown} data-testid={`control-widget-lever-${channel.id}`} style={{ height: LEVER_TRAVEL_PX }}
      onPointerDown={(event) => { if (event.button !== 0) return; event.currentTarget.setPointerCapture(event.pointerId); event.currentTarget.focus(); grab(); setHeld(set(detent(fromPointer(event.clientY)))); }}
      onPointerMove={(event) => { if (held !== undefined) setHeld(set(detent(fromPointer(event.clientY)))); }}
      onPointerUp={letGo} onPointerCancel={letGo} onLostPointerCapture={letGo}
      onWheel={(event) => { if (channel.rest === undefined) set(value + (event.deltaY < 0 ? step : -step)); }}
      onKeyDown={(event) => {
        const next = event.key === "ArrowUp" || event.key === "ArrowRight" ? value + step
          : event.key === "ArrowDown" || event.key === "ArrowLeft" ? value - step
          : event.key === "PageUp" ? value + 10 * step : event.key === "PageDown" ? value - 10 * step
          : event.key === "Home" ? channel.min : event.key === "End" ? channel.max : undefined;
        if (next === undefined) return;
        event.preventDefault();
        set(next);
      }}
      onKeyUp={() => { if (channel.rest !== undefined && value !== channel.rest) onInput(channel.rest); }}>
      {twoWay && <div className="control-widget-lever-zero" style={{ bottom: `${zero * 100}%` }} />}
      <div className="control-widget-lever-fill" style={{ bottom: `${Math.min(zero, fraction) * 100}%`, height: `${Math.abs(fraction - zero) * 100}%` }} />
      <div className="control-widget-lever-knob" style={{ bottom: `calc(${fraction * 100}% - 3px)` }} />
    </div>
    <span className="ui-value">{shown}</span>
    <span className="control-widget-caption">{channel.label}</span>
  </div>;
}

const STICK_RADIUS_PX = 38;

function Stick({ channel, label, onInput }: { readonly channel: WidgetAxis2; readonly label: string; readonly onInput: (value: readonly [number, number]) => void }) {
  const pad = useRef<HTMLDivElement>(null);
  const [held, setHeld] = useState<readonly [number, number]>();
  const value = held ?? channel.value;
  const fromPointer = (clientX: number, clientY: number): readonly [number, number] => {
    const box = pad.current?.getBoundingClientRect();
    if (!box) return [0, 0];
    let x = (clientX - (box.left + box.width / 2)) / STICK_RADIUS_PX, y = (clientY - (box.top + box.height / 2)) / STICK_RADIUS_PX;
    const length = Math.hypot(x, y);
    if (length > 1) { x /= length; y /= length; }
    return [Number(x.toFixed(3)), Number(y.toFixed(3))];
  };
  const set = (next: readonly [number, number]) => { setHeld(next); onInput(next); };
  const { grab, letGo } = useHold(() => { setHeld(undefined); onInput([0, 0]); });
  return <div className="control-widget-stick-wrap">
    <div ref={pad} className="control-widget-stick" role="application" aria-label={`${label} ${channel.label}`}
      data-testid={`control-widget-stick-${channel.id}`} data-held={held ? "" : undefined}
      style={{ width: 2 * STICK_RADIUS_PX + 16, height: 2 * STICK_RADIUS_PX + 16 }}
      onPointerDown={(event) => { if (event.button !== 0) return; event.currentTarget.setPointerCapture(event.pointerId); grab(); set(fromPointer(event.clientX, event.clientY)); }}
      onPointerMove={(event) => { if (held) set(fromPointer(event.clientX, event.clientY)); }}
      onPointerUp={letGo} onPointerCancel={letGo} onLostPointerCapture={letGo}>
      <div className="control-widget-stick-knob" style={{ translate: `${value[0] * STICK_RADIUS_PX}px ${value[1] * STICK_RADIUS_PX}px` }} />
    </div>
    <span className="control-widget-caption">{channel.label}</span>
  </div>;
}

function HoldButton({ channel, onInput }: { readonly channel: WidgetButton; readonly onInput: (value: boolean) => void }) {
  const [held, setHeld] = useState(false);
  const pressed = held || channel.pressed;
  const { grab, letGo } = useHold(() => { setHeld(false); onInput(false); });
  return <button type="button" className="ui-button control-widget-hold" aria-pressed={pressed} data-testid={`control-widget-button-${channel.id}`}
    onPointerDown={(event) => { if (event.button !== 0) return; event.currentTarget.setPointerCapture(event.pointerId); grab(); setHeld(true); onInput(true); }}
    onPointerUp={letGo} onPointerCancel={letGo} onLostPointerCapture={letGo}>
    {channel.label}{channel.shortcut && <kbd>{channel.shortcut}</kbd>}
  </button>;
}
