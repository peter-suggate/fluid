import { createStore, type StoreApi } from "zustand/vanilla";
import type { Vec3 } from "../../core/model";
import type {
  BehaviorRecord,
  CommandResult,
  Connection,
  DocumentInspection,
  EntityRecord,
  InteractionCatalog,
  InteractionCommand,
  InteractionHost,
  InteractionView,
  JsonObject,
  Transform,
} from "../../interaction";
import { placementTransform, snapSurfaceHit, type SurfaceHit } from "./geometry";
import { ViewportInputRouter, type UnsequencedSample } from "./input-router";
import { createDemoInteractionCatalog, DEMO_HOST_CAPABILITIES } from "./demo/catalog";
import { DEMO_RUNTIME_FACTORIES } from "./demo/runtime";
import { createSimulatedInteractionHost, SIMULATED_HOST_NOTE } from "./simulated-host";
import { createLiveInteractionBinding } from "./live-binding";
import type { PaneId } from "../../core/session/session";

/**
 * One pane's interactive authoring and play state, over whatever host the
 * pane was bound to.
 *
 * The UI talks to `InteractionHost` and `InteractionCatalog` only. Which host
 * that is — the simulated double under `?interactive=demo`, or the concrete
 * `SceneInteractionHost` once the integration side registers one — is the
 * binding's business, and the binding says what it is (`hostLabel`) so a
 * simulation is never presented as the real thing.
 */

export interface InteractionBinding {
  readonly catalog: InteractionCatalog;
  readonly host: InteractionHost;
  /** Shown on the HUD badge, e.g. "Simulated". Absent for the production host. */
  readonly hostLabel?: string;
  /** One line on what the host cannot do. */
  readonly hostNote?: string;
  /** Authoring transactions, when the host can list them. */
  history?(): readonly { readonly revision: number; readonly label: string }[];
  /** A camera point clear of solids between `focus` and `desired`, when the host can query collision. */
  cameraPosition?(focus: Vec3, desired: Vec3, clearance_m?: number): Vec3;
  /** True when the renderer already draws the bodies, so the overlay outlines only what is being acted on. */
  readonly presentsBodies?: boolean;
  /** Import a whole document, when the host owns persistence itself. */
  load?(raw: unknown): CommandResult | Promise<CommandResult>;
  dispose?(): void;
}

export interface PlacementDraft {
  readonly kindId: string;
  readonly config: JsonObject;
  readonly yaw_rad: number;
  readonly hit?: SurfaceHit;
  readonly transform?: Transform;
  /** Why the last commit was refused; the draft stays so the user can adjust it. */
  readonly reason?: string;
}

export interface PaneNotice { readonly tone: "error" | "info"; readonly text: string; readonly id: number }

export interface InteractionPaneState {
  readonly view: InteractionView;
  readonly inspection: DocumentInspection;
  readonly draft?: PlacementDraft;
  /** Keep placing the same kind after a commit. */
  readonly repeat: boolean;
  readonly snap_m: number;
  readonly selectedEntityId?: string;
  readonly notice?: PaneNotice;
  readonly busy: boolean;
}

const READY_TIMEOUT_MS = 8000;
const AIM_EPSILON = 1e-4;

let noticeId = 0;

function slug(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "entity";
}

function near(a: Vec3, b: Vec3, epsilon: number): boolean {
  return Math.abs(a.x - b.x) < epsilon && Math.abs(a.y - b.y) < epsilon && Math.abs(a.z - b.z) < epsilon;
}

export class InteractionPane {
  readonly store: StoreApi<InteractionPaneState>;
  readonly router: ViewportInputRouter;
  private commandCount = 0;
  private sequence = 0;
  private readonly prefix = `ui:${Math.random().toString(36).slice(2, 8)}`;
  private lastAim?: { origin: Vec3; direction: Vec3; generation: number };
  private readonly unsubscribe: () => void;

  constructor(readonly paneId: string, readonly binding: InteractionBinding) {
    const view = binding.host.getView();
    this.router = new ViewportInputRouter(binding.catalog);
    this.router.update(view);
    this.store = createStore<InteractionPaneState>(() => ({
      view, inspection: this.inspect(view), repeat: false, snap_m:0, busy: false,
    }));
    this.unsubscribe = binding.host.subscribe(() => this.refresh());
  }

  get catalog(): InteractionCatalog { return this.binding.catalog; }
  get state(): InteractionPaneState { return this.store.getState(); }

  private inspect(view: InteractionView): DocumentInspection {
    try { return this.binding.catalog.inspectDocument(view.document); }
    catch { return { document: view.document, unavailable: [] }; }
  }

  private refresh() {
    const view = this.binding.host.getView();
    const before = this.state.view;
    if (view === before) return;
    const releases = this.rebind(view, before);
    const selected = this.state.selectedEntityId;
    this.store.setState({
      view,
      inspection: view.document === before.document ? this.state.inspection : this.inspect(view),
      selectedEntityId: selected && view.document.entities.some((entity) => entity.id === selected) ? selected : undefined,
    });
    void this.sendSamples(releases);
  }

  /** Rebind the router; a pause, a reset or a lost controller drops every held input. */
  private rebind(view: InteractionView, before: InteractionView): UnsequencedSample[] {
    const dropped = (before.running && !view.running) || view.generation !== before.generation
      || (before.controlledEntityId !== undefined && view.controlledEntityId !== before.controlledEntityId);
    if (view.generation !== before.generation) this.lastAim = undefined;
    const releases = dropped ? this.router.clear() : [];
    this.router.update(view);
    // A release for an action the host already dropped is harmless; one it still holds is required.
    return dropped && view.running ? releases : [];
  }

  private notify(tone: PaneNotice["tone"], text: string) {
    noticeId += 1;
    this.store.setState({ notice: { tone, text, id: noticeId } });
  }

  dismissNotice() { this.store.setState({ notice: undefined }); }

  reportError(text: string) { this.notify("error", text); }

  /** One envelope against the host's current generation and revision. */
  async send(command: InteractionCommand, options: { readonly quiet?: boolean } = {}): Promise<CommandResult> {
    const view = this.binding.host.getView();
    this.commandCount += 1;
    const id = `${this.prefix}:${this.paneId}:${this.commandCount}`;
    let result: CommandResult;
    try {
      result = await this.binding.host.dispatch({ id, generation: view.generation, documentRevision: view.documentRevision, command });
    } catch (error) {
      result = { accepted: false, reason: error instanceof Error ? error.message : String(error) };
    }
    if (!result.accepted && !options.quiet) this.notify("error", result.reason);
    return result;
  }

  // ---- placement ------------------------------------------------------------

  startPlacement(kindId: string) {
    const kind = this.catalog.entityKind(kindId);
    if (!kind) { this.notify("error", `${kindId} is not installed`); return; }
    this.store.setState({ draft: { kindId, config: kind.schema.defaults, yaw_rad: 0 }, selectedEntityId: undefined });
  }

  cancelPlacement() { this.store.setState({ draft: undefined }); }

  setRepeat(repeat: boolean) { this.store.setState({ repeat }); }
  setSnap(snap_m: number) { if(!Number.isFinite(snap_m)||snap_m<0)return;this.store.setState({snap_m});this.hoverPlacement(this.state.draft?.hit); }

  /** Where the draft would land for a surface hit; nothing is authored. */
  hoverPlacement(hit: SurfaceHit | undefined) {
    const draft = this.state.draft;
    if (!draft) return;
    const kind = this.catalog.entityKind(draft.kindId);
    if (!kind) return;
    const transform = hit ? placementTransform(kind.placement.alignment, kind.preview(draft.config), snapSurfaceHit(hit,this.state.snap_m), draft.yaw_rad) : undefined;
    this.store.setState({ draft: { ...draft, hit, transform } });
  }

  rotatePlacement(delta_rad: number) {
    const draft = this.state.draft;
    if (!draft) return;
    this.store.setState({ draft: { ...draft, yaw_rad: draft.yaw_rad + delta_rad } });
    this.hoverPlacement(draft.hit);
  }

  private freshId(base: string, document = this.state.view.document): { id: string; n: number } {
    const used = new Set(document.entities.map((entity) => entity.id));
    for (let n = 1; ; n += 1) {
      const id = `${base}-${n}`;
      if (!used.has(id)) return { id, n };
    }
  }

  /** One `place` command: one history entry, or a refusal that keeps the draft. */
  async commitPlacement(): Promise<CommandResult | undefined> {
    const draft = this.state.draft;
    if (!draft?.transform) return undefined;
    const kind = this.catalog.entityKind(draft.kindId);
    if (!kind) return undefined;
    let entity: EntityRecord;
    const { id, n } = this.freshId(slug(kind.editor.label));
    try {
      entity = this.catalog.createEntity({ id, kind: kind.id, name: `${kind.editor.label} ${n}`, transform: draft.transform, config: draft.config });
    } catch (error) {
      this.store.setState({ draft: { ...draft, reason: error instanceof Error ? error.message : String(error) } });
      return undefined;
    }
    this.store.setState({ busy: true });
    const result = await this.send({ kind: "place", entity }, { quiet: true });
    const current = this.state.draft;
    if (!result.accepted) {
      if (current) this.store.setState({ draft: { ...current, reason: result.reason } });
      this.store.setState({ busy: false });
      return result;
    }
    this.store.setState({
      busy: false,
      draft: this.state.repeat && current ? { ...current, reason: undefined } : undefined,
      selectedEntityId: this.state.repeat ? undefined : entity.id,
    });
    return result;
  }

  // ---- authoring ------------------------------------------------------------

  select(entityId: string | undefined) {
    this.store.setState({ selectedEntityId: entityId, draft: entityId ? undefined : this.state.draft });
  }

  /** Schema errors, checked before anything is sent; empty means valid. */
  configErrors(entityId: string, behaviorId: string | undefined, config: JsonObject): readonly string[] {
    const entity = this.state.view.document.entities.find((candidate) => candidate.id === entityId);
    if (!entity) return ["No such entity"];
    const schema = behaviorId === undefined
      ? this.catalog.entityKind(entity.kind)?.schema
      : this.catalog.behaviorType(entity.behaviors.find((behavior) => behavior.id === behaviorId)?.type ?? "")?.schema;
    if (!schema) return ["This plugin is unavailable; its settings are kept but cannot be edited"];
    return schema.validate(config);
  }

  async configure(entityId: string, behaviorId: string | undefined, config: JsonObject): Promise<CommandResult> {
    const errors = this.configErrors(entityId, behaviorId, config);
    if (errors.length) return { accepted: false, reason: errors.join("; ") };
    const entity = this.state.view.document.entities.find((candidate) => candidate.id === entityId)!;
    const version = behaviorId === undefined
      ? this.catalog.entityKind(entity.kind)!.schema.version
      : this.catalog.behaviorType(entity.behaviors.find((behavior) => behavior.id === behaviorId)!.type)!.schema.version;
    return this.send({ kind: "configure", entityId, behaviorId, value: { version, config } }, { quiet: true });
  }

  addBehavior(entityId: string, type: string): Promise<CommandResult> {
    const entity = this.state.view.document.entities.find((candidate) => candidate.id === entityId);
    const plugin = this.catalog.behaviorType(type);
    if (!entity || !plugin) return Promise.resolve({ accepted: false, reason: "Unavailable behavior" });
    const used = new Set(entity.behaviors.map((behavior) => behavior.id));
    let id = slug(plugin.editor.label);
    for (let n = 2; used.has(id); n += 1) id = `${slug(plugin.editor.label)}-${n}`;
    const behaviors: BehaviorRecord[] = [...entity.behaviors, { id, type, version: plugin.schema.version, config: plugin.schema.defaults }];
    return this.send({ kind: "set-behaviors", entityId, behaviors });
  }

  removeBehavior(entityId: string, behaviorId: string): Promise<CommandResult> {
    const entity = this.state.view.document.entities.find((candidate) => candidate.id === entityId);
    if (!entity) return Promise.resolve({ accepted: false, reason: "No such entity" });
    return this.send({ kind: "set-behaviors", entityId, behaviors: entity.behaviors.filter((behavior) => behavior.id !== behaviorId) });
  }

  remove(entityId: string): Promise<CommandResult> { return this.send({ kind: "remove", entityId }); }

  move(entityId: string, transform: Transform): Promise<CommandResult> { return this.send({ kind: "transform", entityId, transform }); }

  async duplicate(entityId: string): Promise<CommandResult> {
    const entity=this.state.view.document.entities.find(e=>e.id===entityId);
    if(!entity)return {accepted:false,reason:'No such entity'};
    const kind=this.catalog.entityKind(entity.kind);
    const shape=kind?.preview(entity.config);
    const width=shape?.shape==='box'?Math.hypot(shape.dimensions_m.x,shape.dimensions_m.z):shape?shape.radius_m*2:1;
    const before=new Set(this.state.view.document.entities.map(e=>e.id));
    const result=await this.send({kind:'duplicate',entityIds:[entityId],offset_m:{x:width+.1,y:0,z:0},externalConnections:true});
    if(result.accepted)this.select(this.state.view.document.entities.find(e=>!before.has(e.id))?.id);
    return result;
  }

  connect(connection: Omit<Connection, "id">): Promise<CommandResult> {
    const used = new Set(this.state.view.document.connections.map((candidate) => candidate.id));
    let id = `${connection.source.entity}-to-${connection.target.entity}`;
    for (let n = 2; used.has(id); n += 1) id = `${connection.source.entity}-to-${connection.target.entity}-${n}`;
    return this.send({ kind: "connect", connection: { id, ...connection } });
  }

  disconnect(connectionId: string): Promise<CommandResult> { return this.send({ kind: "disconnect", connectionId }); }

  // ---- play -------------------------------------------------------------------

  transport(action: "play" | "pause" | "reset"): Promise<CommandResult> {
    if (action !== "play") this.dropInput();
    return this.send({ kind: "transport", action });
  }

  private waitForReady(): Promise<boolean> {
    const state = () => this.binding.host.getView().readiness.state;
    if (state() === "ready") return Promise.resolve(true);
    return new Promise((resolve) => {
      const finish = (ready: boolean) => { clearTimeout(timer); stop(); resolve(ready); };
      const stop = this.binding.host.subscribe(() => {
        const now = state();
        if (now === "ready") finish(true);
        else if (now === "unavailable" || now === "failed") finish(false);
      });
      const timer = setTimeout(() => finish(state() === "ready"), READY_TIMEOUT_MS);
    });
  }

  /** Control: wait for readiness, possess, and run. Each step's refusal is shown, not swallowed. */
  async control(entityId: string): Promise<CommandResult> {
    this.store.setState({ busy: true, draft: undefined, selectedEntityId: undefined });
    try {
      if (!(await this.waitForReady())) {
        const reason = this.binding.host.getView().readiness.reasons[0] ?? "The simulation is not ready";
        this.notify("error", reason);
        return { accepted: false, reason };
      }
      this.dropInput();
      const possessed = await this.send({ kind: "possess", entityId });
      if (!possessed.accepted || this.binding.host.getView().running) return possessed;
      return await this.send({ kind: "transport", action: "play" });
    } finally {
      this.store.setState({ busy: false });
    }
  }

  release(): Promise<CommandResult> {
    this.dropInput();
    return this.send({ kind: "release-control" });
  }

  /** Aim at most once per meaningful change; the host resolves what the ray offers. */
  aim(origin_m: Vec3, direction: Vec3) {
    const view = this.binding.host.getView();
    const last = this.lastAim;
    if (last && last.generation === view.generation && near(last.origin, origin_m, AIM_EPSILON) && near(last.direction, direction, AIM_EPSILON)) return;
    this.lastAim = { origin: origin_m, direction, generation: view.generation };
    void this.send({ kind: "aim", origin_m, direction }, { quiet: true });
  }

  /** Sequence and dispatch. Refusals are quiet: input that lost its target is not an error to shout about. */
  async sendSamples(samples: readonly UnsequencedSample[]): Promise<void> {
    for (const sample of samples) {
      this.sequence += 1;
      await this.send({ kind: "action", action: { ...sample, sequence: this.sequence } }, { quiet: true });
    }
  }

  /** Every key and button let go now. Blur, hidden tab, lock loss, pane switch, pause, release. */
  dropInput() { void this.sendSamples(this.router.clear()); }

  pumpInput() { void this.sendSamples(this.router.sample()); }

  async load(raw: unknown): Promise<CommandResult> {
    if (!this.binding.load) return { accepted: false, reason: "This host imports through the scene, not here" };
    const result = await this.binding.load(raw);
    if (!result.accepted) this.notify("error", result.reason);
    else this.notify("info", "Imported");
    return result;
  }

  dispose() {
    this.dropInput();
    this.unsubscribe();
    this.binding.dispose?.();
  }
}

// ---- per-pane registry --------------------------------------------------------

const panes = new Map<string, InteractionPane>();
const registryListeners = new Set<() => void>();
let registryVersion = 0;

function changed() {
  registryVersion += 1;
  for (const listener of [...registryListeners]) listener();
}

export function subscribeInteractionPanes(listener: () => void): () => void {
  registryListeners.add(listener);
  return () => { registryListeners.delete(listener); };
}

export function interactionPanesVersion(): number { return registryVersion; }

/**
 * `?interactive=demo` binds every pane to the simulated host instead of the
 * production service; `?interactive=off` hides the interactive UI entirely.
 */
export function interactiveMode(): "live" | "demo" | "off" {
  if (typeof window === "undefined") return "off";
  try {
    const value = new URLSearchParams(window.location.search).get("interactive");
    return value === "demo" ? "demo" : value === "off" ? "off" : "live";
  } catch { return "live"; }
}

export function createDemoInteractionBinding(): InteractionBinding {
  const catalog = createDemoInteractionCatalog();
  const host = createSimulatedInteractionHost({ catalog, capabilities: DEMO_HOST_CAPABILITIES, runtimes: DEMO_RUNTIME_FACTORIES });
  return {
    catalog, host, hostLabel: "Simulated", hostNote: SIMULATED_HOST_NOTE,
    history: () => host.history(), load: (raw) => host.load(raw), dispose: () => host.dispose(),
  };
}

/** Bind a pane to a host, replacing (and disposing) whatever it had. The integration side's entry point. */
export function registerInteractionBinding(paneId: string, binding: InteractionBinding): InteractionPane {
  panes.get(paneId)?.dispose();
  const pane = new InteractionPane(paneId, binding);
  panes.set(paneId, pane);
  changed();
  return pane;
}

export function unregisterInteractionBinding(paneId: string) {
  const pane = panes.get(paneId);
  if (!pane) return;
  pane.dispose();
  panes.delete(paneId);
  changed();
}

/** The pane's interaction controller, bound on first use: the production service, or the double under `?interactive=demo`. */
export function interactionPane(paneId: string): InteractionPane | undefined {
  const existing = panes.get(paneId);
  if (existing) return existing;
  const mode = interactiveMode();
  if (mode === "off") return undefined;
  const pane = new InteractionPane(paneId, mode === "demo" ? createDemoInteractionBinding() : createLiveInteractionBinding(paneId as PaneId));
  panes.set(paneId, pane);
  return pane;
}
