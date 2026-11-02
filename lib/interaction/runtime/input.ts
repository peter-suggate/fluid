import type { ActionDefinition, ActionSample, ActionValue, InteractionCommand } from '../contracts';
import { interactionCatalog } from '../installed';
import type { SceneInteractionHost } from './host';
import { RunScope } from '../../core/run-scope';

let nextRouterId = 0;
let inputSequence = 0;

/** DOM-independent arbitration. The UI passes only events owned by its focused viewport. */
export class InteractionInputRouter {
  private keys = new Set<string>();
  private pointer = new Set<number>();
  private sent = new Map<string, ActionSample>();
  private readonly routerId = ++nextRouterId;
  private command = 0;
  /** Input held since the last reset; a reset supersedes everything sent under it. */
  private readonly holds = new RunScope();
  private hold = this.holds.start();
  private pending: Promise<void> = Promise.resolve();
  private unsubscribe: () => void;
  constructor(private readonly host: SceneInteractionHost) {
    let generation = host.getView().generation, running = host.getView().running, controlled = host.getView().controlledEntityId;
    this.unsubscribe = host.subscribe(() => {
      const view = host.getView(), invalidate = generation !== view.generation || running && !view.running || controlled !== view.controlledEntityId;
      generation = view.generation; running = view.running; controlled = view.controlledEntityId;
      if (invalidate) this.reset(false);
    });
  }
  dispose() { this.unsubscribe(); this.reset(); this.holds.dispose(); }
  private dispatch(command: InteractionCommand) {
    const hold = this.hold;
    // A refusal resets only the hold it was sent under: one that comes back
    // after a newer reset must not clear keys pressed since.
    const refused = () => { if (hold.current()) this.reset(false); };
    this.pending = this.pending.then(async () => {
      if (!hold.current()) return;
      const view = this.host.getView();
      const result = await this.host.dispatch({ id: `input:${this.routerId}:${++this.command}`, generation: view.generation, documentRevision: view.documentRevision, command });
      if (!result.accepted) refused();
    }).catch(refused);
  }
  /** Return true only for gameplay-owned bindings; callers preventDefault only then. */
  key(code: string, down: boolean, blocked = false): boolean {
    if (blocked) { this.reset(); return false; }
    if (!this.host.getView().running) return false;
    const owned = this.actions().some(({ action }) => action.bindings.some(b => b.device === 'keyboard' && b.code === code));
    if (!owned && !this.keys.has(code)) return false;
    if (down) this.keys.add(code); else this.keys.delete(code);
    this.flush(); return true;
  }
  button(button: number, down: boolean, blocked = false): boolean {
    if (blocked) { this.reset(); return false; }
    if (!this.host.getView().running || !this.actions().some(({ action }) => action.bindings.some(b => b.device === 'pointer-button' && b.button === button))) return false;
    if (down) this.pointer.add(button); else this.pointer.delete(button);
    this.flush(); return true;
  }
  private actions() {
    const view = this.host.getView();
    return view.document.entities.flatMap(entity => entity.behaviors.flatMap(behavior => (interactionCatalog.behaviorType(behavior.type)?.actions ?? []).filter(action =>
      entity.id === view.controlledEntityId || view.target?.entityId === entity.id && view.target.actions.some(a => a.behaviorId === behavior.id && a.actionId === action.id)
    ).map(action => ({ entityId: entity.id, behaviorId: behavior.id, action }))));
  }
  private value(action: ActionDefinition): ActionValue {
    const values = action.bindings.flatMap(b => b.device === 'keyboard' && this.keys.has(b.code) || b.device === 'pointer-button' && this.pointer.has(b.button) ? ('value' in b ? [b.value] : []) : []);
    const clamp = (n: number) => Math.max(-1, Math.min(1, n));
    if (action.kind === 'button') return values.some(v => v === true);
    if (action.kind === 'axis1') return clamp(values.reduce<number>((sum, v) => sum + (typeof v === 'number' ? v : 0), 0));
    return [0, 1].map(i => clamp(values.reduce<number>((sum, v) => sum + (Array.isArray(v) ? v[i] : 0), 0))) as [number, number];
  }
  /** Call after targeting changes, as well as keyboard/pointer changes. */
  flush() {
    const active = new Set<string>();
    for (const {entityId, behaviorId, action} of this.actions()) {
      const key = `${entityId}/${behaviorId}/${action.id}`, value = this.value(action), before = this.sent.get(key);
      active.add(key);
      if (JSON.stringify(before?.value) === JSON.stringify(value)) continue;
      const sample: ActionSample = { entityId, behaviorId, actionId: action.id, value, sequence: ++inputSequence,
        edge: action.kind === 'button' ? value ? 'pressed' : 'released' : undefined };
      this.sent.set(key, sample); this.dispatch({ kind: 'action', action: sample });
    }
    for (const [key, before] of this.sent) if (!active.has(key)) {
      this.dispatch({ kind: 'action', action: { ...before, value: Array.isArray(before.value) ? [0, 0] : typeof before.value === 'number' ? 0 : false, sequence: ++inputSequence, edge: typeof before.value === 'boolean' ? 'released' : undefined } });
      this.sent.delete(key);
    }
  }
  /** Blur, hidden tab, modal, pane switch, pointer-lock loss and unmount. */
  reset(releaseControl = true) {
    if (this.holds.disposed) return;
    this.hold = this.holds.start(); this.keys.clear(); this.pointer.clear(); this.sent.clear();
    if (releaseControl) void this.host.releaseInput(); else this.host.clearInput();
  }
  settled(): Promise<void> { return this.pending; }
}
