import type {
  ActionDefinition,
  ActionSample,
  ActionValue,
  InteractionCatalog,
  InteractionView,
} from "../../interaction";

/**
 * Keyboard and pointer state to action samples, for whatever the view says
 * this viewport may drive: every action of the controlled entity plus the
 * available offers of the current target.
 *
 * DOM-free and host-free. The caller hands it only events its focused
 * viewport owns, asks `claims` before calling `preventDefault`, and dispatches
 * what `sample` returns. `clear` is the one exit — blur, hidden tab, pointer
 * lock lost, pane switch, release, pause — and returns the releases for
 * everything still held, so no key outlives the context that pressed it.
 */

interface Binding {
  readonly entityId: string;
  readonly behaviorId: string;
  readonly action: ActionDefinition;
  /** Fed by pointer motion: a per-sample delta rather than a held state. */
  readonly delta: boolean;
}

export type UnsequencedSample = Omit<ActionSample, "sequence">;

const key = (binding: Pick<Binding, "entityId" | "behaviorId"> & { readonly actionId: string }) =>
  `${binding.entityId}\u0000${binding.behaviorId}\u0000${binding.actionId}`;

function resting(kind: ActionDefinition["kind"]): ActionValue {
  return kind === "button" ? false : kind === "axis1" ? 0 : [0, 0];
}

function same(a: ActionValue, b: ActionValue): boolean {
  if (Array.isArray(a) && Array.isArray(b)) return a[0] === b[0] && a[1] === b[1];
  return a === b;
}

const clamp = (value: number) => Math.max(-1, Math.min(1, value));

export class ViewportInputRouter {
  private bindings: readonly Binding[] = [];
  private readonly keys = new Set<string>();
  private readonly buttons = new Set<number>();
  private motion: [number, number] = [0, 0];
  /** The last value each held action was reported with. */
  private readonly sent = new Map<string, { readonly binding: Binding; readonly value: ActionValue }>();

  constructor(private readonly catalog: InteractionCatalog) {}

  /** Rebind to a new view. Actions that are no longer drivable are released by the next `sample`. */
  update(view: InteractionView): void {
    const bindings: Binding[] = [];
    const push = (entityId: string, behaviorId: string, action: ActionDefinition) =>
      bindings.push({ entityId, behaviorId, action, delta: action.bindings.some((binding) => binding.device === "pointer-motion") });
    if (view.running) {
      for (const entity of view.document.entities) {
        const controlled = entity.id === view.controlledEntityId;
        const targeted = view.target?.entityId === entity.id ? view.target : undefined;
        if (!controlled && !targeted) continue;
        for (const behavior of entity.behaviors) {
          for (const action of this.catalog.behaviorType(behavior.type)?.actions ?? []) {
            if (controlled) { push(entity.id, behavior.id, action); continue; }
            const offer = targeted!.actions.find((candidate) => candidate.behaviorId === behavior.id && candidate.actionId === action.id);
            if (offer?.availability.available) push(entity.id, behavior.id, action);
          }
        }
      }
    }
    this.bindings = bindings;
  }

  /** Whether a key belongs to play right now: bound, or held from when it was. */
  claimsKey(code: string): boolean {
    return this.keys.has(code) || this.bindings.some((binding) => binding.action.bindings.some((input) => input.device === "keyboard" && input.code === code));
  }

  claimsButton(button: number): boolean {
    return this.buttons.has(button) || this.bindings.some((binding) => binding.action.bindings.some((input) => input.device === "pointer-button" && input.button === button));
  }

  claimsMotion(): boolean {
    return this.bindings.some((binding) => binding.delta);
  }

  /** The bound keyboard code for an action, for prompts ("E  Toggle"). */
  keyFor(entityId: string, behaviorId: string, actionId: string): string | undefined {
    const binding = this.bindings.find((candidate) => candidate.entityId === entityId && candidate.behaviorId === behaviorId && candidate.action.id === actionId);
    const input = binding?.action.bindings.find((candidate) => candidate.device === "keyboard");
    return input?.device === "keyboard" ? input.code : undefined;
  }

  /** Returns whether the key was claimed. Auto-repeat is claimed but changes nothing. */
  key(code: string, down: boolean, repeat = false): boolean {
    if (!this.claimsKey(code)) return false;
    if (repeat) return true;
    if (down) this.keys.add(code); else this.keys.delete(code);
    return true;
  }

  button(button: number, down: boolean): boolean {
    if (!this.claimsButton(button)) return false;
    if (down) this.buttons.add(button); else this.buttons.delete(button);
    return true;
  }

  move(dx_px: number, dy_px: number): boolean {
    if (!this.claimsMotion()) return false;
    this.motion = [this.motion[0] + dx_px, this.motion[1] + dy_px];
    return true;
  }

  private value(binding: Binding): ActionValue {
    const kind = binding.action.kind;
    let on = false;
    let sum: [number, number] = [0, 0];
    for (const input of binding.action.bindings) {
      let value: ActionValue | undefined;
      if (input.device === "keyboard" && this.keys.has(input.code)) value = input.value;
      else if (input.device === "pointer-button" && this.buttons.has(input.button)) value = input.value;
      else if (input.device === "pointer-motion") {
        sum = [sum[0] + this.motion[0] * input.scale[0], sum[1] + this.motion[1] * input.scale[1]];
        continue;
      }
      if (value === undefined) continue;
      if (typeof value === "boolean") on ||= value;
      else if (typeof value === "number") sum = [sum[0] + value, sum[1]];
      else sum = [sum[0] + value[0], sum[1] + value[1]];
    }
    if (kind === "button") return on;
    // Deltas are motion, not stick position: they are not clamped.
    if (binding.delta) return kind === "axis1" ? sum[0] : [sum[0], sum[1]];
    return kind === "axis1" ? clamp(sum[0]) : [clamp(sum[0]), clamp(sum[1])];
  }

  /** Everything that changed since the last sample, plus releases for actions that stopped being drivable. */
  sample(): UnsequencedSample[] {
    const out: UnsequencedSample[] = [];
    const live = new Set<string>();
    for (const binding of this.bindings) {
      const id = key({ ...binding, actionId: binding.action.id });
      live.add(id);
      const value = this.value(binding);
      const base = { entityId: binding.entityId, behaviorId: binding.behaviorId, actionId: binding.action.id };
      if (binding.delta) {
        if (!same(value, resting(binding.action.kind))) out.push({ ...base, value });
        continue;
      }
      const before = this.sent.get(id)?.value ?? resting(binding.action.kind);
      if (same(before, value)) continue;
      out.push({ ...base, value, edge: binding.action.kind === "button" ? (value ? "pressed" : "released") : undefined });
      if (same(value, resting(binding.action.kind))) this.sent.delete(id);
      else this.sent.set(id, { binding, value });
    }
    for (const [id, held] of [...this.sent]) {
      if (live.has(id)) continue;
      out.push(this.release(held.binding));
      this.sent.delete(id);
    }
    this.motion = [0, 0];
    return out;
  }

  private release(binding: Binding): UnsequencedSample {
    return {
      entityId: binding.entityId, behaviorId: binding.behaviorId, actionId: binding.action.id,
      value: resting(binding.action.kind), edge: binding.action.kind === "button" ? "released" : undefined,
    };
  }

  /** Forget every key and button; return the releases the host still needs to hear. */
  clear(): UnsequencedSample[] {
    const out = [...this.sent.values()].map((held) => this.release(held.binding));
    this.sent.clear();
    this.keys.clear();
    this.buttons.clear();
    this.motion = [0, 0];
    return out;
  }

  /** Anything held right now, for asserting the "nothing stuck" rule. */
  holding(): boolean {
    return this.keys.size > 0 || this.buttons.size > 0 || this.sent.size > 0;
  }
}
