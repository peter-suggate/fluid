import type { Vec3 } from "../../core/model";
import type { PaneId } from "../../core/session/session";
import { simulation, type SimulationController } from "../../core/simulation/controller";
import { interactionCatalog, type CommandEnvelope, type CommandResult, type InteractionHost, type InteractionView } from "../../interaction";
import type { InteractionBinding } from "./pane";

/**
 * The production binding: the pane's own `simulation.interaction(paneId)`
 * service and the installed catalog.
 *
 * The service is resolved lazily. Constructing it builds a Box3D world, and a
 * scene with no interactive entities should not pay for that because the
 * placement row is on screen — so until the scene has an entity, or the user
 * dispatches the first command, the view is an empty edit document and nothing
 * is loaded. The first command brings the host up, waits for it to be ready
 * and is then stamped with the generation and revision the host actually has:
 * there was no world for the placeholder view to be stale against.
 *
 * The controller replaces the service when a pane's session is re-attached, so
 * every read re-resolves it and moves the subscription across.
 */

interface CameraClearance { cameraPosition(focus: Vec3, desired: Vec3, clearance_m?: number): Vec3 }

const PLACEHOLDER: InteractionView = Object.freeze({
  generation: 0,
  documentRevision: 0,
  document: Object.freeze({ schemaVersion: 1 as const, entities: [], connections: [] }),
  readiness: Object.freeze({ state: "ready" as const, reasons: [] }),
  running: false,
  context: "edit" as const,
  controlCandidates: [],
});

export function createLiveInteractionBinding(paneId: PaneId, controller: SimulationController = simulation): InteractionBinding {
  const sceneHasEntities = () => {
    try { return Boolean(controller.session(paneId).scene.getState().scene.interaction?.entities.length); }
    catch { return false; }
  };
  const listeners = new Set<() => void>();
  let host: (InteractionHost & Partial<CameraClearance> & { ready?: Promise<void> }) | undefined;
  let detach: (() => void) | undefined;

  const notify = () => { for (const listener of [...listeners]) listener(); };

  /** The pane's current service, attaching to it if it changed. `create` brings it up. */
  function current(create: boolean) {
    if (!host && !create && !sceneHasEntities()) return undefined;
    let next: typeof host;
    try { next = controller.interaction(paneId); } catch { return host; }
    if (next !== host) {
      detach?.();
      host = next;
      detach = next.subscribe(notify);
      queueMicrotask(notify);
    }
    return host;
  }

  const live: InteractionHost = {
    getView: () => current(false)?.getView() ?? PLACEHOLDER,
    subscribe(listener) {
      listeners.add(listener);
      current(false);
      return () => { listeners.delete(listener); };
    },
    async dispatch(envelope: CommandEnvelope): Promise<CommandResult> {
      const fresh = !host;
      const target = current(true);
      if (!target) return { accepted: false, reason: "The interactive runtime is not available for this pane" };
      if (fresh) {
        await target.ready?.catch(() => undefined);
        const view = target.getView();
        return target.dispatch({ ...envelope, generation: view.generation, documentRevision: view.documentRevision });
      }
      return target.dispatch(envelope);
    },
  };

  return {
    catalog: interactionCatalog,
    host: live,
    presentsBodies: true,
    history:()=>controller.session(paneId).history.getState().past.map((entry,revision)=>({revision:revision+1,label:entry.label})),
    load:raw=>{
      const view=live.getView();
      return live.dispatch({id:`import:${crypto.randomUUID()}`,generation:view.generation,documentRevision:view.documentRevision,command:{kind:'import',document:raw}});
    },
    cameraPosition: (focus, desired, clearance_m) => host?.cameraPosition?.(focus, desired, clearance_m) ?? desired,
    // The service is the controller's: the binding lets go of it, never disposes it.
    dispose: () => { detach?.(); detach = undefined; host = undefined; listeners.clear(); },
  };
}
