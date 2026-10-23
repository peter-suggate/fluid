import type { SceneDescription, Vec3 } from '../../core/model';
import type { RigidBodyState } from '../../core/rigid-body';
import { interactionCatalog, interactionRuntimes } from '../installed';
import type { ActionSample, BehaviorEvent, BehaviorRuntime, CommandEnvelope, CommandResult, InteractionDocument, InteractionHost, InteractionPlan, InteractionPhase, InteractionView, PhysicsCommand, WorldSnapshot } from '../contracts';
import { Box3DWorld } from '../adapters/box3d/world';
import { configurationImpact, duplicateEntities } from '../authoring';
import type { UpdateImpact } from '../../framework/composition';
import { EMPTY_INTERACTIONS, type InteractionFluidDriver, type SensorContact } from './physics';
export interface InteractionHostOptions {
  commitScene?(scene: SceneDescription, label: string, impact: UpdateImpact): void;
  transport?(action: 'play' | 'pause' | 'reset'): void;
  published?(bodies: RigidBodyState[], snapshot: WorldSnapshot): void;
  fluid?: InteractionFluidDriver;
}
const copy = <T>(value: T): T => structuredClone(value);
/** Concrete pane host. All structural mutations and steps serialize on one queue. */
export class SceneInteractionHost implements InteractionHost {
  private view: InteractionView;
  private listeners = new Set<() => void>();
  private world?: Box3DWorld;
  private plan?: InteractionPlan;
  private runtimes = new Map<string, BehaviorRuntime>();
  private tail: Promise<unknown> = Promise.resolve();
  private commandIds = new Set<string>();
  private held = new Map<string, ActionSample>();
  private edges: ActionSample[] = [];
  private lastSequence = -1;
  private events: BehaviorEvent[] = [];
  private sensors: SensorContact[] = [];
  private origin: Vec3 = { x: 0, y: 0, z: 0 };
  private direction: Vec3 = { x: 0, y: 0, z: -1 };
  private disposed = false;
  private detachFluidReadiness?: () => void;
  private revision = 0;
  private generation = 0;
  private solidRevision = 0;
  readonly ready: Promise<void>;
  constructor(private scene: SceneDescription, private options: InteractionHostOptions = {}) {
    this.view = { generation: 0, documentRevision: 0, document: scene.interaction ?? EMPTY_INTERACTIONS, readiness: { state: 'preparing', reasons: [] }, running: false, context: 'edit', controlCandidates: [] };
    this.setFluidDriver(options.fluid);
    this.ready = this.enqueue(() => this.rebuild(scene));
    this.ready.catch(() => undefined);
  }
  getView = (): InteractionView => this.view;
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; };
  private publish(patch: Partial<InteractionView>) { this.view = { ...this.view, ...patch }; for (const listener of this.listeners) listener(); }
  private enqueue<T>(operation: () => Promise<T> | T): Promise<T> {
    const task = this.tail.then(() => { if (this.disposed) throw new Error('Interaction host disposed'); return operation(); });
    this.tail = task.catch(() => undefined); return task;
  }
  private async build(scene: SceneDescription, generation: number) {
    const plan = interactionCatalog.planDocument(scene.interaction ?? EMPTY_INTERACTIONS, ['box3d']);
    const runtimes = new Map<string, BehaviorRuntime>();
    const entities = plan.document.entities.flatMap(entity => {
      const bodyFactories = entity.behaviors.map(b => interactionRuntimes.find(f => f.type === b.type)).filter(f => f?.body);
      if (bodyFactories.length > 1) throw new Error(`Multiple physics bodies own ${entity.id}`);
      return bodyFactories.length ? [{ entity, recipe: bodyFactories[0]!.body!(entity) }] : [];
    });
    let world: Box3DWorld | undefined;
    try {
      world = await Box3DWorld.create(scene, entities, generation);
      for (const entity of plan.document.entities) for (const behavior of entity.behaviors) {
        const factory = interactionRuntimes.find(f => f.type === behavior.type);
        if (!factory) throw new Error(`Missing runtime for ${behavior.type}`);
        runtimes.set(`${entity.id}/${behavior.id}`, factory.create(entity, behavior));
      }
      return { plan, world, runtimes };
    } catch (error) { world?.destroy(); for (const runtime of runtimes.values()) runtime.dispose(); throw error; }
  }
  private adopt(scene: SceneDescription, built: Awaited<ReturnType<SceneInteractionHost["build"]>>) {
    this.world?.destroy(); for (const r of this.runtimes.values()) r.dispose();
    this.world = built.world; this.runtimes = built.runtimes; this.plan = built.plan; this.scene = scene;
    this.generation++; this.solidRevision++; this.commandIds.clear(); this.lastSequence = -1; this.clearInput(); this.events = []; this.sensors = [];
    this.publish({ generation: this.generation, document: built.plan.document, snapshot: undefined, documentRevision: this.revision, readiness: { state: 'preparing', reasons: [] }, running: false, controlledEntityId: undefined, context: "edit",
      controlCandidates: built.plan.document.entities.filter(e => e.behaviors.some(b => interactionCatalog.behaviorType(b.type)?.provides?.includes("controllable"))).map(e => ({ entityId: e.id, availability: { available: true as const } })), target: undefined });
    this.updateReadiness(); this.publishSnapshot(0, 0);
  }
  /** Conservative ray fan for third-person camera clearance; never changes physics. */
  cameraPosition(focus: Vec3, desired: Vec3, clearance_m = .15): Vec3 {
    const d = { x: desired.x-focus.x, y: desired.y-focus.y, z: desired.z-focus.z }, length = Math.hypot(d.x,d.y,d.z);
    if (!Number.isFinite(length) || !(clearance_m >= 0 && Number.isFinite(clearance_m))) throw new Error('Invalid camera query');
    if (!this.world || length < 1e-6) return { ...desired };
    const direction = {x:d.x/length,y:d.y/length,z:d.z/length};
    let distance=length;
    for (const offset of [{x:0,y:0,z:0},{x:clearance_m,y:0,z:0},{x:-clearance_m,y:0,z:0},{x:0,y:clearance_m,z:0},{x:0,y:-clearance_m,z:0},{x:0,y:0,z:clearance_m},{x:0,y:0,z:-clearance_m}]) {
      const hit=this.world.ray({x:focus.x+offset.x,y:focus.y+offset.y,z:focus.z+offset.z},direction,length,1);
      if(hit)distance=Math.min(distance,Math.max(0,hit.distance-clearance_m));
    }
    return {x:focus.x+direction.x*distance,y:focus.y+direction.y*distance,z:focus.z+direction.z*distance};
  }
  currentBodies(): RigidBodyState[] { return this.world?.renderBodies() ?? []; }
  private async rebuild(scene: SceneDescription) {
    this.publish({ readiness: { state: 'preparing', reasons: [] }, running: false, controlledEntityId: undefined, context: 'edit' });
    try {
      const built = await this.build(scene, this.generation + 1);
      if (this.disposed) { built.world.destroy(); for (const r of built.runtimes.values()) r.dispose(); return; }
      this.adopt(scene, built);
    } catch (error) {
      this.publish({ readiness: { state: 'unavailable', reasons: [error instanceof Error ? error.message : String(error)] }, running: false });
      throw error;
    }
  }
  private updateReadiness(recoverSceneEdit = false) {
    if (!this.world || this.disposed || (this.view.readiness.state === 'failed' && !recoverSceneEdit)) return;
    const failure = this.scene.systems?.fluid !== false ? this.options.fluid?.failureReason?.() : undefined;
    if (failure) {
      this.clearInput();
      this.publish({ running: false, readiness: { state: 'failed', reasons: [failure] } });
      this.options.transport?.('pause');
      return;
    }
    const missing = this.scene.systems?.fluid !== false && (!this.options.fluid || this.options.fluid.isReady?.() === false);
    const readiness: InteractionView['readiness'] = missing
      ? { state: 'preparing', reasons: ['Waiting for fluid adapter to be ready'] }
      : { state: 'ready', reasons: [] };
    if (JSON.stringify(readiness) !== JSON.stringify(this.view.readiness)) this.publish({ readiness });
  }
  setFluidDriver(driver: InteractionFluidDriver | undefined) {
    this.detachFluidReadiness?.();
    this.options.fluid = driver;
    this.detachFluidReadiness = driver?.subscribeReadiness?.(() => this.updateReadiness());
    this.updateReadiness();
  }
  setRunning(running: boolean) {
    if (!running) this.clearInput();
    this.publish({ running, context: this.view.controlledEntityId ? 'play' : running ? 'observe' : 'edit' });
  }
  clearInput() { this.held.clear(); this.edges = []; this.world?.releaseGrab(); }
  releaseInput() { this.clearInput(); this.publish({ controlledEntityId: undefined, target: undefined, context: this.view.running ? 'observe' : 'edit' }); }
  validateSolidEdit(scene: SceneDescription) { this.world?.validateSolidEdit(scene); }
  replaceScene(scene: SceneDescription, reset = false): Promise<void> {
    return this.enqueue(async () => {
      if (!reset && JSON.stringify(scene.interaction) === JSON.stringify(this.scene.interaction) && JSON.stringify(scene.rigidBodies) === JSON.stringify(this.scene.rigidBodies)
        && JSON.stringify(scene.fluid.gravity_m_s2) === JSON.stringify(this.scene.fluid.gravity_m_s2) && scene.systems?.fluid === this.scene.systems?.fluid && this.world) {
        this.world.validateSolidEdit(scene); this.world.updateSolids(scene); this.scene = scene; this.solidRevision++; this.updateReadiness(true); return;
      }
      this.revision++; await this.rebuild(scene);
    }).catch(error => {
      this.publish({ running: false, readiness: { state: 'failed', reasons: [String(error)] } });
      this.options.transport?.('pause'); throw error;
    });
  }
  dispatch(envelope: CommandEnvelope): Promise<CommandResult> {
    return this.enqueue(async (): Promise<CommandResult> => {
      try {
        if (!envelope.id || envelope.generation !== this.generation || envelope.documentRevision !== this.revision) throw new Error('Stale interaction command');
        if (this.commandIds.has(envelope.id)) throw new Error('Duplicate interaction command');
        this.commandIds.add(envelope.id);
        const command = envelope.command;
        if (command.kind === 'possess' || command.kind === 'transport') this.updateReadiness();
        if (command.kind === 'transport') {
          if (command.action === 'reset') await this.rebuild(this.scene);
          else { if (command.action === 'play' && this.view.readiness.state !== 'ready') throw new Error(this.view.readiness.reasons.join('; ')); this.setRunning(command.action === 'play'); }
          this.options.transport?.(command.action);
          return { accepted: true, documentRevision: this.revision, impact: command.action === 'reset' ? 'reset' : 'live' };
        }
        if (command.kind === 'possess') {
          if (!this.view.controlCandidates.some(c => c.entityId === command.entityId && c.availability.available)) throw new Error('Entity is not controllable');
          if (this.view.readiness.state !== 'ready') throw new Error(this.view.readiness.reasons.join('; '));
          this.clearInput(); this.publish({ controlledEntityId: command.entityId, context: 'play', running: true }); this.options.transport?.('play');
        } else if (command.kind === 'release-control') this.releaseInput();
        else if (command.kind === 'aim') {
          const d = command.direction;
          if (![...Object.values(d), ...Object.values(command.origin_m)].every(Number.isFinite) || Math.abs(Math.hypot(d.x, d.y, d.z) - 1) > 1e-5) throw new Error('Aim needs a finite origin and unit direction');
          this.direction = { ...d }; this.origin = { ...command.origin_m };
          const hit = this.world?.ray(command.origin_m, d, 3);
          const entity = this.view.document.entities.find(e => e.id === hit?.entityId);
          const actions = entity?.behaviors.filter(b => !interactionCatalog.behaviorType(b.type)?.provides?.includes('controllable')).flatMap(b => (interactionCatalog.behaviorType(b.type)?.actions ?? []).filter(a => a.kind === 'button').map(a => ({ behaviorId: b.id, actionId: a.id, label: a.label, availability: { available: true as const } }))) ?? [];
          this.publish({ target: entity && actions.length ? { entityId: entity.id, actions } : undefined });
        } else if (command.kind === 'action') {
          const sample = command.action;
          if (!this.view.running) throw new Error('Simulation is paused');
          interactionCatalog.validateAction(this.view.document, sample);
          if (sample.sequence <= this.lastSequence) throw new Error('Stale input sequence');
          const neutral = sample.value === false || sample.value === 0 || Array.isArray(sample.value) && sample.value.every(n => n === 0);
          if (!neutral && sample.entityId !== this.view.controlledEntityId && !(this.view.target?.entityId === sample.entityId && this.view.target.actions.some(a => a.actionId === sample.actionId && a.behaviorId === sample.behaviorId))) throw new Error('Input target is not controlled or in reach');
          this.lastSequence = sample.sequence;
          const key = `${sample.entityId}/${sample.behaviorId}/${sample.actionId}`;
          this.held.set(key, { ...copy(sample), edge: undefined });
          if (sample.edge) this.edges.push(copy(sample));
        } else {
          const doc = this.view.document;
          let next: InteractionDocument = doc;
          const entityId = 'entityId' in command ? command.entityId : undefined;
          if (entityId && !doc.entities.some(e => e.id === entityId)) throw new Error('Unknown entity');
          if (command.kind === 'place') next = { ...doc, entities: [...doc.entities, command.entity] };
          else if (command.kind === 'duplicate') next = duplicateEntities(doc,command.entityIds,command.offset_m,command.externalConnections);
          else if (command.kind === 'import') next = interactionCatalog.inspectDocument(command.document).document;
          else if (command.kind === 'remove') next = { ...doc, entities: doc.entities.filter(e => e.id !== command.entityId), connections: doc.connections.filter(c => c.source.entity !== command.entityId && c.target.entity !== command.entityId) };
          else if (command.kind === 'transform') next = { ...doc, entities: doc.entities.map(e => e.id === command.entityId ? { ...e, transform: command.transform } : e) };
          else if (command.kind === 'set-behaviors') next = { ...doc, entities: doc.entities.map(e => e.id === command.entityId ? { ...e, behaviors: command.behaviors } : e) };
          else if (command.kind === 'configure') next = { ...doc, entities: doc.entities.map(e => {
            if (e.id !== command.entityId) return e;
            if (command.behaviorId && !e.behaviors.some(b => b.id === command.behaviorId)) throw new Error('Unknown behavior');
            return command.behaviorId ? { ...e, behaviors: e.behaviors.map(b => b.id === command.behaviorId ? { ...b, version: command.value.version, config: command.value.config } : b) } : { ...e, version: command.value.version, config: command.value.config };
          }) };
          else if (command.kind === 'connect') next = { ...doc, connections: [...doc.connections, command.connection] };
          else if (command.kind === 'disconnect') { if (!doc.connections.some(c => c.id === command.connectionId)) throw new Error('Unknown connection'); next = { ...doc, connections: doc.connections.filter(c => c.id !== command.connectionId) }; }
          const inspection=interactionCatalog.inspectDocument(next);
          if(inspection.unavailable.length){
            if(command.kind!=='import'&&command.kind!=='remove')throw new Error('Resolve missing plugins before editing their configuration');
            this.world?.destroy();this.world=undefined;for(const runtime of this.runtimes.values())runtime.dispose();this.runtimes.clear();this.plan=undefined;
            this.scene={...this.scene,interaction:inspection.document};this.generation++;this.revision++;this.clearInput();this.commandIds.clear();this.lastSequence=-1;this.events=[];this.sensors=[];
            this.publish({generation:this.generation,documentRevision:this.revision,document:inspection.document,snapshot:undefined,running:false,context:'edit',controlledEntityId:undefined,target:undefined,controlCandidates:[],readiness:{state:'unavailable',reasons:inspection.unavailable.map(x=>x.reason)}});
            this.options.commitScene?.(this.scene,`Interactive ${command.kind}`,'reset');
            this.options.published?.([],{generation:this.generation,tick:0,solidRevision:this.solidRevision,time_s:0,bodies:[]});
            return {accepted:true,documentRevision:this.revision,impact:'reset'};
          }
          const nextPlan=interactionCatalog.planDocument(inspection.document, ['box3d']);
          const validated = nextPlan.document;
          const scene = { ...this.scene, interaction: validated };
          if(command.kind==='configure'){
            const previous=doc.entities.find(e=>e.id===command.entityId)!;
            const entity=validated.entities.find(e=>e.id===command.entityId)!;
            const record=command.behaviorId?previous.behaviors.find(b=>b.id===command.behaviorId)!:previous;
            const controls=command.behaviorId?interactionCatalog.behaviorType(previous.behaviors.find(b=>b.id===command.behaviorId)!.type)!.editor.controls:interactionCatalog.entityKind(previous.kind)!.editor.controls;
            const impact=configurationImpact(record.config,command.value.config,controls);
            if(JSON.stringify(record.config)===JSON.stringify(command.value.config))return {accepted:true,documentRevision:this.revision,impact:'live'};
            if(impact!=='reset'){
              const replacements=new Map<string,BehaviorRuntime>();
              try{
                // Geometry rebuilds retain behavior state (e.g. an activated lift).
                if(impact==='rebuild'&&command.behaviorId)for(const behavior of entity.behaviors){
                  if(command.behaviorId&&behavior.id!==command.behaviorId)continue;
                  replacements.set(`${entity.id}/${behavior.id}`,interactionRuntimes.find(f=>f.type===behavior.type)!.create(entity,behavior));
                }
                const body=entity.behaviors.map(b=>interactionRuntimes.find(f=>f.type===b.type)).find(f=>f?.body);
                this.world!.configure(entity,body?.body?.(entity),impact==='rebuild');
              }catch(error){for(const runtime of replacements.values())runtime.dispose();throw error;}
              for(const [key,runtime] of replacements){this.runtimes.get(key)?.dispose();this.runtimes.set(key,runtime);}
              this.plan=nextPlan;this.scene=scene;this.revision++;
              this.publish({document:validated,documentRevision:this.revision});
              this.options.commitScene?.(scene,'Interactive configure',impact);
              this.publishSnapshot(this.view.snapshot!.tick,this.view.snapshot!.time_s);
              return {accepted:true,documentRevision:this.revision,impact};
            }
          }
          // Build before committing: failed configurations never enter history.
          const candidate = await this.build(scene, this.generation + 1);
          try {
            if (command.kind === 'place') candidate.world.validatePlacement(command.entity.id);
            else if(command.kind==='duplicate'||command.kind==='import')for(const entity of validated.entities) {
              if(command.kind==='import'||!doc.entities.some(e=>e.id===entity.id))candidate.world.validatePlacement(entity.id);
            }
            else if (entityId && command.kind !== 'remove') candidate.world.validatePlacement(entityId);
          } catch (error) { candidate.world.destroy(); for (const r of candidate.runtimes.values()) r.dispose(); throw error; }
          this.revision++; this.adopt(scene, candidate); this.options.commitScene?.(scene, `Interactive ${command.kind}`, 'reset');
          return { accepted: true, documentRevision: this.revision, impact: 'reset' };
        }
        return { accepted: true, documentRevision: this.revision, impact: 'live' };
      } catch (error) { return { accepted: false, reason: error instanceof Error ? error.message : String(error) }; }
    });
  }
  private publishSnapshot(tick: number, time_s: number) {
    if (!this.world) return;
    const snapshot: WorldSnapshot = { generation: this.generation, tick, solidRevision: this.solidRevision, time_s, bodies: this.world.snapshots() };
    this.publish({ snapshot }); this.options.published?.(this.world.renderBodies(), snapshot);
  }
  /** One shared-clock step; fluid completion precedes Box3D integration. */
  advance(dt_s: number): Promise<void> {
    return this.enqueue(async () => {
      this.updateReadiness();
      if (!this.world || !this.plan || this.view.readiness.state !== 'ready') throw new Error('Interactive runtime is not ready');
      if (!(dt_s > 0 && dt_s <= .1)) throw new Error("Interactive dt must be in (0, 0.1]");
      const world = this.world, tick = (this.view.snapshot?.tick ?? 0) + 1, time_s = (this.view.snapshot?.time_s ?? 0) + dt_s;
      const identity = { generation: this.generation, tick, solidRevision: this.solidRevision };
      // Revalidate ordinary use actions at the admitted tick. A grab's joint
      // owns its continuation after acquisition until release/focus loss.
      const reachable = world.ray(this.origin, this.direction, 3)?.entityId;
      const actions = [...this.held.values(), ...this.edges].filter(a => a.edge === 'released' || a.entityId === this.view.controlledEntityId || a.entityId === reachable ||
        this.view.document.entities.find(e => e.id === a.entityId)?.behaviors.some(b => b.id === a.behaviorId && interactionCatalog.behaviorType(b.type)?.provides?.includes('grabbable')));
      this.edges = [];
      const commands: PhysicsCommand[] = [], outgoing: BehaviorEvent[] = [];
      const run = (phase: InteractionPhase) => {
        for (const step of this.plan!.phases[phase]) {
          const entity = this.plan!.document.entities.find(e => e.id === step.entityId)!;
          const behavior = entity.behaviors.find(b => b.id === step.behaviorId)!;
          this.runtimes.get(`${entity.id}/${behavior.id}`)!.tick(phase, { ...identity, dt_s, entity, behavior, viewDirection: this.direction, viewOrigin_m: this.origin,
            input: { ...identity, actions }, snapshot: this.view.snapshot!, sensors: this.sensors,
            events: this.events.filter(event => event.target.entity === entity.id && event.target.behavior === behavior.id),
            enqueue: c => commands.push(c), emit: (portId, value) => {
              if (!interactionCatalog.behaviorType(behavior.type)?.outputs?.some(p => p.id === portId)) throw new Error('Undeclared behavior output');
              for (const connection of this.plan!.document.connections) if (connection.source.entity === entity.id && connection.source.behavior === behavior.id && connection.source.port === portId) outgoing.push({ ...identity, connectionId: connection.id, source: connection.source, target: connection.target, value: copy(value) });
            } });
        }
      };
      try {
        world.ensureCoverage();
        run('intent'); run('before-physics'); for (const command of commands) world.command(command, dt_s); commands.length = 0;
        if (this.scene.systems?.fluid !== false) {
          const exchange = await this.options.fluid!.step({ ...identity, time_s, dt_s, bodies: world.renderBodies(), bodyGenerations:world.fluidBodyGenerations() });
          world.applyFluid(exchange, identity);
        }
        this.sensors = world.step(dt_s); this.publishSnapshot(tick, time_s); run('after-physics');
        if (commands.length) throw new Error('Physics commands must be submitted before physics');
        this.events = outgoing;
      } catch (error) { this.publish({ running: false, readiness: { state: 'failed', reasons: [error instanceof Error ? error.message : String(error)] } }); this.options.transport?.('pause'); throw error; }
    });
  }
  async dispose() { this.disposed = true; this.detachFluidReadiness?.(); this.detachFluidReadiness = undefined; await this.tail; this.world?.destroy(); this.world = undefined; for (const r of this.runtimes.values()) r.dispose(); this.runtimes.clear(); this.listeners.clear(); }
}
