import { assertNoSolidOverlap } from "../solid-world/overlap";
import { fluidSolidWorldForScene } from '../../../core/solid-world';
import type { RigidBodyDescription, SceneDescription, Vec3 } from '../../../core/model';
import { boundingRadius, initializeRigidBody, primitiveVolume, quaternionMultiply, type RigidBodyState } from '../../../core/rigid-body';
import { cupWallThickness_m } from '../../../core/scene-shape';
import type { BodySnapshot, EntityRecord, FluidExchange, PhysicsCommand, TickIdentity } from '../../contracts';
import { assertFiniteSnapshot, shapeDescription, fluidBodyParts, type BodyRecipe, type SensorContact } from '../../runtime/physics';
import { compileSolidChunks, SolidCollisionCache } from '../solid-world/collision';
import { loadBox3D, type B3Body, type B3Joint, type B3Ray, type B3Shape, type B3World } from './api';
export interface WorldBody { entityId: string; generation: number; description: RigidBodyDescription; recipe?: BodyRecipe; native: B3Body; tag: number; grounded: boolean; coyote: number; jumpBuffer: number; submerged: number; facing?: number; flipTime?: number; flipped?: boolean; climbing?: string }
const zero = () => ({ x: 0, y: 0, z: 0 });
export class Box3DWorld {
  readonly bodies = new Map<string, WorldBody>();
  private tags = new Map<number, WorldBody>();
  private shapeEntities = new Map<number, string>();
  private sensorPairs = new Map<string, Set<string>>();
  private chunks = new Map<string, { signature: string; body: B3Body }>();
  private collisionCache = new SolidCollisionCache();
  private disposed = false;
  private coverage = new Map<string, Vec3>();
  private lastFluidTick = -1;
  private grab?: { entityId: string; anchor: B3Body; joint: B3Joint };
  private constructor(private world: B3World, private scene: SceneDescription, readonly generation: number) {}
  static async create(scene: SceneDescription, entities: readonly { entity: EntityRecord; recipe: BodyRecipe }[], generation: number) {
    const module = await loadBox3D();
    const native = new module.World({ gravity: scene.fluid.gravity_m_s2, enableContinuous: true });
    const result = new Box3DWorld(native, scene, generation);
    try {
      for (const body of scene.rigidBodies) result.add(body.id, body);
      for (const { entity, recipe } of entities) result.add(entity.id, shapeDescription(entity, recipe), recipe);
      if (result.renderBodies().length > 12) throw new Error('Interactive rendering and fluid coupling currently support 12 physical bodies per scene');
      result.updateSolids(scene);
      return result;
    } catch (error) { result.destroy(); throw error; }
  }
  private add(entityId: string, description: RigidBodyDescription, recipe?: BodyRecipe) {
    if (recipe?.character && (Math.abs(description.orientation.x) > 1e-6 || Math.abs(description.orientation.z) > 1e-6)) throw new Error("Character capsules must be upright");
    if (this.bodies.has(entityId)) throw new Error(`Duplicate runtime entity ${entityId}`);
    const native = this.world.createBody({ type: recipe?.motion ?? description.motion ?? 'dynamic', position: description.position_m, rotation: description.orientation, linearVelocity: description.linearVelocity_m_s, angularVelocity: description.angularVelocity_rad_s,
      motionLocks: recipe?.character ? { angularX: true, angularY: true, angularZ: true } : {}, enableSleep: !recipe?.character, isBullet: true });
    const body: WorldBody = { entityId, generation: this.generation, description, recipe, native, tag: native.getUserData(), grounded: false, coyote: 0, jumpBuffer: 0, submerged: 0 };
    this.bodies.set(entityId, body); this.tags.set(body.tag, body);
    const material = { density: description.density_kg_m3, friction: description.friction, restitution: description.restitution, isSensor: recipe?.sensor ?? false, enableSensorEvents: true, filter: { categoryBits: recipe?.sensor ? 2 : recipe?.character ? 4 : 1, maskBits: 7 } };
    const keep = (shape: B3Shape) => { if (!shape.isValid()) { shape.delete(); throw new Error(`Invalid collision shape for ${entityId}`); } this.shapeEntities.set(shape.getUserData(), entityId); shape.delete(); };
    const d = description.dimensions_m;
    if (recipe?.colliders) {
      for (const part of recipe.colliders) keep(native.createBox({ ...material, offset: part.offset_m,
        halfExtents: { x: part.dimensions_m.x/2, y: part.dimensions_m.y/2, z: part.dimensions_m.z/2 } }));
    } else if (description.shape === 'box') keep(native.createBox({ ...material, halfExtents: { x: d.x / 2, y: d.y / 2, z: d.z / 2 } }));
    else if (description.shape === 'sphere') keep(native.createSphere({ ...material, radius: d.x }));
    else if (description.shape === 'capsule') keep(native.createCapsule({ ...material, radius: d.x, height: d.y }));
    else if (description.shape === 'cylinder') {
      const points = Array.from({ length: 32 }, (_, i) => ({ x: d.x * Math.cos(i % 16 * Math.PI / 8), y: i < 16 ? -d.y / 2 : d.y / 2, z: d.x * Math.sin(i % 16 * Math.PI / 8) }));
      keep(native.createHull({ ...material, points }));
    } else {
      // Hollow cup: a bottom and a ring of wall segments; never a sealed hull.
      const thickness = cupWallThickness_m(d), inner = d.x - thickness;
      const floor = Array.from({ length: 32 }, (_, i) => ({ x: d.x * Math.cos(i % 16 * Math.PI / 8), y: -d.y / 2 + (i < 16 ? 0 : thickness), z: d.x * Math.sin(i % 16 * Math.PI / 8) }));
      keep(native.createHull({ ...material, points: floor }));
      for (let i = 0; i < 16; i++) {
        const points = [inner, d.x].flatMap(r => [i, i + 1].flatMap(a => [-d.y / 2 + thickness, d.y / 2].map(y => ({ x: r * Math.cos(a * Math.PI / 8), y, z: r * Math.sin(a * Math.PI / 8) }))));
        keep(native.createHull({ ...material, points }));
      }
    }
  }
  validateSolidEdit(scene: SceneDescription) { assertNoSolidOverlap(scene, this.renderBodies().filter(b => b.description.motion !== "static"), this.scene); }
  validatePlacement(entityId: string) {
    const body = this.bodies.get(entityId);
    if (!body || body.recipe?.sensor) return;
    const bindings = this.fluidBindings(false), placed = bindings.filter(b => b.body === body).map(b => b.state);
    assertNoSolidOverlap(this.scene, placed, undefined, bindings.filter(b => b.body !== body).map(b => b.state));
  }
  /** Prepare a replacement before retiring the old native body. Other bodies,
   * contacts, poses and the world clock survive an entity rebuild. */
  configure(entity: EntityRecord, recipe: BodyRecipe | undefined, rebuild: boolean) {
    const old=this.bodies.get(entity.id);
    if (!old && !recipe) return;
    if (!old || !recipe) throw new Error('Changing body ownership requires a reset');
    const description=shapeDescription(entity,recipe);
    const physical=(r:BodyRecipe|undefined)=>r&&({...r,character:r.character?true:undefined});
    if (!rebuild) {
      if(JSON.stringify(physical(old.recipe))!==JSON.stringify(physical(recipe))) throw new Error('This live setting changes physics geometry or material; declare rebuild');
      old.recipe=recipe;old.description=description;return;
    }
    const current={...description,position_m:old.native.getPosition(),orientation:old.native.getRotation(),linearVelocity_m_s:old.native.getLinearVelocity(),angularVelocity_rad_s:old.native.getAngularVelocity()};
    if(!recipe.sensor)assertNoSolidOverlap(this.scene,fluidBodyParts(initializeRigidBody(current),recipe,false),undefined,this.fluidBindings(false).filter(b=>b.body!==old).map(b=>b.state));
    const previousShapes=new Set(this.shapeEntities.keys());
    this.bodies.delete(entity.id);
    try { this.add(entity.id,current,recipe); }
    catch(error) {
      const candidate=this.bodies.get(entity.id);
      if(candidate){this.tags.delete(candidate.tag);candidate.native.destroy();candidate.native.delete();}
      for(const tag of this.shapeEntities.keys())if(!previousShapes.has(tag))this.shapeEntities.delete(tag);
      this.bodies.set(entity.id,old);throw error;
    }
    const next=this.bodies.get(entity.id)!;next.description=description;next.generation=old.generation+1;
    next.submerged=old.submerged;next.grounded=old.grounded;next.coyote=old.coyote;
    if(this.grab?.entityId===entity.id)this.releaseGrab();
    this.tags.delete(old.tag);old.native.destroy();old.native.delete();
    for(const [tag,id] of this.shapeEntities)if(id===entity.id&&previousShapes.has(tag))this.shapeEntities.delete(tag);
    for(const key of this.sensorPairs.keys())if(key.split('/').includes(entity.id))this.sensorPairs.delete(key);
  }
  updateSolids(scene: SceneDescription) {
    const next = compileSolidChunks(scene, [...this.bodies.values()].map(b => b.native.getPosition()), this.collisionCache);
    const staged = new Map<string, { signature: string; body: B3Body }>();
    try {
      for (const chunk of next) {
        if (this.chunks.get(chunk.key)?.signature === chunk.signature) continue;
        // Native hull construction uses floats. Keep small sampled features
        // close to the body's origin instead of baking world offsets into them.
        const first = chunk.shapes[0];
        const origin = first.kind === 'box' ? first.center : first.points[0];
        const local = (p: Vec3) => ({ x: p.x - origin.x, y: p.y - origin.y, z: p.z - origin.z });
        const body = this.world.createBody({ type: 'static', position: origin }); staged.set(chunk.key, { signature: chunk.signature, body });
        for (const shape of chunk.shapes) {
          const handle = shape.kind === 'box' ? body.createBox({ halfExtents: shape.half, offset: local(shape.center), friction: .7 }) : body.createHull({ points: shape.points.map(local), friction: .7 });
          if (!handle.isValid()) { handle.delete(); throw new Error(`Invalid solid collision hull at ${chunk.key}`); }
          handle.delete();
        }
      }
    } catch (error) { for (const { body } of staged.values()) { body.destroy(); body.delete(); } throw error; }
    const keys = new Set(next.map(chunk => chunk.key));
    for (const [key, entry] of this.chunks) if (!keys.has(key) || staged.has(key)) { entry.body.destroy(); entry.body.delete(); this.chunks.delete(key); }
    for (const [key, entry] of staged) this.chunks.set(key, entry);
    this.scene = scene;
    this.coverage = new Map([...this.bodies].map(([id, b]) => [id, b.native.getPosition()]));
  }
  ensureCoverage() {
    // Finite authored solids were all compiled at admission. Only procedural
    // ground needs a moving neighbourhood; rebuilding the whole pool every
    // metre otherwise stalls steering and climbing.
    if (!fluidSolidWorldForScene(this.scene).ground) return;
    if ([...this.bodies].some(([id, b]) => { const p = b.native.getPosition(), old = this.coverage.get(id); return !old || Math.hypot(p.x-old.x,p.y-old.y,p.z-old.z) > 1; })) this.updateSolids(this.scene);
  }
  ray(origin: Vec3, direction: Vec3, distance: number, maskBits = 5): { entityId?: string; point: Vec3; normal: Vec3; distance: number } | undefined {
    const ray: B3Ray = this.world.castRayClosest(origin, { x: direction.x * distance, y: direction.y * distance, z: direction.z * distance }, { maskBits });
    ray.shape?.delete();
    return ray.hit ? { entityId: this.tags.get(ray.bodyUserData!)?.entityId, point: ray.point!, normal: ray.normal!, distance: ray.fraction! * distance } : undefined;
  }
  releaseGrab() { if (this.grab) { this.grab.joint.destroy(); this.grab.joint.delete(); this.bodies.get(this.grab.entityId)?.native.applyLinearImpulseToCenter(zero(), true); this.grab.anchor.destroy(); this.grab.anchor.delete(); this.grab = undefined; } }
  command(command: PhysicsCommand, dt: number) {
    if (command.kind === 'grab') {
      if (!command.active) { if (this.grab?.entityId === command.entityId) this.releaseGrab(); return; }
      const body = this.bodies.get(command.entityId);
      if (!body || body.recipe?.motion !== 'dynamic' || body.recipe.character) throw new Error('Body cannot be grabbed');
      if (this.grab?.entityId !== command.entityId) {
        this.releaseGrab();
        const anchor = this.world.createBody({ type: 'kinematic', position: body.native.getPosition() });
        try { this.grab = { entityId: command.entityId, anchor, joint: this.world.createMotorJoint(anchor, body.native, { linearHertz: 5, linearDampingRatio: 1, maxSpringForce: body.native.getMass() * 40, maxSpringTorque: 0, maxVelocityForce: 0, maxVelocityTorque: 0 }) }; }
        catch (error) { anchor.destroy(); anchor.delete(); throw error; }
      }
      this.grab!.anchor.setTargetTransform({ position: command.target_m, rotation: body.native.getRotation() }, dt, true);
      return;
    }
    if (command.kind === 'character-move') { const body = this.bodies.get(command.entityId); if (body?.recipe?.character) this.moveCharacter(body, command.velocity_m_s, command.jump, dt, command.somersault, command.climb, command.turn_rad_s); return; }
    const body = [...this.bodies.values()].find(b => b.description.id === command.bodyId);
    if (!body || command.bodyGeneration !== body.generation) throw new Error('Stale physics body command');
    if (command.kind === 'impulse') { body.native.applyLinearImpulseToCenter(command.linear_N_s, true); body.native.applyAngularImpulse(command.angular_N_m_s, true); }
    else body.native.setTargetTransform({ position: command.transform.position_m, rotation: command.transform.orientation }, dt, true);
  }
  private facing(body: WorldBody) { return body.facing ?? 2 * Math.atan2(body.description.orientation.y, body.description.orientation.w); }
  private moveCharacter(body: WorldBody, target: Vec3, jump: boolean, dt: number, somersault = false, climb = false, turn?: number) {
    const motor = body.recipe!.character!, pos = body.native.getPosition(), velocity = body.native.getLinearVelocity();
    const d = body.description.dimensions_m, height = d.y + 2 * d.x;
    const hit = this.ray(pos, { x: 0, y: -1, z: 0 }, height / 2 + .12, 1);
    body.grounded = Boolean(hit && hit.normal.y >= Math.cos(motor.maxSlope * Math.PI / 180) && velocity.y < 1);
    body.coyote = body.grounded ? .1 : Math.max(0, body.coyote - dt);
    body.jumpBuffer = jump ? .12 : Math.max(0, body.jumpBuffer - dt);
    if (body.grounded) body.flipped = false;
    if ((body.flipTime ?? 0) > 0) body.flipTime = Math.max(0, body.flipTime! - dt);
    if (turn !== undefined) {
      // A body-frame command: turn the facing, then carry the velocity into the world.
      body.facing = this.facing(body) + turn * dt;
      const c = Math.cos(body.facing), s = Math.sin(body.facing);
      target = { x: target.x * c + target.z * s, y: target.y, z: target.z * c - target.x * s };
    }
    const targetSpeed = Math.hypot(target.x, target.z);
    if (turn === undefined && targetSpeed > .05 && !body.flipTime) {
      const desired = Math.atan2(target.x, target.z), current = body.facing ?? desired;
      body.facing = current + Math.atan2(Math.sin(desired - current), Math.cos(desired - current)) * Math.min(1, dt * 14);
    }
    // Hold jump beside an authored ladder to climb its outside face, then
    // step onto the landing after the capsule's feet clear the board. Native
    // collision still decides the resulting pose; there are no teleports.
    if (!climb || somersault) body.climbing = undefined;
    const footY = pos.y - height / 2;
    const ladders = this.scene.interaction?.ladders ?? [];
    const ladder = ladders.find(l => l.id === body.climbing) ?? (jump ? ladders.find(l =>
      Math.hypot(pos.x - l.bottom_m.x, pos.z - l.bottom_m.z) < l.width_m / 2 + d.x + .25
      && footY >= l.bottom_m.y - .15 && footY < l.bottom_m.y + .65) : undefined);
    if (ladder && climb && !somersault) {
      body.climbing = ladder.id;
      const bottom = ladder.bottom_m, top = ladder.top_m;
      const length = Math.max(.001, Math.hypot(top.x - bottom.x, top.z - bottom.z));
      const dx = (top.x - bottom.x) / length, dz = (top.z - bottom.z) / length;
      const t = Math.max(0, Math.min(1, (footY - bottom.y) / (top.y - bottom.y)));
      const cleared = footY >= top.y + .12;
      const offset = cleared ? 0 : d.x + .16;
      const goal = { x: bottom.x + (top.x - bottom.x) * t - dx * offset,
        y: top.y + height / 2 + .24, z: bottom.z + (top.z - bottom.z) * t - dz * offset };
      const vx = Math.max(-1.5, Math.min(1.5, (goal.x - pos.x) * 8));
      const vz = Math.max(-1.5, Math.min(1.5, (goal.z - pos.z) * 8));
      const vy = Math.min(2.4, (goal.y - pos.y) * 8);
      const mass = body.native.getMass();
      body.native.applyLinearImpulseToCenter({ x: (vx - velocity.x) * mass,
        y: (vy - velocity.y - this.scene.fluid.gravity_m_s2.y * dt) * mass,
        z: (vz - velocity.z) * mass }, true);
      body.facing = Math.atan2(dx, dz); body.jumpBuffer = 0; body.coyote = 0;
      if (cleared && Math.hypot(pos.x - top.x, pos.z - top.z) < .08) body.climbing = undefined;
      return;
    }
    const support = hit?.entityId ? this.bodies.get(hit.entityId)?.native.getLinearVelocity() : undefined;
    const swimming = body.submerged > .6;
    const wetSpeed = 1 - .45 * body.submerged;
    let tx = target.x * wetSpeed, tz = target.z * wetSpeed;
    if (hit && hit.normal.y > 0 && hit.normal.y < Math.cos(motor.maxSlope * Math.PI / 180)) {
      const into = tx * hit.normal.x + tz * hit.normal.z, n2 = hit.normal.x ** 2 + hit.normal.z ** 2;
      if (into < 0 && n2 > 1e-6) { tx -= into * hit.normal.x / n2; tz -= into * hit.normal.z / n2; }
    }
    const maxChange = motor.acceleration * dt * (swimming ? .45 : body.grounded ? 1 : .3);
    const dx = tx + (support?.x ?? 0) - velocity.x;
    const dz = tz + (support?.z ?? 0) - velocity.z;
    const scale = Math.min(1, maxChange / Math.max(1e-9, Math.hypot(dx, dz)));
    const mass = body.native.getMass();
    let dy = 0;
    if (somersault && !body.flipped && !body.flipTime && !swimming) {
      body.flipTime = .8; body.flipped = true;
      if (body.coyote > 0) body.jumpBuffer = .12;
    }
    if (body.jumpBuffer > 0 && (body.coyote > 0 || swimming)) { dy = Math.sqrt(2 * Math.abs(this.scene.fluid.gravity_m_s2.y) * motor.jumpHeight) + (support?.y ?? 0) - velocity.y; body.jumpBuffer = 0; body.coyote = 0; }
    // Small step-up, only when grounded and the upper ray has clearance.
    const speed = Math.hypot(target.x, target.z);
    if (body.grounded && speed > .01 && motor.stepHeight > 0 && dy === 0) {
      const dir = { x: target.x / speed, y: 0, z: target.z / speed }, foot = pos.y - height / 2;
      const lower = this.ray({ ...pos, y: foot + .03 }, dir, d.x + speed * dt, 1);
      const upper = this.ray({ ...pos, y: foot + motor.stepHeight + .03 }, dir, d.x + speed * dt, 1);
      if (lower && lower.normal.y < .5 && !upper) {
        const down = this.ray({ x: pos.x + dir.x * (d.x + speed * dt), y: foot + motor.stepHeight + .03, z: pos.z + dir.z * (d.x + speed * dt) }, { x: 0, y: -1, z: 0 }, motor.stepHeight, 1);
        const ceiling = this.ray(pos, { x: 0, y: 1, z: 0 }, height / 2 + motor.stepHeight, 1);
        if (down && down.normal.y >= Math.cos(motor.maxSlope * Math.PI / 180) && !ceiling) body.native.setTransform({ ...pos, y: down.point.y + height / 2 + .01 }, body.native.getRotation());
      }
    }
    body.native.applyLinearImpulseToCenter({ x: dx * scale * mass, y: dy * mass, z: dz * scale * mass }, true);
  }
  applyFluid(exchanges: readonly FluidExchange[], identity: TickIdentity) {
    if (identity.tick <= this.lastFluidTick) throw new Error("Fluid exchange already applied");
    const bindings = this.fluidBindings(), byId = new Map(bindings.map(b => [b.state.description.id, b]));
    const validated: { body: WorldBody; state: RigidBodyState; packet: FluidExchange }[] = [];
    const seen = new Set<string>();
    for (const packet of exchanges) {
      if (packet.generation !== identity.generation || packet.tick !== identity.tick || packet.solidRevision !== identity.solidRevision || seen.has(packet.bodyId)) throw new Error('Stale or duplicate fluid exchange');
      seen.add(packet.bodyId);
      const binding = byId.get(packet.bodyId);
      if (!binding || binding.body.generation !== packet.bodyGeneration) throw new Error('Fluid exchange references a stale body');
      const numbers = [...Object.values(packet.linearImpulse_N_s), ...Object.values(packet.angularImpulse_N_m_s), packet.displacedVolume_m3, ...Object.values(packet.fluidVelocity_m_s)];
      if (!numbers.every(Number.isFinite) || packet.displacedVolume_m3 < 0) throw new Error('Invalid fluid exchange');
      validated.push({ ...binding, packet });
    }
    if (seen.size !== bindings.length || bindings.some(b => !seen.has(b.state.description.id))) throw new Error('Incomplete fluid exchange roster');
    const totals = new Map<WorldBody, { linear: Vec3; angular: Vec3; volume: number; materialVolume: number }>();
    for (const {body, state, packet} of validated) {
      let sum = totals.get(body);
      if (!sum) totals.set(body, sum = { linear: zero(), angular: zero(), volume: 0, materialVolume: 0 });
      const center = body.native.getWorldCenterOfMass(), p = state.position_m;
      const r = { x: p.x-center.x, y: p.y-center.y, z: p.z-center.z }, j = packet.linearImpulse_N_s;
      sum.linear.x += j.x; sum.linear.y += j.y; sum.linear.z += j.z;
      // GPU pressure torque is about the panel centre. Transfer it to the native COM.
      sum.angular.x += packet.angularImpulse_N_m_s.x + r.y*j.z-r.z*j.y;
      sum.angular.y += packet.angularImpulse_N_m_s.y + r.z*j.x-r.x*j.z;
      sum.angular.z += packet.angularImpulse_N_m_s.z + r.x*j.y-r.y*j.x;
      sum.volume += packet.displacedVolume_m3;
      sum.materialVolume += primitiveVolume(state.description.shape, state.description.dimensions_m);
    }
    for (const [body, sum] of totals) {
      body.submerged = Math.min(1, sum.volume / Math.max(1e-9, sum.materialVolume));
      if ((body.recipe?.motion ?? body.description.motion ?? 'dynamic') === 'dynamic') {
        body.native.applyLinearImpulseToCenter(sum.linear, true);
        body.native.applyAngularImpulse(sum.angular, true);
      }
    }
    this.lastFluidTick = identity.tick;
  }
  step(dt: number): SensorContact[] {
    if (this.disposed) throw new Error('Physics world disposed');
    if (!(dt > 0 && dt <= .1)) throw new Error('Interactive step must be in (0, 0.1] seconds');
    this.world.step(dt, 4);
    const events = this.world.getSensorEvents();
    return ([['begin', true], ['end', false]] as const).flatMap(([name, entered]) => events[name].flatMap(e => {
      const sensorEntityId = this.shapeEntities.get(e.sensorUserData!), visitorEntityId = this.shapeEntities.get(e.visitorUserData!);
      if (!sensorEntityId || !visitorEntityId) return [];
      const key = `${sensorEntityId}/${visitorEntityId}`, shapeKey = `${e.sensorUserData}/${e.visitorUserData}`;
      const pairs = this.sensorPairs.get(key) ?? new Set<string>(), before = pairs.size > 0;
      if (entered) pairs.add(shapeKey); else pairs.delete(shapeKey);
      if (pairs.size) this.sensorPairs.set(key, pairs); else this.sensorPairs.delete(key);
      return before !== (pairs.size > 0) ? [{ sensorEntityId, visitorEntityId, entered: pairs.size > 0 }] : [];
    }));
  }
  private bodyState(body: WorldBody): RigidBodyState {
      const state = initializeRigidBody(body.description);
      state.position_m = body.native.getPosition(); state.orientation = body.native.getRotation();
      state.linearVelocity_m_s = body.native.getLinearVelocity(); state.angularVelocity_rad_s = body.native.getAngularVelocity();
      const center = body.native.getWorldCenterOfMass(), p = state.position_m, w = state.angularVelocity_rad_s;
      const r = { x: p.x-center.x, y: p.y-center.y, z: p.z-center.z };
      state.linearVelocity_m_s = { x: state.linearVelocity_m_s.x+w.y*r.z-w.z*r.y,
        y: state.linearVelocity_m_s.y+w.z*r.x-w.x*r.z, z: state.linearVelocity_m_s.z+w.x*r.y-w.y*r.x };
      if (body.recipe?.character) {
        const yaw = this.facing(body);
        const pitch = body.flipTime ? (1 - body.flipTime / .8) * Math.PI * 2 : 0;
        state.renderOrientation = quaternionMultiply({ w: Math.cos(yaw / 2), x: 0, y: Math.sin(yaw / 2), z: 0 },
          { w: Math.cos(pitch / 2), x: Math.sin(pitch / 2), y: 0, z: 0 });
      }
      return state;
  }
  private fluidBindings(boundary = true) {
    return [...this.bodies.values()].filter(body => !body.recipe?.sensor)
      .flatMap(body => fluidBodyParts(this.bodyState(body), body.recipe, boundary).map(state => ({ body, state })));
  }
  renderBodies(): RigidBodyState[] { return this.fluidBindings().map(b => b.state); }
  fluidBodyGenerations(): Readonly<Record<string, number>> {
    return Object.fromEntries(this.fluidBindings().map(b => [b.state.description.id, b.body.generation]));
  }
  snapshots(): BodySnapshot[] {
    return [...this.bodies.values()].map(body => {
      const position = body.native.getPosition(), radius = boundingRadius(body.description);
      const value: BodySnapshot = { entityId: body.entityId, bodyId: body.description.id, bodyGeneration: body.generation,
        authority: body.recipe?.character ? 'character' : body.recipe?.motion === 'kinematic' ? 'kinematic' : (body.recipe?.motion ?? body.description.motion) === 'static' ? 'static' : 'dynamic',
        // A character's capsule never rotates; its orientation is the way it faces.
        transform: { position_m: position, orientation: body.recipe?.character ? { w: Math.cos(this.facing(body) / 2), x: 0, y: Math.sin(this.facing(body) / 2), z: 0 } : body.native.getRotation() }, linearVelocity_m_s: body.native.getLinearVelocity(), angularVelocity_rad_s: body.native.getAngularVelocity(), shapeId: body.description.shape,
        grounded: body.grounded, submergedFraction: body.submerged,
        bounds_m: { min: { x: position.x - radius, y: position.y - radius, z: position.z - radius }, max: { x: position.x + radius, y: position.y + radius, z: position.z + radius } } };
      assertFiniteSnapshot(value); return value;
    });
  }
  destroy() {
    if (this.disposed) return; this.disposed = true;
    this.releaseGrab();
    this.world.destroy();
    for (const body of this.bodies.values()) body.native.delete();
    for (const chunk of this.chunks.values()) chunk.body.delete();
    this.world.delete(); this.bodies.clear(); this.tags.clear(); this.shapeEntities.clear(); this.sensorPairs.clear(); this.chunks.clear();
  }
}
