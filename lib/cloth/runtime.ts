import { createClothState, point3, prepareClothTopology, validateClothDocument, type PreparedClothTopology } from "./topology";
import { finishClothMechanicalStep, nonFluidClothPredictor, stepCloth, type ClothStepOptions, type ClothStepReport } from "./dynamics";
import type { ClothCollider } from "./contact";
import { SourceMotionRuntime, type SourceMotionCheckpoint } from "./source-motion";
import { sourceClothColliders } from "./source-contact";
import { GPUClothMechanics } from "./gpu-mechanics";
import { solveClothIncrement } from "./nonlinear-solve";
import type { WetClothSourceField } from "./source-types";
import type { ClothDocument, ClothFrameIdentity, ClothState, ClothSurfacePublication, ClothTopologyData, ClothVector } from "./types";
export interface ClothRuntimeOptions {
    readonly sourceFields?: readonly WetClothSourceField[];
    readonly sourceOffset_m?: ClothVector;
    readonly mechanics?: ClothStepOptions;
    readonly excludedSourceColliderIndices?:readonly number[];
    /** CPU Float64 owner of the incremental-potential minimization: global
     * Newton-PCG (lib/cloth/nonlinear-solve.ts) or coordinate Gauss-Seidel. */
    readonly nonlinearSolver?: 'newton-pcg' | 'gauss-seidel';
}
export interface ClothRuntimeCheckpoint {
    readonly state: ClothState;
    readonly motion: SourceMotionCheckpoint;
    readonly tick: number;
    readonly pendingImpulse?: Float64Array;
    readonly pendingIdentity?: ClothFrameIdentity;
}
let nextClothRuntimeGeneration = 1;
/** Transactional CPU reference host. Fluid impulses are consumed once, and a
 * failed nonlinear solve restores both physical state and prescribed motion. */
export class ClothRuntime {
    readonly prepared: PreparedClothTopology;
    readonly state: ClothState;
    readonly motion: SourceMotionRuntime;
    private tick = 0;
    private gpu?: GPUClothMechanics;
    private stepping = false;
    private excludedSourceColliderIndices:readonly number[]=[];
    private staged?: {dt_s:number; checkpoint:ClothRuntimeCheckpoint; previousFields:readonly WetClothSourceField[]};
    private pendingImpulse?: Float64Array;
    private pendingIdentity?: ClothFrameIdentity;
    constructor(readonly document: ClothDocument, data?: ClothTopologyData, private readonly options: ClothRuntimeOptions = {}) {
        const errors = validateClothDocument(document);
        if (errors.length)
            throw new Error(errors.join("; "));
        const topology = data ?? (document.geometry.kind === "inline" ? document.geometry.data : undefined);
        if (!topology)
            throw new Error(`Cloth asset must be resolved before creating runtime: ${document.id}`);
        if (document.geometry.kind === "asset" && topology.restPositions_m.length !== document.geometry.vertexCount * 3)
            throw new Error("Cloth asset vertex count mismatch");
        this.prepared = prepareClothTopology(topology, document.materials);
        this.state = createClothState(this.prepared);
        this.state.generation = nextClothRuntimeGeneration++;
        if(document.physics?.varyingSolidFraction===false){this.state.controlVolume_m3.set(this.state.restControlVolume_m3);for(let i=0;i<this.prepared.vertexCount;i++)this.state.solidFraction[i]=this.state.solidVolume_m3[i]!/this.state.restControlVolume_m3[i]!;}
        this.motion = new SourceMotionRuntime(document.scripts ?? [], topology, options.sourceFields, options.sourceOffset_m);
        this.excludedSourceColliderIndices=[...(options.excludedSourceColliderIndices??[])];
    }
    get identity(): ClothFrameIdentity { return { generation: this.state.generation, topologyRevision: this.state.topologyRevision, tick: this.tick, time_s: this.state.time_s }; }
    setExcludedSourceColliderIndices(indices:readonly number[]):void{this.excludedSourceColliderIndices=[...indices];}
    private matches(identity: ClothFrameIdentity): boolean {
        const current = this.identity;
        return identity.generation === current.generation && identity.topologyRevision === current.topologyRevision && identity.tick === current.tick && Math.abs(identity.time_s - current.time_s) < 1e-12;
    }
    acceptFluidImpulse(impulse_N_s: ArrayLike<number>, identity: ClothFrameIdentity = this.identity): void {
        if (this.stepping)
            throw new Error("Cannot replace cloth fluid impulse during a step");
        if (!this.matches(identity))
            throw new Error("Stale cloth fluid exchange");
        if (this.pendingImpulse)
            throw new Error("Cloth fluid impulse already pending for this step");
        if (impulse_N_s.length !== this.state.positions_m.length)
            throw new Error("Cloth fluid impulse length mismatch");
        const copy = Float64Array.from(impulse_N_s);
        if (!copy.every(Number.isFinite))
            throw new Error("Nonfinite cloth fluid impulse");
        this.pendingImpulse = copy;
        this.pendingIdentity = { ...identity };
    }
    checkpoint(): ClothRuntimeCheckpoint {
        return { state: structuredClone(this.state), motion: this.motion.checkpoint(), tick: this.tick,
            ...(this.pendingImpulse ? { pendingImpulse: this.pendingImpulse.slice() } : {}), ...(this.pendingIdentity ? { pendingIdentity: { ...this.pendingIdentity } } : {}) };
    }
    restore(snapshot: ClothRuntimeCheckpoint): void {
        this.staged=undefined;
        if (snapshot.state.positions_m.length !== this.state.positions_m.length || snapshot.state.twist_rad.length !== this.state.twist_rad.length)
            throw new Error("Incompatible cloth checkpoint");
        for (const key of Object.keys(this.state) as (keyof ClothState)[]) {
            const value = this.state[key];
            if (value instanceof Float64Array)
                value.set(snapshot.state[key] as Float64Array);
        }
        this.state.time_s = snapshot.state.time_s;
        this.state.generation = snapshot.state.generation;
        this.state.topologyRevision = snapshot.state.topologyRevision;
        this.motion.restore(snapshot.motion);
        this.tick = snapshot.tick;
        this.pendingImpulse = snapshot.pendingImpulse?.slice();
        this.pendingIdentity = snapshot.pendingIdentity ? { ...snapshot.pendingIdentity } : undefined;
    }
    step(dt_s: number, colliders: readonly ClothCollider[] = [], options: ClothStepOptions = {}): ClothStepReport {
        if (!(dt_s > 0 && Number.isFinite(dt_s)))
            throw new Error("Cloth timestep must be positive and finite");
        if (this.stepping)
            throw new Error("Cloth step already in flight");
        const checkpoint = this.staged?.checkpoint ?? this.checkpoint();
        try {
            const prepared = this.prepareStep(dt_s, colliders, options);
            const report = this.document.physics?.solveSolid===false ? this.stepPrescribed(dt_s,prepared) : this.options.nonlinearSolver==='newton-pcg' ? solveClothIncrement(this.prepared, this.state, dt_s, prepared) : stepCloth(this.prepared, this.state, dt_s, prepared);
            if (!report.converged)
                throw new Error(`Cloth nonlinear solve did not converge (${report.maximumUpdate_m} m after ${report.iterations} iterations)`);
            this.pendingImpulse = undefined;
            this.pendingIdentity = undefined;
            this.staged=undefined;
            this.tick++;
            return report;
        }
        catch (error) {
            this.restore(checkpoint);
            throw error;
        }
    }
    private prepareStep(dt_s: number, colliders: readonly ClothCollider[], options: ClothStepOptions): ClothStepOptions {
        if (this.pendingImpulse) {
            if (!this.pendingIdentity || !this.matches(this.pendingIdentity))
                throw new Error("Stale pending cloth fluid exchange");
            for (let i = 0; i < this.prepared.vertexCount; i++) {
                if (this.document.physics?.solveSolid===false || (this.prepared.kinematicFixed[i] ?? 0) & 1)
                    continue;
                const density = this.prepared.materials[this.prepared.vertexMaterial[i]!]!.waterDensity_kg_m3;
                const mass = this.state.dryMass_kg[i]! + density * this.state.waterVolume_m3[i]!;
                for (let a = 0; a < 3; a++)
                    this.state.velocities_m_s[3 * i + a]! += this.pendingImpulse[3 * i + a]! / mass;
            }
        }
        if(this.staged&&this.staged.dt_s!==dt_s)throw new Error('Cloth predictor timestep mismatch');
        const previousFields=this.staged?.previousFields??this.motion.fields;
        if(!this.staged)this.advanceSupports(dt_s);
        return { iterations: 64, tolerance_m: 1e-8, selfCollision: true, varyingSolidFraction:this.document.physics?.varyingSolidFraction, ...this.options.mechanics, ...options, colliders: [...colliders, ...sourceClothColliders(previousFields, this.motion.fields, dt_s,this.excludedSourceColliderIndices)], supportPositions_m: options.supportPositions_m ?? this.motion.supportPositions_m };
    }
    /** Pure accepted-state prediction, including a native temporal half-kick. */
    predictAcceptedNonFluidVelocity(dt_s:number,options:ClothStepOptions={}):Float64Array {
        if(!(dt_s>0&&Number.isFinite(dt_s)))throw new Error('Invalid cloth predictor timestep');
        return this.document.physics?.solveSolid===false?new Float64Array(this.state.velocities_m_s.length):nonFluidClothPredictor(this.prepared,this.state,dt_s,{...this.options.mechanics,...options,supportPositions_m:options.supportPositions_m??this.motion.supportPositions_m});
    }
    /** Advance prescribed targets once without integrating physical state.
     * The returned velocity is temporary fluid-predictor input only. */
    prepareNonFluidPredictor(dt_s:number,options:ClothStepOptions={}):Float64Array {
        if(this.stepping||this.staged||this.pendingImpulse)throw new Error('Cloth predictor requires an unstaged accepted step');
        if(!(dt_s>0&&Number.isFinite(dt_s)))throw new Error('Invalid cloth predictor timestep');
        const checkpoint=this.checkpoint(),previousFields=this.motion.fields;
        try{
            this.advanceSupports(dt_s);
            const velocity=this.document.physics?.solveSolid===false?new Float64Array(this.state.velocities_m_s.length):nonFluidClothPredictor(this.prepared,this.state,dt_s,{...this.options.mechanics,...options,supportPositions_m:options.supportPositions_m??this.motion.supportPositions_m});
            this.staged={dt_s,checkpoint,previousFields};return velocity;
        }catch(error){this.restore(checkpoint);throw error;}
    }
    private advanceSupports(dt_s:number):void {
        this.motion.step(dt_s, this.state.time_s);
        // Source soft fixed bits drive finite spring targets. Twist uses the
        // pinned source's unsigned projected-quaternion AngleAxis angle.
        for(const rod of this.prepared.rods)for(let j=0;j<rod.restLengths_m.length;j++){
            const segment=rod.segmentOffset+j,vertex=rod.vertices[j]!;
            if(this.prepared.attachmentTwist_N_m[segment] === 0)continue;
            const q=this.motion.groupRotationIncrement(this.prepared.data.groups?.[vertex]??0),dir=[0,0,0];
            for(const index of this.prepared.incidentRods[vertex]!) {const adjacent=this.prepared.rods[index]!;for(let k=0;k<adjacent.restLengths_m.length;k++)if(adjacent.vertices[k]===vertex||adjacent.vertices[k+1]===vertex)for(let a=0;a<3;a++)dir[a]!+=this.motion.supportPositions_m[3*adjacent.vertices[k+1]!+a]!-this.motion.supportPositions_m[3*adjacent.vertices[k]!+a]!;}
            const length=Math.hypot(...dir);if(!(length>0))throw new Error('Source attachment twist has an undefined rest direction');
            const projected=(q[0]*dir[0]!+q[1]*dir[1]!+q[2]*dir[2]!)/length;
            if(Math.hypot(projected,q[3])<1e-14)throw new Error('Source attachment twist projection is undefined');
            this.state.attachmentTwistTarget_rad[segment]!+=2*Math.atan2(Math.abs(projected),Math.abs(q[3]));
        }
        // A clamped material director follows its support rotation, including a
        // rotation around a straight rod's tangent (which positions cannot encode).
        for (const rod of this.prepared.rods)
            for (let j = 0; j < rod.restLengths_m.length; j++) {
                const a = rod.vertices[j]!, b = rod.vertices[j + 1]!;
                const fixed = ((this.prepared.kinematicFixed[a] ?? 0) & 2) ? a : ((this.prepared.kinematicFixed[b] ?? 0) & 2) ? b : -1;
                if (fixed < 0)
                    continue;
                const q = this.motion.groupRotationIncrement(this.prepared.data.groups?.[fixed] ?? 0), s = rod.segmentOffset + j;
                for (const frame of [this.state.rodDirectors, this.state.rodTangents]) {
                    const v = point3(frame, s), t = [2 * (q[1] * v[2] - q[2] * v[1]), 2 * (q[2] * v[0] - q[0] * v[2]), 2 * (q[0] * v[1] - q[1] * v[0])];
                    frame.set([v[0] + q[3] * t[0]! + q[1] * t[2]! - q[2] * t[1]!, v[1] + q[3] * t[1]! + q[2] * t[0]! - q[0] * t[2]!, v[2] + q[3] * t[2]! + q[0] * t[1]! - q[1] * t[0]!], 3 * s);
                }
            }
    }
    private stepPrescribed(dt:number,options:ClothStepOptions):ClothStepReport {
        const old=this.state.positions_m.slice(),twist=this.state.twist_rad.slice(),target=options.supportPositions_m!;
        for(let i=0;i<this.prepared.vertexCount;i++)if((this.prepared.kinematicFixed[i]??0)&1)for(let a=0;a<3;a++)this.state.positions_m[3*i+a]=target[3*i+a]!;
        return finishClothMechanicalStep(this.prepared,this.state,old,twist,dt,{...options,cohesionPairs:[],colliders:[],selfCollision:false},{iterations:0,converged:true,maximumUpdate_m:0});
    }
    async initializeGPU(device: GPUDevice): Promise<void> {
        if(this.document.physics?.solveSolid===false)return;
        if (this.gpu)
            return;
        const solver = new GPUClothMechanics(device, this.prepared);
        try {
            await solver.initialize();
            this.gpu = solver;
        }
        catch (error) {
            solver.destroy();
            throw error;
        }
    }
    async stepAsync(dt_s: number, colliders: readonly ClothCollider[] = [], options: ClothStepOptions = {}): Promise<ClothStepReport> {
        if (!this.gpu)
            return this.step(dt_s, colliders, options);
        if (this.stepping)
            throw new Error("Cloth step already in flight");
        if (!(dt_s > 0 && Number.isFinite(dt_s)))
            throw new Error("Cloth timestep must be positive and finite");
        this.stepping = true;
        const checkpoint = this.staged?.checkpoint ?? this.checkpoint();
        try {
            const report = await this.gpu.step(this.state, dt_s, this.prepareStep(dt_s, colliders, options));
            if (!report.converged)
                throw new Error(`Cloth GPU nonlinear solve did not converge (${report.maximumUpdate_m} m after ${report.iterations} iterations)`);
            this.pendingImpulse = undefined;
            this.pendingIdentity = undefined;
            this.staged=undefined;
            this.tick++;
            return report;
        }
        catch (error) {
            this.restore(checkpoint);
            throw error;
        }
        finally {
            this.stepping = false;
        }
    }
    dispose(): void { this.gpu?.destroy(); this.gpu = undefined; }
    publish(tick = this.tick): ClothSurfacePublication {
        if (!Number.isInteger(tick) || tick < 0)
            throw new Error("Cloth publication tick must be a nonnegative integer");
        const saturation = Float32Array.from(this.state.waterVolume_m3, (water, i) => {
            const capacity = this.state.controlVolume_m3[i]! * (1 - this.state.solidFraction[i]!);
            return capacity > 0 ? Math.min(1, Math.max(0, water / capacity)) : 0;
        });
        return Object.freeze({ ...this.identity, tick, documentId: this.document.id, topology: this.prepared.data,
            positions_m: this.state.positions_m.slice(), supportPositions_m:Float64Array.from(this.motion.supportPositions_m), attachmentTwistTarget_rad:this.state.attachmentTwistTarget_rad.slice(), velocities_m_s: this.state.velocities_m_s.slice(), waterVolume_m3: this.state.waterVolume_m3.slice(),
            controlVolume_m3: this.state.controlVolume_m3.slice(), solidFraction: this.state.solidFraction.slice(), saturation,
            rodTwist_rad:this.state.twist_rad.slice(),rodReferenceTwist_rad:this.state.rodReferenceTwist_rad.slice(),rodAngularVelocity_rad_s:this.state.angularVelocity_rad_s.slice(),rodDirectors:this.state.rodDirectors.slice(),rodTangents:this.state.rodTangents.slice() });
    }
}
export function createClothRuntime(document: ClothDocument, data?: ClothTopologyData, options?: ClothRuntimeOptions): ClothRuntime { return new ClothRuntime(document, data, options); }
