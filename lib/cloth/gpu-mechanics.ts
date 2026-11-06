import type { GPUInitializationTask } from '../core/gpu-initialization';
import { bendingRigidity, membraneModulus, rodRigidity, rodSegmentInertia_kg_m2 } from './material';
import { extendClothStepReport, finishClothMechanicalStep, type ClothStepOptions, type ClothStepReport } from './dynamics';
import { point3, type PreparedClothTopology } from './topology';
import type { ClothState } from './types';
/** GPU stores increments, not absolute updated positions. Small physical dt
 * then retains motion below a world-coordinate float32 ULP. Each color is an
 * independent set of all elastic energy stencils, including bending/twist. */
export const clothMechanicsWGSL = /* wgsl */ `
struct Params { step:vec4f, bases:vec4u, work:vec4u, sizes:vec4u }
@group(0) @binding(0) var<storage,read_write> delta:array<vec4f>;
@group(0) @binding(1) var<storage,read> old:array<vec4f>;
@group(0) @binding(2) var<storage,read> prediction:array<vec4f>;
@group(0) @binding(3) var<storage,read> data:array<f32>;
@group(0) @binding(4) var<storage,read> adjacency:array<u32>;
@group(0) @binding(5) var<storage,read_write> angles:array<vec4f>;
@group(0) @binding(6) var<uniform> params:Params;
@group(0) @binding(7) var<storage,read_write> update:atomic<u32>;
fn ids(at:u32)->vec3u{return vec3u(u32(data[at]),u32(data[at+1u]),u32(data[at+2u]));}
fn vec(at:u32)->vec3f{return vec3f(data[at],data[at+1u],data[at+2u]);}
fn displacement(i:u32,v:u32,x:vec3f,previous:bool)->vec3f{if(previous){return vec3f(0);}if(i==v){return x;}return delta[i].xyz;}
fn edge(a:u32,b:u32,v:u32,x:vec3f,previous:bool)->vec3f{return ((old[b].xyz-old[a].xyz)+(vec(params.sizes.w+3u*b)-vec(params.sizes.w+3u*a)))+(displacement(b,v,x,previous)-displacement(a,v,x,previous));}
fn wrap(a:f32)->f32{return atan2(sin(a),cos(a));}
fn strain(f:u32,v:u32,x:vec3f,previous:bool)->vec3f{
 let at=params.bases.x+f*16u;let id=ids(at);let e=edge(id.x,id.y,v,x,previous);let d=edge(id.x,id.z,v,x,previous);
 let u=e*data[at+5u]+d*data[at+6u];let w=e*data[at+9u]+d*data[at+10u];
 return vec3f(.5*(dot(u,u)-1.0),.5*(dot(w,w)-1.0),dot(u,w));
}
fn quadratic(e:vec3f,nu:f32)->f32{return e.x*e.x+e.y*e.y+2.0*nu*e.x*e.y+.5*(1.0-nu)*e.z*e.z;}
fn membraneEnergy(f:u32,v:u32,x:vec3f)->f32{let at=params.bases.x+f*16u;let e=strain(f,v,x,false);let de=e-strain(f,v,x,true);return .5*data[at+3u]*(quadratic(e,data[at+7u])+data[at+11u]*quadratic(de,data[at+7u]));}
fn membraneDerivative(f:u32,v:u32,x:vec3f,axis:u32)->vec2f{
 let at=params.bases.x+f*16u;let id=ids(at);var j=0u;if(id.y==v){j=1u;}if(id.z==v){j=2u;}
 let ru=data[at+4u+j];let rv=data[at+8u+j];let e=strain(f,v,x,false);let stress=e+data[at+11u]*(e-strain(f,v,x,true));
 let a=edge(id.x,id.y,v,x,false);let b=edge(id.x,id.z,v,x,false);let u=a*data[at+5u]+b*data[at+6u];let w=a*data[at+9u]+b*data[at+10u];
 let d=vec3f(ru*u[axis],rv*w[axis],ru*w[axis]+rv*u[axis]);let nu=data[at+7u];let s=vec3f(stress.x+nu*stress.y,stress.y+nu*stress.x,.5*(1.0-nu)*stress.z);
 return data[at+3u]*vec2f(dot(s,d),(1.0+data[at+11u])*quadratic(d,nu)+dot(s,vec3f(ru*ru,rv*rv,2.0*ru*rv)));
}
fn hingeAngle(h:u32,v:u32,x:vec3f,previous:bool)->f32{let at=params.bases.y+h*8u;let a=u32(data[at]);let b=u32(data[at+1u]);let c=u32(data[at+2u]);let d=u32(data[at+3u]);let e=edge(a,b,v,x,previous);let n=cross(e,edge(a,c,v,x,previous));let m=cross(edge(a,d,v,x,previous),e);return atan2(dot(normalize(e),cross(n,m)),dot(n,m));}
fn bendingPhi(angle:f32,mode:f32)->vec3f{if(mode==1.0){let t=tan(angle*.5);return vec3f(2.0*t,1.0+t*t,t*(1.0+t*t));}if(mode==2.0){return vec3f(sin(angle*.5),.5*cos(angle*.5),-.25*sin(angle*.5));}return vec3f(angle,1,0);}
fn hingeEnergy(h:u32,v:u32,x:vec3f)->f32{let at=params.bases.y+h*8u;let angle=hingeAngle(h,v,x,false);let phi=bendingPhi(angle,data[at+7u]).x;let d=phi-bendingPhi(data[at+4u],data[at+7u]).x;let oldD=phi-bendingPhi(hingeAngle(h,v,x,true),data[at+7u]).x;return .5*data[at+5u]*(d*d+data[at+6u]*oldD*oldD);}
fn hingeDerivative(h:u32,v:u32,xv:vec3f,axis:u32)->vec2f{
 let at=params.bases.y+h*8u;let a=u32(data[at]);let b=u32(data[at+1u]);let c=u32(data[at+2u]);let d=u32(data[at+3u]);
 var unit=vec3f(0);unit[axis]=1.0;let da=select(0.0,1.0,v==a);let de=unit*(select(0.0,1.0,v==b)-da);let du=unit*(select(0.0,1.0,v==c)-da);let dv=unit*(select(0.0,1.0,v==d)-da);
 let e=edge(a,b,v,xv,false);let u=edge(a,c,v,xv,false);let w=edge(a,d,v,xv,false);let rawN=cross(e,u);let rawM=cross(w,e);let scaleN=length(rawN);let scaleM=length(rawM);let n=rawN/scaleN;let m=rawM/scaleM;let dn=(cross(de,u)+cross(e,du))/scaleN;let dm=(cross(dv,e)+cross(w,de))/scaleM;
 let l=length(e);let dl=dot(e,de)/l;let ddl=(dot(de,de)-dl*dl)/l;let t=e/l;let td=de/l-e*(dl/(l*l));let tdd=de*(-2.0*dl/(l*l))+e*(2.0*dl*dl/(l*l*l)-ddl/(l*l));
 let product=cross(n,m);let productD=cross(dn,m)+cross(n,dm);let productDD=2.0*cross(dn,dm);
 let xx=dot(n,m);let dx=dot(dn,m)+dot(n,dm);let ddx=2.0*dot(dn,dm);let yy=dot(t,product);let dy=dot(td,product)+dot(t,productD);let ddy=dot(tdd,product)+2.0*dot(td,productD)+dot(t,productDD);
 let denominator=xx*xx+yy*yy;let numerator=xx*dy-yy*dx;let derivative=numerator/denominator;let second=((xx*ddy-yy*ddx)*denominator-numerator*2.0*(xx*dx+yy*dy))/(denominator*denominator);
 let angle=atan2(yy,xx);let phi=bendingPhi(angle,data[at+7u]);let stress=phi.x-bendingPhi(data[at+4u],data[at+7u]).x+data[at+6u]*(phi.x-bendingPhi(hingeAngle(h,v,xv,true),data[at+7u]).x);
 return data[at+5u]*vec2f(stress*phi.y*derivative,(1.0+data[at+6u])*phi.y*phi.y*derivative*derivative+stress*(phi.y*second+phi.z*derivative*derivative));
}
fn segmentEnergy(s:u32,v:u32,x:vec3f)->f32{let at=params.bases.z+s*16u;let a=u32(data[at]);let b=u32(data[at+1u]);let l=length(edge(a,b,v,x,false));let oldL=length(edge(a,b,v,x,true));let strain=l-data[at+2u];let d=l-oldL;return .5*data[at+3u]*(strain*strain+data[at+4u]*d*d);}
fn segmentDerivative(s:u32,v:u32,x:vec3f,axis:u32)->vec2f{let at=params.bases.z+s*16u;let a=u32(data[at]);let b=u32(data[at+1u]);let e=edge(a,b,v,x,false);let l=length(e);let d=e[axis]/l;let stress=l-data[at+2u]+data[at+4u]*(l-length(edge(a,b,v,x,true)));return data[at+3u]*vec2f(stress*d*select(-1.0,1.0,v==b),(1.0+data[at+4u])*d*d+stress/l*(1.0-d*d));}
fn transport(d:vec3f,priorTangent:vec3f,to:vec3f)->vec3f{let axis=cross(priorTangent,to);let cosine=clamp(dot(priorTangent,to),-1.0,1.0);var result=d;if(cosine> -1.0+1e-6){result=d+cross(axis,d)+cross(axis,cross(axis,d))/(1.0+cosine);}let n=result-to*dot(result,to);if(length(n)<1e-8){return normalize(cross(to,select(vec3f(0,1,0),vec3f(1,0,0),abs(to.x)<.8)));}return normalize(n);}
struct Frame{t:vec3f,d:vec3f,m1:vec3f,m2:vec3f}
fn frame(s:u32,v:u32,x:vec3f,changedAngle:u32,theta:f32,previous:bool)->Frame{
 let at=params.bases.z+s*16u;let t=normalize(edge(u32(data[at]),u32(data[at+1u]),v,x,previous));let d=transport(vec(at+8u),vec(at+12u),t);let other=cross(t,d);var angle=angles[s].x;if(previous){angle=angles[s].z;}else if(s==changedAngle){angle=theta;}let m=cos(angle)*d+sin(angle)*other;return Frame(t,d,m,cross(t,m));
}
struct JointValue { curvature:vec4f, twist:f32 }
fn jointValue(j:u32,v:u32,x:vec3f,changedAngle:u32,theta:f32,previous:bool)->JointValue{
 let at=params.bases.w+j*16u;let a=frame(u32(data[at]),v,x,changedAngle,theta,previous);let b=frame(u32(data[at+1u]),v,x,changedAngle,theta,previous);let kb=2.0*cross(a.t,b.t)/max(1e-10,1.0+dot(a.t,b.t));let referenceDirector=transport(a.d,a.t,b.t);let rawTwist=atan2(dot(b.t,cross(referenceDirector,b.d)),dot(referenceDirector,b.d));let priorReference=data[at+12u];let twist=priorReference+atan2(sin(rawTwist-priorReference),cos(rawTwist-priorReference));
 var ta=angles[u32(data[at])].x;var tb=angles[u32(data[at+1u])].x;
 if(previous){ta=angles[u32(data[at])].z;tb=angles[u32(data[at+1u])].z;}else{if(u32(data[at])==changedAngle){ta=theta;}if(u32(data[at+1u])==changedAngle){tb=theta;}}
 return JointValue(vec4f(dot(kb,a.m2),-dot(kb,a.m1),dot(kb,b.m2),-dot(kb,b.m1)),tb-ta+twist);
}
fn jointEnergy(j:u32,v:u32,x:vec3f,changedAngle:u32,theta:f32)->f32{
 let at=params.bases.w+j*16u;let q=jointValue(j,v,x,changedAngle,theta,false);let previous=jointValue(j,v,x,changedAngle,theta,true);let d=q.curvature-vec4f(data[at+6u],data[at+7u],data[at+8u],data[at+9u]);let dv=q.curvature-previous.curvature;let weights=vec4f(data[at+2u],data[at+10u],data[at+2u],data[at+10u]);let twist=q.twist-data[at+11u];let viscousTwist=q.twist-previous.twist;return .25*dot(weights,d*d+data[at+4u]*dv*dv)+.5*data[at+3u]*(twist*twist+data[at+5u]*viscousTwist*viscousTwist);
}
fn vertexEnergy(v:u32,x:vec3f)->f32{
 let attach=params.sizes.w+3u*params.sizes.x+4u*v;let support=x-vec(attach);let difference=x-prediction[v].xyz;var energy=.5*data[attach+3u]*dot(support,support)+.5*old[v].w/(params.step.x*params.step.x)*dot(difference,difference);let header=v*8u;
 for(var i=0u;i<adjacency[header+1u];i++){energy+=membraneEnergy(adjacency[adjacency[header]+i],v,x);}
 for(var i=0u;i<adjacency[header+3u];i++){energy+=hingeEnergy(adjacency[adjacency[header+2u]+i],v,x);}
 for(var i=0u;i<adjacency[header+5u];i++){energy+=segmentEnergy(adjacency[adjacency[header+4u]+i],v,x);}
 for(var i=0u;i<adjacency[header+7u];i++){energy+=jointEnergy(adjacency[adjacency[header+6u]+i],v,x,0xffffffffu,0);}
 return energy;
}
fn vertexDerivative(v:u32,x:vec3f,axis:u32)->vec2f{
 let inertia=old[v].w/(params.step.x*params.step.x);let attach=params.sizes.w+3u*params.sizes.x+4u*v;var derivative=vec2f(inertia*(x[axis]-prediction[v][axis])+data[attach+3u]*(x[axis]-data[attach+axis]),inertia+data[attach+3u]);let header=v*8u;
 for(var i=0u;i<adjacency[header+1u];i++){derivative+=membraneDerivative(adjacency[adjacency[header]+i],v,x,axis);}
 for(var i=0u;i<adjacency[header+3u];i++){derivative+=hingeDerivative(adjacency[adjacency[header+2u]+i],v,x,axis);}
 for(var i=0u;i<adjacency[header+5u];i++){derivative+=segmentDerivative(adjacency[adjacency[header+4u]+i],v,x,axis);}
 let h=max(1e-7,delta[v].w*.001);var plus=x;var minus=x;plus[axis]+=h;minus[axis]-=h;
 for(var i=0u;i<adjacency[header+7u];i++){let j=adjacency[adjacency[header+6u]+i];let e=jointEnergy(j,v,x,0xffffffffu,0);let a=jointEnergy(j,v,plus,0xffffffffu,0);let b=jointEnergy(j,v,minus,0xffffffffu,0);derivative+=vec2f((a-b)/(2.0*h),(a+b-2.0*e)/(h*h));}
 return derivative;
}
@compute @workgroup_size(64) fn solvePositions(@builtin(global_invocation_id) gid:vec3u){
 if(gid.x>=params.work.y){return;}let v=adjacency[params.work.x+gid.x];if(prediction[v].w>0.0){return;}var x=delta[v].xyz;let before=x;var residual=0.0;
 for(var axis=0u;axis<3u;axis++){let e=vertexEnergy(v,x);let d=vertexDerivative(v,x,axis);let inertia=old[v].w/(params.step.x*params.step.x);if(!(abs(e)<3e38&&abs(d.x)<3e38&&abs(d.y)<3e38)){atomicMax(&update,0x7f800000u);return;}let requested=-d.x/max(inertia,d.y);residual=max(residual,abs(requested));var change=clamp(requested,-delta[v].w,delta[v].w);var accepted=x;
  for(var line=0u;line<16u;line++){var candidate=x;candidate[axis]+=change;if(vertexEnergy(v,candidate)<=e+max(1e-25,abs(e)*2e-7)){accepted=candidate;break;}change*=.5;}x=accepted;
 }
 delta[v]=vec4f(x,delta[v].w);let change=max(max(abs(x.x-before.x),abs(x.y-before.y)),abs(x.z-before.z));atomicMax(&update,bitcast<u32>(max(change,residual)));
}
fn angleEnergy(s:u32,theta:f32)->f32{let header=params.sizes.y+s*2u;let d=theta-angles[s].y;let attach=params.sizes.w+7u*params.sizes.x+2u*s;var energy=.5*angles[s].w/(params.step.x*params.step.x)*d*d+.5*data[attach+1u]*(theta-data[attach])*(theta-data[attach]);for(var i=0u;i<adjacency[header+1u];i++){energy+=jointEnergy(adjacency[adjacency[header]+i],0xffffffffu,vec3f(0),s,theta);}return energy;}
@compute @workgroup_size(64) fn solveTwists(@builtin(global_invocation_id) gid:vec3u){
 if(gid.x>=params.work.y){return;}let s=adjacency[params.work.x+gid.x];if(data[params.bases.z+s*16u+7u]>0.0){return;}let theta=angles[s].x;let h=.001;let e=angleEnergy(s,theta);let a=angleEnergy(s,theta+h);let b=angleEnergy(s,theta-h);if(!(abs(e)<3e38&&abs(a)<3e38&&abs(b)<3e38)){atomicMax(&update,0x7f800000u);return;}let requested=-(a-b)/(2.0*h)/max(angles[s].w/(params.step.x*params.step.x),(a+b-2.0*e)/(h*h));var change=clamp(requested,-.25,.25);var accepted=theta;
 for(var line=0u;line<16u;line++){if(angleEnergy(s,theta+change)<=e+max(1e-25,abs(e)*2e-7)){accepted=theta+change;break;}change*=.5;}angles[s].x=accepted;atomicMax(&update,bitcast<u32>(max(abs(accepted-theta),abs(requested))*data[params.bases.z+s*16u+5u]));
}
`;
interface Color {
    offset: number;
    count: number;
    kind: 'positions' | 'twists';
}
export class GPUClothMechanics {
    private readonly buffers: GPUBuffer[] = [];
    private readonly colors: Color[] = [];
    private readonly adjacency: Uint32Array<ArrayBuffer>;
    private readonly segmentVertices: number[][] = [];
    private readonly joints: {
        s0: number;
        s1: number;
        rod: number;
        local: number;
    }[] = [];
    private readonly segmentRod: {
        rod: number;
        local: number;
    }[] = [];
    private readonly bases: readonly number[];
    private readonly angleHeader: number;
    private readonly position: GPUBuffer;
    private readonly old: GPUBuffer;
    private readonly predicted: GPUBuffer;
    private readonly data: GPUBuffer;
    private readonly indices: GPUBuffer;
    private readonly angles: GPUBuffer;
    private readonly parameters: GPUBuffer;
    private readonly update: GPUBuffer;
    private positionPipeline?: GPUComputePipeline;
    private twistPipeline?: GPUComputePipeline;
    private bindings?: GPUBindGroup;
    constructor(private readonly device: GPUDevice, readonly topology: PreparedClothTopology) {
        const n = topology.vertexCount, segments: number[][] = Array.from({ length: n }, () => []), joints: number[][] = Array.from({ length: n }, () => []), angleJoints: number[][] = Array.from({ length: topology.segmentCount }, () => []);
        for (let r = 0; r < topology.rods.length; r++) {
            const rod = topology.rods[r]!;
            for (let j = 0; j < rod.restLengths_m.length; j++) {
                const s = rod.segmentOffset + j, vertices = [rod.vertices[j]!, rod.vertices[j + 1]!];
                this.segmentVertices[s] = vertices;
                this.segmentRod[s] = { rod: r, local: j };
                for (const v of vertices)
                    segments[v]!.push(s);
                if (j) {
                    const id = this.joints.length;
                    this.joints.push({ s0: s - 1, s1: s, rod: r, local: j });
                    for (const v of new Set([rod.vertices[j - 1]!, ...vertices]))
                        joints[v]!.push(id);
                    angleJoints[s - 1]!.push(id);
                    angleJoints[s]!.push(id);
                }
            }
        }
        this.angleHeader = n * 8;
        const index: number[] = new Array(n * 8 + topology.segmentCount * 2).fill(0);
        for (let i = 0; i < n; i++)
            for (const [j, list] of [topology.incidentTriangles[i]!, topology.incidentHinges[i]!, segments[i]!, joints[i]!].entries()) {
                index[i * 8 + j * 2] = index.length;
                index[i * 8 + j * 2 + 1] = list.length;
                index.push(...list);
            }
        for (let i = 0; i < angleJoints.length; i++) {
            index[this.angleHeader + i * 2] = index.length;
            index[this.angleHeader + i * 2 + 1] = angleJoints[i]!.length;
            index.push(...angleJoints[i]!);
        }
        const colorGraph = (count: number, neighbors: (i: number) => Iterable<number>, kind: Color['kind']) => { const assigned = new Int32Array(count).fill(-1), groups: number[][] = []; for (let i = 0; i < count; i++) {
            const blocked = new Set<number>();
            for (const v of neighbors(i))
                if (assigned[v]! >= 0)
                    blocked.add(assigned[v]!);
            let c = 0;
            while (blocked.has(c))
                c++;
            assigned[i] = c;
            (groups[c] ??= []).push(i);
        } for (const group of groups) {
            this.colors.push({ offset: index.length, count: group.length, kind });
            for (const i of group)
                index.push(i);
        } };
        colorGraph(n, i => [...topology.incidentTriangles[i]!.flatMap(f => topology.triangles[f]!.vertices), ...topology.incidentHinges[i]!.flatMap(h => topology.hinges[h]!.vertices), ...segments[i]!.flatMap(s => this.segmentVertices[s]!), ...joints[i]!.flatMap(j => { const q = this.joints[j]!; return [...this.segmentVertices[q.s0]!, ...this.segmentVertices[q.s1]!]; })], 'positions');
        colorGraph(topology.segmentCount, s => angleJoints[s]!.flatMap(j => [this.joints[j]!.s0, this.joints[j]!.s1]), 'twists');
        this.adjacency = Uint32Array.from(index);
        this.bases = [0, topology.triangles.length * 16, topology.triangles.length * 16 + topology.hinges.length * 8, topology.triangles.length * 16 + topology.hinges.length * 8 + topology.segmentCount * 16];
        const buffer = (label: string, size: number, usage: number) => { const b = device.createBuffer({ label, size: Math.max(16, Math.ceil(size / 8) * 8), usage }); this.buffers.push(b); return b; }, rw = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
        this.position = buffer('Cloth displacement unknowns', n * 16, rw);
        this.old = buffer('Cloth accepted positions and mass', n * 16, rw);
        this.predicted = buffer('Cloth inertial targets and supports', n * 16, rw);
        this.data = buffer('Cloth physical energy stencils', (this.bases[3]! + this.joints.length * 16 + n * 7 + topology.segmentCount * 2) * 4, rw);
        this.indices = buffer('Cloth energy adjacency and independent colors', this.adjacency.byteLength, rw);
        this.angles = buffer('Cloth rod material angle unknowns', topology.segmentCount * 16, rw);
        this.parameters = buffer('Cloth solve color parameters', this.colors.length * 256, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.update = buffer('Cloth nonlinear convergence', 16, rw);
        this.writeWords(this.indices,this.adjacency);
    }
    private writeWords(buffer:GPUBuffer,values:Float32Array<ArrayBuffer>|Uint32Array<ArrayBuffer>):void{
        // Bound upload staging and avoid Dawn/Metal's >4 MiB odd-word
        // UploadData assertion. The full authored topology remains unchanged.
        const chunkWords=256*1024;
        for(let from=0;from<values.length;from+=chunkWords)this.device.queue.writeBuffer(buffer,from*4,values,from,Math.min(chunkWords,values.length-from));
    }
    initializationTasks(): GPUInitializationTask[] {
        return [{ id: 'cloth.mechanics.pipeline', phase: 'solver-pipelines', label: 'Compile physical cloth mechanics', run: async () => {
                    const shader = this.device.createShaderModule({ label: 'Physical shell and discrete rod energies', code: clothMechanicsWGSL });
                    const diagnostics = await shader.getCompilationInfo();
                    const errors = diagnostics.messages.filter(m => m.type === 'error');
                    if (errors.length)
                        throw new Error(errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
                    const layout = this.device.createBindGroupLayout({ entries: Array.from({ length: 8 }, (_, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding === 6 ? 'uniform' : ([0, 5, 7].includes(binding) ? 'storage' : 'read-only-storage'), ...(binding === 6 ? { hasDynamicOffset: true, minBindingSize: 64 } : {}) } })) });
                    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [layout] });
                    [this.positionPipeline, this.twistPipeline] = await Promise.all(['solvePositions', 'solveTwists'].map(entryPoint => this.device.createComputePipelineAsync({ label: `Physical cloth ${entryPoint}`, layout: pipelineLayout, compute: { module: shader, entryPoint } })));
                    this.bindings = this.device.createBindGroup({ layout, entries: [this.position, this.old, this.predicted, this.data, this.indices, this.angles, this.parameters, this.update].map((buffer, binding) => ({ binding, resource: { buffer, ...(binding === 6 ? { size: 64 } : {}) } })) });
                } }];
    }
    async initialize() { for (const task of this.initializationTasks())
        await task.run(new AbortController().signal); }
    async step(state: ClothState, dt: number, options: ClothStepOptions = {}): Promise<ClothStepReport> {
        if (!this.positionPipeline || !this.twistPipeline || !this.bindings)
            throw new Error('Initialize cloth GPU mechanics before stepping');
        if (!(dt > 0 && Number.isFinite(dt)))
            throw new Error('Invalid cloth GPU timestep');
        if (options.cohesionPairs?.length)
            throw new Error('The GPU Gauss-Seidel sweep has no cohesion energy: cohesive cloth steps on the GPU Newton-PCG owner');
        const topology = this.topology, n = topology.vertexCount, old = state.positions_m.slice(), oldTwist = state.twist_rad.slice(), positions = new Float32Array(n * 4), previous = new Float32Array(n * 4), prediction = new Float32Array(n * 4), data = new Float32Array(this.bases[3]! + this.joints.length * 16 + n * 7 + topology.segmentCount * 2), angles = new Float32Array(topology.segmentCount * 4), parameters = new ArrayBuffer(this.colors.length * 256), pf = new Float32Array(parameters), pu = new Uint32Array(parameters);
        const gravity = options.gravity_m_s2 ?? [0, -9.81, 0], target = options.supportPositions_m ?? topology.data.initialPositions_m ?? topology.data.restPositions_m;
        const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
        for (let i = 0; i < old.length; i++) {
            const a = i % 3;
            lo[a] = Math.min(lo[a]!, old[i]!);
            hi[a] = Math.max(hi[a]!, old[i]!);
        }
        const origin = lo.map((v, a) => (v + hi[a]!) / 2);
        for (let i = 0; i < n; i++) {
            const mass = state.dryMass_kg[i]! + state.waterVolume_m3[i]! * topology.materials[topology.vertexMaterial[i]!]!.waterDensity_kg_m3, fixed = (topology.kinematicFixed[i] ?? 0) & 1;
            previous.set([old[3 * i]! - origin[0]!, old[3 * i + 1]! - origin[1]!, old[3 * i + 2]! - origin[2]!, mass], i * 4);
            for (let a = 0; a < 3; a++) {
                data[this.bases[3]! + this.joints.length * 16 + 3 * i + a] = (old[3 * i + a]! - origin[a]!) - previous[4 * i + a]!;
                const value = fixed ? target[3 * i + a]! - old[3 * i + a]! : dt * state.velocities_m_s[3 * i + a]! + dt * dt * (gravity[a]! + (options.externalForces_N?.[3 * i + a] ?? 0) / mass);
                positions[4 * i + a] = value;
                prediction[4 * i + a] = value;
            }
            positions[4 * i + 3] = Math.cbrt(state.restControlVolume_m3[i]!);
            prediction[4 * i + 3] = fixed;
            data.set([target[3*i]!-old[3*i]!,target[3*i+1]!-old[3*i+1]!,target[3*i+2]!-old[3*i+2]!,topology.attachmentTranslation_N_m[i]!],this.bases[3]!+this.joints.length*16+n*3+4*i);
        }
        for (let f = 0; f < topology.triangles.length; f++) {
            const face = topology.triangles[f]!, m = topology.materials[face.material]!, fraction = face.vertices.reduce((s, i) => s + state.solidFraction[i]!, 0) / 3;
            data.set([...face.vertices, face.restArea_m2 * membraneModulus(m, fraction), ...face.ru, m.poissonRatio, ...face.rv, (m.membraneDamping_s??m.mechanicalDamping_s) / dt], this.bases[0]! + f * 16);
        }
        for (let h = 0; h < topology.hinges.length; h++) {
            const hinge = topology.hinges[h]!, m = topology.materials[topology.triangles[hinge.faces[0]]!.material]!, fraction = (state.solidFraction[hinge.vertices[0]]! + state.solidFraction[hinge.vertices[1]]!) / 2;
            data.set([...hinge.vertices, hinge.restAngle_rad, bendingRigidity(m, fraction) * hinge.shapeFactor, (m.bendingDamping_s??m.mechanicalDamping_s) / dt, m.bendingMode === 'tan-half' ? 1 : m.bendingMode === 'sin-half' ? 2 : 0], this.bases[1]! + h * 8);
        }
        for (let s = 0; s < topology.segmentCount; s++) {
            const entry = this.segmentRod[s]!, rod = topology.rods[entry.rod]!, m = topology.materials[rod.material]!, vertices = this.segmentVertices[s]!, fraction = m.attachmentMultiplier!==undefined?state.solidFraction[vertices[0]!]!:(state.solidFraction[vertices[0]!]! + state.solidFraction[vertices[1]!]!) / 2, rigidity = rodRigidity(m, fraction), inertia = rodSegmentInertia_kg_m2(topology,state,entry.rod,entry.local), fixed = ((topology.kinematicFixed[vertices[0]!] ?? 0) | (topology.kinematicFixed[vertices[1]!] ?? 0)) & 2;
            data.set([...vertices, rod.restLengths_m[entry.local]!, rigidity.stretch_N / rod.restLengths_m[entry.local]!, (m.rodStretchDamping_s??m.mechanicalDamping_s) / dt * (m.attachmentMultiplier!==undefined?rod.restLengths_m[entry.local]!/Math.hypot(...[0,1,2].map(a=>old[3*vertices[1]!+a]!-old[3*vertices[0]!+a]!)):1), Math.max(m.rodRadius_m,m.rodRadiusB_m??m.rodRadius_m), inertia, fixed, ...point3(state.rodDirectors, s), 0, ...point3(state.rodTangents, s), 0], this.bases[2]! + s * 16);
            const predicted = state.twist_rad[s]! + dt * state.angularVelocity_rad_s[s]!;
            data.set([state.attachmentTwistTarget_rad[s]!,topology.attachmentTwist_N_m[s]!],this.bases[3]!+this.joints.length*16+n*7+2*s);
            angles.set([fixed ? state.twist_rad[s]! : predicted, predicted, state.twist_rad[s]!, inertia], s * 4);
        }
        this.joints.forEach((joint,j)=>{const rod=topology.rods[joint.rod]!,m=topology.materials[rod.material]!,v=joint.local,fraction=m.attachmentMultiplier!==undefined?state.solidFraction[rod.vertices[v]!]!:(state.solidFraction[rod.vertices[v-1]!]!+state.solidFraction[rod.vertices[v]!]!+state.solidFraction[rod.vertices[v+1]!]!)/3,rigidity=rodRigidity(m,fraction),length=(rod.restLengths_m[v-1]!+rod.restLengths_m[v]!)/2;
            data.set([joint.s0,joint.s1,rigidity.bend1_N_m2/length,rigidity.twist_N_m2/length,(m.rodBendingDamping_s??m.mechanicalDamping_s)/dt,(m.rodTwistDamping_s??m.mechanicalDamping_s)/dt,...rod.restCurvature.subarray(4*(v-1),4*v),rigidity.bend2_N_m2/length,rod.restTwist[v-1]!,state.rodReferenceTwist_rad[rod.segmentOffset+v]!,0,0,0],this.bases[3]!+j*16);
        });
        this.colors.forEach((color, i) => { pf[i * 64] = dt; pu.set(this.bases, i * 64 + 4); pu.set([color.offset, color.count, 0, 0], i * 64 + 8); pu.set([n, this.angleHeader, topology.segmentCount, this.bases[3]! + this.joints.length * 16], i * 64 + 12); });
        for (const [buffer, values] of [[this.position, positions], [this.old, previous], [this.predicted, prediction], [this.data, data], [this.angles, angles]] as const)
            if (values.byteLength)
                this.writeWords(buffer,values);
        this.device.queue.writeBuffer(this.parameters, 0, parameters);
        const positionBytes = n * 16, angleBytes = topology.segmentCount * 16, readback = this.device.createBuffer({ label: 'Accepted cloth nonlinear solve', size: positionBytes + angleBytes + 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const maxIterations = options.iterations ?? 64, tolerance = options.tolerance_m ?? 1e-8;
        let iterations = 0, maximumUpdate = Infinity;
        try {
            while (iterations < maxIterations) {
                const batch = Math.min(4, maxIterations - iterations), encoder = this.device.createCommandEncoder({ label: 'Cloth implicit energy minimization' });
                for (let sweep = 0; sweep < batch; sweep++) {
                    encoder.clearBuffer(this.update);
                    const pass = encoder.beginComputePass({ label: 'Cloth independent energy colors' });
                    for (let ci = 0; ci < this.colors.length; ci++) {
                        const color = this.colors[ci]!;
                        pass.setPipeline(color.kind === 'positions' ? this.positionPipeline : this.twistPipeline);
                        pass.setBindGroup(0, this.bindings, [ci * 256]);
                        pass.dispatchWorkgroups(Math.ceil(color.count / 64));
                    }
                    pass.end();
                }
                encoder.copyBufferToBuffer(this.position, 0, readback, 0, positionBytes);
                if (angleBytes)
                    encoder.copyBufferToBuffer(this.angles, 0, readback, positionBytes, angleBytes);
                encoder.copyBufferToBuffer(this.update, 0, readback, positionBytes + angleBytes, 4);
                this.device.queue.submit([encoder.finish()]);
                await readback.mapAsync(GPUMapMode.READ);
                const values = new Float32Array(readback.getMappedRange());
                maximumUpdate = values[(positionBytes + angleBytes) / 4]!;
                iterations += batch;
                if (!Number.isFinite(maximumUpdate))
                    throw new Error('Nonfinite GPU cloth nonlinear update');
                if (maximumUpdate < tolerance || iterations >= maxIterations) {
                    for (let i = 0; i < n; i++)
                        for (let a = 0; a < 3; a++)
                            state.positions_m[3 * i + a] = old[3 * i + a]! + values[4 * i + a]!;
                    for (let s = 0; s < topology.segmentCount; s++)
                        state.twist_rad[s] = values[positionBytes / 4 + 4 * s]!;
                    readback.unmap();
                    break;
                }
                readback.unmap();
            }
        }
        finally {
            if (readback.mapState === 'mapped')
                readback.unmap();
            readback.destroy();
        }
        if(maximumUpdate>=tolerance){
            // No Float64 CPU refinement: the sweep's failure is the step's.
            state.positions_m.set(old);state.twist_rad.set(oldTwist);
            throw new Error(`GPU Gauss-Seidel cloth sweep did not converge (${maximumUpdate} m after ${iterations} sweeps, tolerance ${tolerance} m)`);
        }
        return extendClothStepReport(finishClothMechanicalStep(topology, state, old, oldTwist, dt, options, { iterations, converged: maximumUpdate < tolerance, maximumUpdate_m: maximumUpdate }),{gpuIterations:iterations,refinementIterations:0});
    }
    destroy() { for (const buffer of this.buffers)
        buffer.destroy(); }
}
