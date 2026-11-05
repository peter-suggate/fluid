import type {Quaternion,Vec3} from './model';
/** An immutable subtraction in the owning surface's local coordinates. */
export interface SolidBoxCut {readonly center_m:Vec3;readonly halfExtents_m:Vec3;readonly orientation:Quaternion}
export const SOLID_BOX_CUT_FLOATS=12;
export function validateSolidBoxCuts(cuts:readonly SolidBoxCut[]):void{
 if(!Array.isArray(cuts))throw new Error('Solid cuts must be an array');
 for(const c of cuts){if(!c||![c.center_m?.x,c.center_m?.y,c.center_m?.z,c.halfExtents_m?.x,c.halfExtents_m?.y,c.halfExtents_m?.z,c.orientation?.w,c.orientation?.x,c.orientation?.y,c.orientation?.z].every(Number.isFinite)||![c.halfExtents_m.x,c.halfExtents_m.y,c.halfExtents_m.z].every(v=>v>0)||Math.abs(Math.hypot(c.orientation.w,c.orientation.x,c.orientation.y,c.orientation.z)-1)>1e-5)throw new Error('Invalid oriented solid box cut');}
}
export function solidBoxCutDistance(c:SolidBoxCut,p:Vec3):number{
 const q=c.orientation,x=p.x-c.center_m.x,y=p.y-c.center_m.y,z=p.z-c.center_m.z;
 const tx=2*(-q.y*z+q.z*y),ty=2*(-q.z*x+q.x*z),tz=2*(-q.x*y+q.y*x);
 const a=Math.abs(x+q.w*tx-q.y*tz+q.z*ty)-c.halfExtents_m.x,b=Math.abs(y+q.w*ty-q.z*tx+q.x*tz)-c.halfExtents_m.y,d=Math.abs(z+q.w*tz-q.x*ty+q.y*tx)-c.halfExtents_m.z;
 return Math.min(Math.max(a,b,d),0)+Math.hypot(Math.max(a,0),Math.max(b,0),Math.max(d,0));
}
export function applySolidBoxCuts(distance:number,cuts:readonly SolidBoxCut[]|undefined,p:Vec3):number{
 for(const cut of cuts??[])distance=Math.max(distance,-solidBoxCutDistance(cut,p));return distance;
}
export function packSolidBoxCuts(cuts:readonly SolidBoxCut[]):Float32Array<ArrayBuffer>{
 validateSolidBoxCuts(cuts);const out=new Float32Array(cuts.length*SOLID_BOX_CUT_FLOATS);
 cuts.forEach((c,i)=>out.set([c.center_m.x,c.center_m.y,c.center_m.z,0,c.halfExtents_m.x,c.halfExtents_m.y,c.halfExtents_m.z,0,c.orientation.x,c.orientation.y,c.orientation.z,c.orientation.w],12*i));return out;
}
export function unpackSolidBoxCuts(values:Float32Array,count:number,offset=0):SolidBoxCut[]{
 if(!Number.isSafeInteger(count)||count<0||offset+12*count>values.length)throw new Error('Malformed packed solid cut range');
 return Array.from({length:count},(_,i)=>{const at=offset+12*i;return {center_m:{x:values[at]!,y:values[at+1]!,z:values[at+2]!},halfExtents_m:{x:values[at+4]!,y:values[at+5]!,z:values[at+6]!},orientation:{x:values[at+8]!,y:values[at+9]!,z:values[at+10]!,w:values[at+11]!}};});
}
export const solidBoxCutWGSL=/* wgsl */`
fn solidBoxCutDistance(p:vec3f,center:vec3f,extent:vec3f,rotation:vec4f)->f32{
 let q=vec4f(-rotation.xyz,rotation.w);let relative=p-center;let local=relative+2.0*cross(q.xyz,cross(q.xyz,relative)+q.w*relative);
 let d=abs(local)-extent;return min(max(d.x,max(d.y,d.z)),0.0)+length(max(d,vec3f(0.0)));
}`;

export const RIGID_BODY_CUT_ATLAS_WIDTH=2048;
/** A texel per body: (firstCutTexel, cutCount, 0, 0), followed by three
 * texels per local oriented box (center, extent, quaternion xyzw). */
export function packRigidBodyCutAtlas(bodies:readonly {cuts?:readonly SolidBoxCut[]}[],bodyCapacity=24):Float32Array<ArrayBuffer>{
 if(bodies.length>bodyCapacity)throw new Error('Solid-cut roster exceeds native rigid capacity');
 const values=new Float32Array(RIGID_BODY_CUT_ATLAS_WIDTH*4);let at=bodyCapacity;
 bodies.forEach((body,i)=>{const cuts=body.cuts??[];if(at+3*cuts.length>RIGID_BODY_CUT_ATLAS_WIDTH)throw new Error('Body-local cut atlas capacity exceeded');values[4*i]=at;values[4*i+1]=cuts.length;values.set(packSolidBoxCuts(cuts),4*at);at+=3*cuts.length;});return values;
}
