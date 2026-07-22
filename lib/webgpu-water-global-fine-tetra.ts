const TETS = [
  [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
  [0, 7, 4, 6], [0, 5, 6, 4], [0, 5, 1, 6],
] as const;

const value = (corner: number) => `v${corner}`;
const countCall = (tetra: readonly number[]) => `countTet(${tetra.map(value).join(",")})`;

export const globalFineClassifiedCountShader = /* wgsl */ `
struct A{vertexCount:atomic<u32>,instanceCount:u32,firstVertex:u32,firstInstance:u32,activeCubeCount:atomic<u32>,vertexAllocator:atomic<u32>}
@group(0)@binding(4)var<storage,read_write>args:A;@group(0)@binding(5)var<storage,read>cubes:array<vec2u>;@group(0)@binding(6)var<storage,read>values:array<vec4f>;@group(0)@binding(7)var<storage,read_write>offsets:array<u32>;
fn countTet(a:f32,b:f32,c:f32,d:f32)->u32{let n=select(0u,1u,a>=.5)+select(0u,1u,b>=.5)+select(0u,1u,c>=.5)+select(0u,1u,d>=.5);if(n==0u||n==4u){return 0u;}return select(3u,6u,n==2u);}
@compute @workgroup_size(64)fn countGlobalFineTriangles(@builtin(global_invocation_id)g:vec3u){let i=g.x;let count=min(atomicLoad(&args.activeCubeCount),min(arrayLength(&cubes),arrayLength(&offsets)));if(i>=count||i*2u+1u>=arrayLength(&values)){return;}let lo=values[i*2u];let hi=values[i*2u+1u];let v0=lo.x;let v1=lo.y;let v2=lo.z;let v3=lo.w;let v4=hi.x;let v5=hi.y;let v6=hi.z;let v7=hi.w;offsets[i]=${TETS.map(countCall).join("+")};}
`;

export const globalFineClassifiedScanShader = /* wgsl */ `
struct V{position:vec4f,normal:vec4f}struct A{vertexCount:atomic<u32>,instanceCount:u32,firstVertex:u32,firstInstance:u32,activeCubeCount:atomic<u32>,vertexAllocator:atomic<u32>}
@group(0)@binding(3)var<storage,read_write>out:array<V>;@group(0)@binding(4)var<storage,read_write>args:A;@group(0)@binding(5)var<storage,read>cubes:array<vec2u>;@group(0)@binding(6)var<storage,read>values:array<vec4f>;@group(0)@binding(7)var<storage,read_write>offsets:array<u32>;
fn countTet(a:f32,b:f32,c:f32,d:f32)->u32{let n=select(0u,1u,a>=.5)+select(0u,1u,b>=.5)+select(0u,1u,c>=.5)+select(0u,1u,d>=.5);if(n==0u||n==4u){return 0u;}return select(3u,6u,n==2u);}
@compute @workgroup_size(1)fn scanGlobalFineTriangles(){if(atomicLoad(&args.vertexAllocator)==0xffffffffu){return;}let count=min(atomicLoad(&args.activeCubeCount),min(arrayLength(&cubes),min(arrayLength(&offsets)/6u,arrayLength(&values)/2u)));var total=0u;for(var i=0u;i<count;i+=1u){let lo=values[i*2u];let hi=values[i*2u+1u];let v0=lo.x;let v1=lo.y;let v2=lo.z;let v3=lo.w;let v4=hi.x;let v5=hi.y;let v6=hi.z;let v7=hi.w;${TETS.map((tetra,index)=>`offsets[i*6u+${index}u]=total;total+=${countCall(tetra)};`).join("")}}let capacity=arrayLength(&out)-arrayLength(&out)%3u;let fitted=min(total,capacity);atomicStore(&args.vertexCount,fitted);atomicStore(&args.vertexAllocator,total);}
`;

const cornerPosition = [
  "vec3f(0,0,0)", "vec3f(1,0,0)", "vec3f(1,1,0)", "vec3f(0,1,0)",
  "vec3f(0,0,1)", "vec3f(1,0,1)", "vec3f(1,1,1)", "vec3f(0,1,1)",
] as const;

function emitShader(tetraIndex: number): string {
  const [a, b, c, d] = TETS[tetraIndex];
  return /* wgsl */ `
struct U{viewport:vec4f,cameraPosition:vec4f,cameraTarget:vec4f,container:vec4f,options:vec4f,gridInfo:vec4f,debug:vec4f}struct V{position:vec4f,normal:vec4f}struct A{vertexCount:atomic<u32>,instanceCount:u32,firstVertex:u32,firstInstance:u32,activeCubeCount:atomic<u32>,vertexAllocator:atomic<u32>}struct P{sample:vec4u,bricks:vec4u,table:vec4u,settings:vec4f,cell:vec4f,sizing:vec4f,physical:vec4f}
@group(0)@binding(0)var<uniform>u:U;@group(0)@binding(3)var<storage,read_write>out:array<V>;@group(0)@binding(4)var<storage,read_write>args:A;@group(0)@binding(5)var<storage,read>cubes:array<vec2u>;@group(0)@binding(6)var<storage,read>values:array<vec4f>;@group(0)@binding(7)var<storage,read_write>offsets:array<u32>;@group(0)@binding(10)var<uniform>p:P;
fn world(q:vec3f)->vec3f{let x=clamp((q-.5)/vec3f(p.sample.xyz),vec3f(0),vec3f(1));return vec3f(-.5*u.container.x,0,-.5*u.container.z)+x*u.container.xyz;}fn edge(x:vec3f,y:vec3f,a:f32,b:f32)->vec3f{return mix(x,y,clamp((.5-a)/(b-a),.02,.98));}fn countTet(a:f32,b:f32,c:f32,d:f32)->u32{let n=select(0u,1u,a>=.5)+select(0u,1u,b>=.5)+select(0u,1u,c>=.5)+select(0u,1u,d>=.5);if(n==0u||n==4u){return 0u;}return select(3u,6u,n==2u);}
fn tri(cursor:ptr<function,u32>,a:vec3f,b:vec3f,c:vec3f,n:vec3f){let first=*cursor;*cursor=first+3u;let limit=min(atomicLoad(&args.vertexCount),arrayLength(&out));if(first+3u>limit){return;}let x=V(vec4f(world(a),1),vec4f(n,0));let y=V(vec4f(world(b),1),vec4f(n,0));let z=V(vec4f(world(c),1),vec4f(n,0));out[first]=x;if(dot(cross(y.position.xyz-x.position.xyz,z.position.xyz-x.position.xyz),n)>=0.){out[first+1u]=y;out[first+2u]=z;}else{out[first+1u]=z;out[first+2u]=y;}}
fn tet(cursor:ptr<function,u32>,pa:vec3f,pb:vec3f,pc:vec3f,pd:vec3f,va:f32,vb:f32,vc:f32,vd:f32,n:vec3f){let m=select(0u,1u,va>=.5)|select(0u,2u,vb>=.5)|select(0u,4u,vc>=.5)|select(0u,8u,vd>=.5);if(m==1u||m==14u){tri(cursor,edge(pa,pb,va,vb),edge(pa,pc,va,vc),edge(pa,pd,va,vd),n);}else if(m==2u||m==13u){tri(cursor,edge(pb,pa,vb,va),edge(pb,pd,vb,vd),edge(pb,pc,vb,vc),n);}else if(m==4u||m==11u){tri(cursor,edge(pc,pa,vc,va),edge(pc,pb,vc,vb),edge(pc,pd,vc,vd),n);}else if(m==8u||m==7u){tri(cursor,edge(pd,pa,vd,va),edge(pd,pc,vd,vc),edge(pd,pb,vd,vb),n);}else if(m==3u||m==12u){let ac=edge(pa,pc,va,vc);let ad=edge(pa,pd,va,vd);let bc=edge(pb,pc,vb,vc);let bd=edge(pb,pd,vb,vd);tri(cursor,ac,bc,bd,n);tri(cursor,ac,bd,ad,n);}else if(m==5u||m==10u){let ab=edge(pa,pb,va,vb);let ad=edge(pa,pd,va,vd);let cb=edge(pc,pb,vc,vb);let cd=edge(pc,pd,vc,vd);tri(cursor,ab,cb,cd,n);tri(cursor,ab,cd,ad,n);}else if(m==6u||m==9u){let ba=edge(pb,pa,vb,va);let bd=edge(pb,pd,vb,vd);let ca=edge(pc,pa,vc,va);let cd=edge(pc,pd,vc,vd);tri(cursor,ba,ca,cd,n);tri(cursor,ba,cd,bd,n);}}
fn clipped(base:vec3f,scale:f32,q:vec3f,descriptor:u32)->vec3f{var r=base+scale*q;let mode=(descriptor>>8u)&3u;if(mode==1u){let high=((descriptor>>11u)&1u)!=0u;r.z=base.z+scale*select(.5*q.z,.5+.5*q.z,high);}else if(mode==2u){let high=((descriptor>>10u)&1u)!=0u;r.x=base.x+scale*select(.5*q.x,.5+.5*q.x,high);}return r;}
@compute @workgroup_size(64)fn emitGlobalFineTetra${tetraIndex}(@builtin(global_invocation_id)g:vec3u){let i=g.x;let count=min(atomicLoad(&args.activeCubeCount),min(arrayLength(&cubes),arrayLength(&offsets)/6u));if(i>=count||i*2u+1u>=arrayLength(&values)){return;}let packed=cubes[i];let descriptor=packed.y>>16u;let scale=f32(max(1u,descriptor&255u));let base=vec3f(f32(packed.x&0xffffu),f32(packed.y&0xffffu),f32(packed.x>>16u));let lo=values[i*2u];let hi=values[i*2u+1u];let v0=lo.x;let v1=lo.y;let v2=lo.z;let v3=lo.w;let v4=hi.x;let v5=hi.y;let v6=hi.z;let v7=hi.w;let gx=.25*((v1+v2+v5+v6)-(v0+v3+v4+v7));let gy=.25*((v2+v3+v6+v7)-(v0+v1+v4+v5));let gz=.25*((v4+v5+v6+v7)-(v0+v1+v2+v3));let size=vec3f(p.sample.xyz);let gradient=vec3f(gx*size.x/u.container.x,gy*size.y/u.container.y,gz*size.z/u.container.z);var n=vec3f(0,1,0);if(length(gradient)>1e-5){n=-normalize(gradient);}var cursor=offsets[i*6u+${tetraIndex}u];tet(&cursor,clipped(base,scale,${cornerPosition[a]},descriptor),clipped(base,scale,${cornerPosition[b]},descriptor),clipped(base,scale,${cornerPosition[c]},descriptor),clipped(base,scale,${cornerPosition[d]},descriptor),v${a},v${b},v${c},v${d},n);}
`;
}

export const globalFineClassifiedEmitShaders = TETS.map((_, index) => emitShader(index));

/** One bounded 2-D dispatch: x selects a classified cube, y one of six exact tetrahedra. */
export const globalFineClassifiedEmitShader = emitShader(0).replace(
  /@compute @workgroup_size\(64\)fn emitGlobalFineTetra0[\s\S]*$/,
  `@compute @workgroup_size(64)fn emitGlobalFineTetrahedra(@builtin(global_invocation_id)g:vec3u){let i=g.x;let tetrahedron=g.y;if(tetrahedron>=6u){return;}let count=min(atomicLoad(&args.activeCubeCount),min(arrayLength(&cubes),arrayLength(&offsets)/6u));if(i>=count||i*2u+1u>=arrayLength(&values)){return;}let packed=cubes[i];let descriptor=packed.y>>16u;let scale=f32(max(1u,descriptor&255u));let base=vec3f(f32(packed.x&0xffffu),f32(packed.y&0xffffu),f32(packed.x>>16u));let lo=values[i*2u];let hi=values[i*2u+1u];let samples=array<f32,8>(lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w);let positions=array<vec3f,8>(clipped(base,scale,vec3f(0,0,0),descriptor),clipped(base,scale,vec3f(1,0,0),descriptor),clipped(base,scale,vec3f(1,1,0),descriptor),clipped(base,scale,vec3f(0,1,0),descriptor),clipped(base,scale,vec3f(0,0,1),descriptor),clipped(base,scale,vec3f(1,0,1),descriptor),clipped(base,scale,vec3f(1,1,1),descriptor),clipped(base,scale,vec3f(0,1,1),descriptor));let ids=array<vec4u,6>(vec4u(0,1,2,6),vec4u(0,2,3,6),vec4u(0,3,7,6),vec4u(0,7,4,6),vec4u(0,5,6,4),vec4u(0,5,1,6))[tetrahedron];let v0=lo.x;let v1=lo.y;let v2=lo.z;let v3=lo.w;let v4=hi.x;let v5=hi.y;let v6=hi.z;let v7=hi.w;let gx=.25*((v1+v2+v5+v6)-(v0+v3+v4+v7));let gy=.25*((v2+v3+v6+v7)-(v0+v1+v4+v5));let gz=.25*((v4+v5+v6+v7)-(v0+v1+v2+v3));let size=vec3f(p.sample.xyz);let gradient=vec3f(gx*size.x/u.container.x,gy*size.y/u.container.y,gz*size.z/u.container.z);var n=vec3f(0,1,0);if(length(gradient)>1e-5){n=-normalize(gradient);}var cursor=offsets[i*6u+tetrahedron];tet(&cursor,positions[ids.x],positions[ids.y],positions[ids.z],positions[ids.w],samples[ids.x],samples[ids.y],samples[ids.z],samples[ids.w],n);}`,
);
