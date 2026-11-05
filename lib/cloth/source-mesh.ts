import type {ClothVector} from './types';
import type {WetClothSourceField} from './source-types';
import {rayClothTriangle} from './geometry';

export const SPOT_SOURCE_ASSET={filename:'assets/meshes/spot_triangulated.obj',url:'/assets/wetcloth/meshes/spot_triangulated.obj',
 sha256:'0738b5e8608fed74e5e8c7aa8dd0af97b4b74f9f6cbf7aac84cd7e40b2e44a75',
 origin:'https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/spot.zip',license:'CC0-1.0'} as const;
type V=ClothVector;
const sub=(a:V,b:V):V=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const dot=(a:V,b:V)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a:V,b:V):V=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const add=(a:V,b:V,t=1):V=>[a[0]+t*b[0],a[1]+t*b[1],a[2]+t*b[2]];
function triangleDistance2(p:V,a:V,b:V,c:V):number{
 const ab=sub(b,a),ac=sub(c,a),ap=sub(p,a),d1=dot(ab,ap),d2=dot(ac,ap);
 if(d1<=0&&d2<=0)return dot(ap,ap);
 const bp=sub(p,b),d3=dot(ab,bp),d4=dot(ac,bp);if(d3>=0&&d4<=d3)return dot(bp,bp);
 const vc=d1*d4-d3*d2;if(vc<=0&&d1>=0&&d3<=0){const q=sub(p,add(a,ab,d1/(d1-d3)));return dot(q,q);}
 const cp=sub(p,c),d5=dot(ab,cp),d6=dot(ac,cp);if(d6>=0&&d5<=d6)return dot(cp,cp);
 const vb=d5*d2-d1*d6;if(vb<=0&&d2>=0&&d6<=0){const q=sub(p,add(a,ac,d2/(d2-d6)));return dot(q,q);}
 const va=d3*d6-d5*d4;if(va<=0&&d4-d3>=0&&d5-d6>=0){const q=sub(p,add(b,sub(c,b),(d4-d3)/(d4-d3+d5-d6)));return dot(q,q);}
 const denom=1/(va+vb+vc),q=sub(p,add(add(a,ab,vb*denom),ac,vc*denom));return dot(q,q);
}
interface Node {min:V;max:V;ids?:number[];children?:[Node,Node]}
/** Exact closed-triangle query for baking source SDFs; never a replacement
 * bounding box. Open, nonmanifold or degenerate input fails explicitly. */
export class ClosedSourceMesh {
 readonly vertices:V[]=[];readonly triangles:[number,number,number][]=[];readonly centre:V;readonly bounds:{min:V;max:V};private root:Node;
 constructor(obj:string){
  for(const line of obj.split(/\r?\n/)){
   const [kind,...v]=line.trim().split(/\s+/);
   if(kind==='v'){const p=v.slice(0,3).map(Number);if(p.length!==3||!p.every(Number.isFinite))throw new Error('Invalid OBJ vertex');this.vertices.push(p as unknown as V);}
   else if(kind==='f'){
    if(v.length!==3)throw new Error('Source mesh must already be triangulated');
    const f=v.map(s=>{const n=Number(s.split('/')[0]);return n>0?n-1:this.vertices.length+n;});
    if(f.some(n=>!Number.isInteger(n)||n<0||n>=this.vertices.length)||new Set(f).size!==3)throw new Error('Invalid OBJ triangle');
    this.triangles.push(f as [number,number,number]);
   }
  }
  if(!this.vertices.length||!this.triangles.length)throw new Error('Empty source mesh');
  const edges=new Map<string,{count:number;orientation:number}>();
  for(const ids of this.triangles){
   const [a,b,c]=ids.map(i=>this.vertices[i]!) as [V,V,V];if(Math.hypot(...cross(sub(b,a),sub(c,a)))<1e-14)throw new Error('Degenerate source triangle');
   for(let j=0;j<3;j++){const a=ids[j]!,b=ids[(j+1)%3]!,key=`${Math.min(a,b)},${Math.max(a,b)}`,e=edges.get(key)??{count:0,orientation:0};e.count++;e.orientation+=a<b?1:-1;edges.set(key,e);}
  }
  if([...edges.values()].some(e=>e.count!==2||e.orientation!==0))throw new Error('Source mesh must be closed and consistently oriented');
  this.centre=[0,1,2].map(a=>this.vertices.reduce((s,p)=>s+p[a]!,0)/this.vertices.length) as unknown as V;
  const build=(ids:number[]):Node=>{
   const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
   for(const t of ids)for(const i of this.triangles[t]!)for(let a=0;a<3;a++){min[a]=Math.min(min[a]!,this.vertices[i]![a]!);max[a]=Math.max(max[a]!,this.vertices[i]![a]!);}
   const node:Node={min:min as unknown as V,max:max as unknown as V};
   if(ids.length<=8)node.ids=ids;else{const axis=max.map((v,a)=>v-min[a]!).indexOf(Math.max(...max.map((v,a)=>v-min[a]!)));ids.sort((a,b)=>this.triangles[a]!.reduce((s,i)=>s+this.vertices[i]![axis]!,0)-this.triangles[b]!.reduce((s,i)=>s+this.vertices[i]![axis]!,0));const mid=ids.length>>1;node.children=[build(ids.slice(0,mid)),build(ids.slice(mid))];}
   return node;
  };
  this.root=build(this.triangles.map((_,i)=>i));this.bounds={min:this.root.min,max:this.root.max};
 }
 signedDistance(p:V):number{
  const lower=(n:Node)=>n.min.reduce((s,v,a)=>s+Math.max(v-p[a]!,0,p[a]!-n.max[a]!)**2,0);
  let best=Infinity;const stack=[this.root];
  while(stack.length){const n=stack.pop()!;if(lower(n)>=best)continue;
   if(n.ids)for(const t of n.ids){const [a,b,c]=this.triangles[t]!.map(i=>this.vertices[i]!) as [V,V,V];best=Math.min(best,triangleDistance2(p,a,b,c));}
   else {const [a,b]=n.children!;if(lower(a)<lower(b))stack.push(b,a);else stack.push(a,b);}
  }
  if(best<1e-28)return 0;
  let crossings=0,ambiguous=false;stack.push(this.root);
  while(stack.length){const n=stack.pop()!;if(n.max[0]<p[0]||p[1]<n.min[1]||p[1]>n.max[1]||p[2]<n.min[2]||p[2]>n.max[2])continue;
   if(n.ids)for(const t of n.ids){const [a,b,c]=this.triangles[t]!.map(i=>this.vertices[i]!) as [V,V,V],hit=rayClothTriangle({origin:p,direction:[1,0,0]},a,b,c);if(hit){crossings++;if(hit.coordinates.some(v=>v<1e-9))ambiguous=true;}}
   else stack.push(...n.children!);
  }
  let inside=(crossings&1)!==0;
  // A ray through an edge has an ambiguous parity; solid angle remains
  // well-defined at every point whose closest distance is nonzero.
  if(ambiguous){let angle=0;for(const ids of this.triangles){const [a,b,c]=ids.map(i=>sub(this.vertices[i]!,p)) as [V,V,V],la=Math.hypot(...a),lb=Math.hypot(...b),lc=Math.hypot(...c);angle+=2*Math.atan2(dot(a,cross(b,c)),la*lb*lc+dot(a,b)*lc+dot(b,c)*la+dot(c,a)*lb);}inside=Math.abs(angle)>2*Math.PI;}
  return (inside?-1:1)*Math.sqrt(best);
 }
}
const meshes=new Map<string,ClosedSourceMesh>();
export async function loadSourceMesh(filename:string,request:typeof fetch=fetch):Promise<ClosedSourceMesh>{
 const held=meshes.get(filename);if(held)return held;
 if(filename!==SPOT_SOURCE_ASSET.filename)throw new Error(`Unknown external source geometry: ${filename}`);
 const response=await request(SPOT_SOURCE_ASSET.url);if(!response.ok)throw new Error(`Source mesh load failed: ${response.status}`);
 const bytes=await response.arrayBuffer(),hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
 if(hash!==SPOT_SOURCE_ASSET.sha256)throw new Error('Source mesh SHA-256 mismatch');
 const mesh=new ClosedSourceMesh(new TextDecoder().decode(bytes));meshes.set(filename,mesh);return mesh;
}
export function cachedSourceMesh(filename:string):ClosedSourceMesh{const mesh=meshes.get(filename);if(!mesh)throw new Error(`External source mesh not loaded: ${filename}`);return mesh;}

interface MeshGrid {mesh:ClosedSourceMesh;scale:number;origin:V;size:V;spacing:number;values:Float64Array}
const grids=new Map<string,MeshGrid>();
function fieldGrid(f:WetClothSourceField):MeshGrid{
 const key=`${f.filename}:${f.scale}:${f.sampleSpacing_m}`,held=grids.get(key);if(held)return held;
 const mesh=cachedSourceMesh(f.filename!),scale=(f.scale??1)*.01,spacing=f.sampleSpacing_m!;
 if(!(scale>0&&spacing>0))throw new Error('Mesh source requires positive scale and source sample spacing');
 const origin=mesh.bounds.min.map(v=>v*scale-3*spacing) as unknown as V;
 const size=mesh.bounds.max.map((v,a)=>Math.ceil((v*scale+3*spacing-origin[a]!)/spacing)) as unknown as V;
 const values=new Float64Array(size[0]*size[1]*size[2]).fill(NaN),grid={mesh,scale,origin,size,spacing,values};grids.set(key,grid);return grid;
}
const rotate=(p:V,axis:V,angle:number):V=>{
 const len=Math.hypot(...axis);if(!angle)return p;if(!len)throw new Error('Invalid mesh rotation');
 const a=axis.map(v=>v/len) as unknown as V,c=Math.cos(angle),s=Math.sin(angle);return add(add(p.map(v=>v*c) as unknown as V,cross(a,p),s),a,dot(a,p)*(1-c));
};
/** Upstream file fields recenter about the vertex mean, retain that mean in
 * the world translation, then trilinearly sample a dx-spaced signed field. */
export function sourceMeshDistance(f:WetClothSourceField,world:V):number{
 const g=fieldGrid(f),mean=g.mesh.centre.map(v=>v*g.scale) as unknown as V;
 const local=add(rotate(sub(world,add(f.center_m??[0,0,0],mean)),f.axis??[0,1,0],-(f.angle_rad??0)),mean);
 const q=local.map((v,a)=>Math.max(0,Math.min(g.size[a]!-1-1e-12,(v-g.origin[a]!)/g.spacing))),base=q.map(Math.floor),w=q.map((v,a)=>v-base[a]!);
 let result=0;
 for(let k=0;k<8;k++){
  const bit=[k&1,(k>>1)&1,(k>>2)&1],p=base.map((v,a)=>v+bit[a]!),i=p[0]!+g.size[0]*(p[1]!+g.size[1]*p[2]!);
  if(Number.isNaN(g.values[i]!))g.values[i]=g.mesh.signedDistance(p.map((v,a)=>(g.origin[a]!+v*g.spacing)/g.scale) as unknown as V)*g.scale;
  result+=g.values[i]!*bit.reduce((s,b,a)=>s*(b?w[a]!:1-w[a]!),1);
 }
 return f.inside?-result:result;
}
export function sourceMeshBounds(f:WetClothSourceField):{min:V;max:V}{
 const g=fieldGrid(f),mean=g.mesh.centre.map(v=>v*g.scale) as unknown as V,min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
 for(const p of g.mesh.vertices){const world=add(add(f.center_m??[0,0,0],mean),rotate(sub(p.map(v=>v*g.scale) as unknown as V,mean),f.axis??[0,1,0],f.angle_rad??0));for(let a=0;a<3;a++){min[a]=Math.min(min[a]!,world[a]!);max[a]=Math.max(max[a]!,world[a]!);}}
 return {min:min as unknown as V,max:max as unknown as V};
}
