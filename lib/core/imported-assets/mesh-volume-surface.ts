import { MeshoptSimplifier } from "meshoptimizer";
import type { MeshVolume } from './mesh-volume';
import { marchingCubesLookupWGSL } from '../marching-cubes-lookup.wgsl';

// Decode the canonical packed marching-cubes cases once, shared with the GPU
// extractor. Each edge vertex is owned by its lattice edge, including seams.
const cases = (name: string) => [...marchingCubesLookupWGSL.split(`const ${name}=`)[1]!.split(');')[0]!.matchAll(/0x([0-9a-f]+)u/g)].map(m => parseInt(m[1]!, 16));
const lo = cases('MC_TRI_LO'), hi = cases('MC_TRI_HI');
const corners = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]];
const edgeA = [0,1,2,3,4,5,6,7,0,1,2,3], edgeB = [1,2,3,0,5,6,7,4,4,5,6,7];
export interface MeshVolumeSurface { positions: Float32Array<ArrayBuffer>; indices: Uint32Array<ArrayBuffer>; levels: readonly { first: number; count: number; error: number }[] }
let simplifierReady = false;
export async function prepareMeshVolumeSurfaces(): Promise<void> { await MeshoptSimplifier.ready; simplifierReady = true; }
const cache = new WeakMap<MeshVolume, { surface: MeshVolumeSurface; prepared: boolean }>();
/** Immutable source-space isosurface at the asset's native sample spacing.
 * Instances share this geometry. Shading still samples the original normal
 * and material channels, so geometry extraction cannot blur the PBR palette. */
export function meshVolumeSurface(volume: MeshVolume): MeshVolumeSurface {
  const held = cache.get(volume);
  if (held && (!simplifierReady || held.prepared)) return held.surface;
  const h = volume.header, positions: number[] = [], indices: number[] = [], edges = new Map<number, number>();
  const nx = h.dimensions[0] + 1, ny = h.dimensions[1] + 1;
  const values = new Float32Array(8), ids = new Uint32Array(12);
  const planes = new Map<string, { axis: number; coordinate: number; direction: number; cells: Set<number> }>();
  for (const brick of volume.bricks.values()) {
    for (let z=0;z<8;z++) for(let y=0;y<8;y++) for(let x=0;x<8;x++) {
      let mask=0;
      for(let c=0;c<8;c++) { const q=corners[c]!; const d=brick.distance[x+q[0]!+9*(y+q[1]!+9*(z+q[2]!))]!; values[c]=d; if(d<0) mask|=1<<c; }
      if(mask===0||mask===255) continue;
      // Exact axis-aligned planes need no interior tessellation. Preserve all
      // nonplanar cells at native resolution; PBR/normal lookup is independent
      // of this geometry, including on a merged paving or fence rectangle.
      let planar = false;
      for (let axis=0;axis<3&&!planar;axis++) {
        const end = corners.findIndex(q=>q[axis]===1&&q.every((v,j)=>j===axis||v===0));
        const a=values[0]!, b=values[end]!;
        if ((a<0)===(b<0)) continue;
        if (!corners.every((q,c)=>Math.abs(values[c]!-(q[axis]===0?a:b))<h.cellSize*1e-6)) continue;
        const coordinate=brick.origin[axis]!+[x,y,z][axis]!+a/(a-b);
        const key=`${axis}:${coordinate.toFixed(7)}`;
        let plane=planes.get(key); if(!plane) {plane={axis,coordinate,direction:Math.sign(b-a),cells:new Set()};planes.set(key,plane);}
        const u=(axis+1)%3,v=(axis+2)%3;
        plane.cells.add(brick.origin[u]!+[x,y,z][u]!+h.dimensions[u]!*(brick.origin[v]!+[x,y,z][v]!));
        planar=true;
      }
      if(planar) continue;
      const count=hi[mask]!>>>28;
      let visited=0;
      for(let i=0;i<count;i++) {
        const e=((i<8?lo[mask]!:hi[mask]!)>>>((i%8)*4))&15;
        if(!(visited&(1<<e))) {
          visited|=1<<e;
          const a=edgeA[e]!, b=edgeB[e]!, ca=corners[a]!, cb=corners[b]!;
          const axis=ca.findIndex((v,j)=>v!==cb[j]);
          const gx=brick.origin[0]+x+Math.min(ca[0]!,cb[0]!), gy=brick.origin[1]+y+Math.min(ca[1]!,cb[1]!), gz=brick.origin[2]+z+Math.min(ca[2]!,cb[2]!);
          const key=(gx+nx*(gy+ny*gz))*3+axis;
          let index=edges.get(key);
          if(index===undefined) {
            index=positions.length/3; edges.set(key,index);
            const t=values[a]!/(values[a]!-values[b]!);
            for(let j=0;j<3;j++) positions.push((brick.origin[j]!+[x,y,z][j]!+ca[j]!+t*(cb[j]!-ca[j]!)-h.dimensions[j]!/2)*h.cellSize);
          }
          ids[e]=index;
        }
        indices.push(ids[e]!);
      }
      // The shared MC table winds toward negative values. Hardware back-face
      // rejection needs the outward (positive-distance) orientation.
      for(let at=indices.length-count;at<indices.length;at+=3){const swap=indices[at+1]!;indices[at+1]=indices[at+2]!;indices[at+2]=swap;}
    }
  }
  for(const {axis,coordinate,direction,cells} of planes.values()) {
    const u=(axis+1)%3,v=(axis+2)%3,width=h.dimensions[u]!;
    for(const start of [...cells].sort((a,b)=>a-b)) {
      if(!cells.has(start)) continue;
      const x=start%width,y=Math.floor(start/width);let w=1,rows=1;
      while(x+w<width&&cells.has(start+w))w++;
      outer: while(y+rows<h.dimensions[v]!) {for(let k=0;k<w;k++)if(!cells.has(start+rows*width+k))break outer;rows++;}
      for(let j=0;j<rows;j++)for(let k=0;k<w;k++)cells.delete(start+j*width+k);
      const base=positions.length/3;
      for(const [du,dv] of [[0,0],[w,0],[0,rows],[w,rows]]) {
        const q=[0,0,0];q[axis]=coordinate;q[u]=x+du!;q[v]=y+dv!;
        positions.push(...q.map((value,j)=>(value-h.dimensions[j]!/2)*h.cellSize));
      }
      if(direction>0)indices.push(base,base+1,base+2,base+2,base+1,base+3);
      else indices.push(base,base+2,base+1,base+2,base+3,base+1);
    }
  }
  const points=new Float32Array(positions), native=new Uint32Array(indices);
  const levels=[{first:0,count:native.length,error:0}], buffers=[native];let length=native.length;
  // Keep the full native mesh. Lower-detail index streams are selected only
  // when their projected error is below a quarter of a pixel. Normal/PBR
  // sampling always uses the original field, independent of the chosen level.
  if(simplifierReady&&native.length>6) for(const error of [.000025,.0001,.0004,.0016,.003125,.00625]) {
    const [simplified, actual]=MeshoptSimplifier.simplify(native,points,3,0,error,['ErrorAbsolute','LockBorder']);
    if(simplified.length>=levels.at(-1)!.count||simplified.length===0)continue;
    levels.push({first:length,count:simplified.length,error:Math.max(error,actual)});buffers.push(new Uint32Array(simplified));length+=simplified.length;
  }
  const combined=new Uint32Array(length);let offset=0;for(const buffer of buffers){combined.set(buffer,offset);offset+=buffer.length;}
  const surface={positions:points,indices:combined,levels};
  cache.set(volume,{surface,prepared:simplifierReady}); return surface;
}
