import type { SolidWorld } from "../../../core/solid-world";
import { SOLID_WORLD_BRICK_CELLS as N } from "../../../core/solid-world";
import type { SparseSceneAxisAlignedBounds } from "../../../core/webgpu-sparse-scene-proxies";

/** Uniform occupancy proofs over composed pages. Prefix sums also prove
 * partial-page slabs, avoiding an entire 400 mm page of finest render cells
 * around a 50 mm physics boundary. Clears and fractional cells block the
 * solid proof. No source geometry or fluid data is altered. */
export function buriedSolidClassifier(world: SolidWorld,
  sourceOrigin: readonly number[], sourceCell: readonly number[],
  renderOrigin: readonly number[], nodeEdges: readonly (readonly number[])[], brickSize: number,
  blockers: readonly (SparseSceneAxisAlignedBounds & { axisAlignedBox?: boolean;
    horizontalSlab?: { minimum: readonly number[]; maximum: readonly number[] };
    nativeSurface?: boolean;
    intersectsSupport?: (minimum: readonly number[], maximum: readonly number[]) => boolean })[] = [],
): (level: number, coordinate: { x: number; y: number; z: number }) => boolean {
  const pages = new Map<string, { full: Uint16Array; any: Uint16Array }>();
  const stride=N+1, index=(x:number,y:number,z:number)=>x+stride*(y+stride*z);
  if (!world.regions?.length && !world.ground) for (const page of world.pages) {
    const full=new Uint16Array(stride**3), any=new Uint16Array(stride**3);
    for(let z=1;z<=N;z++)for(let y=1;y<=N;y++)for(let x=1;x<=N;x++){
      const fraction=page.solidFraction[x-1+N*(y-1+N*(z-1))]!;
      for(const [table,value] of [[full,Number(fraction===255)],[any,Number(fraction>0)]] as const){
        table[index(x,y,z)]=value+table[index(x-1,y,z)]!+table[index(x,y-1,z)]!+table[index(x,y,z-1)]!
          -table[index(x-1,y-1,z)]!-table[index(x-1,y,z-1)]!-table[index(x,y-1,z-1)]!+table[index(x-1,y-1,z-1)]!;
      }
    }
    pages.set(page.coordinate.join(","),{full,any});
  }
  const sum=(table:Uint16Array,lo:number[],hi:number[])=>{
    let value=0;
    for(let k=0;k<8;k++){
      const p=[k&1,k>>1&1,k>>2];
      value+=((p[0]!+p[1]!+p[2]!)%2?1:-1)*table[index(p[0]?hi[0]!:lo[0]!,p[1]?hi[1]!:lo[1]!,p[2]?hi[2]!:lo[2]!)]!;
    }return value;
  };
  return (level, coordinate) => {
    if (!pages.size) return false;
    const p=[coordinate.x,coordinate.y,coordinate.z],edge=nodeEdges[level]!;
    // Two fit cells plus the four-tap cubic interpolation stencil.
    const min=p.map((v,a)=>renderOrigin[a]!+v*edge[a]!-2*edge[a]!/brickSize);
    const max=p.map((v,a)=>renderOrigin[a]!+(v+1)*edge[a]!+2*edge[a]!/brickSize);
    // Only page interpolation reads the source-grid apron. Analytic props
    // need the fit-cell apron; expanding them by the fluid spacing needlessly
    // refines whole volumes around a thin chair, canopy or patio seam.
    const lo=min.map((v,a)=>Math.floor((v-sourceOrigin[a]!)/sourceCell[a]!-2.5));
    const hi=max.map((v,a)=>Math.floor((v-sourceOrigin[a]!)/sourceCell[a]!+2.5)+1);
    const classify=(lo:number[],hi:number[])=>{
      const first=lo.map(v=>Math.floor(v/N)),last=hi.map(v=>Math.floor((v-1)/N));
      let allFull=true,allAir=true;
      for(let z=first[2]!;z<=last[2]!;z++)for(let y=first[1]!;y<=last[1]!;y++)for(let x=first[0]!;x<=last[0]!;x++){
        const page=pages.get(`${x},${y},${z}`);
        if(!page){allFull=false;}else{
          const base=[x*N,y*N,z*N],a=lo.map((v,i)=>Math.max(0,v-base[i]!)),b=hi.map((v,i)=>Math.min(N,v-base[i]!));
          const count=b.reduce((n,v,i)=>n*(v-a[i]!),1);
          allFull=allFull&&sum(page.full,a,b)===count;
          allAir=allAir&&sum(page.any,a,b)===0;
        }
        if(!allFull&&!allAir)return 0;
      }
      return allFull?1:allAir?-1:0;
    };
    const state=classify(lo,hi);
    if(state===1)return true;
    // Imported surfaces are drawn from their native SDF, not these DMC cells.
    // Their redundant world occupancy/shadow proxy may be at most 4x coarser,
    // capped at 25 mm so ordinary render settings do not lose their shadows.
    // Ordinary edited geometry underneath still has to pass the proofs below.
    const nativeProxyLevel=level>=nodeEdges.length-3&&Math.max(...edge)/brickSize<=.025+1e-9;
    const touching=blockers.filter(b=>!(b.nativeSurface&&nativeProxyLevel)
      &&b.maximum.every((v,a)=>v>=min[a]!)&&b.minimum.every((v,a)=>v<=max[a]!)
      &&(b.intersectsSupport?.(min,max)??true));
    if(state===-1&&!touching.length)return true;
    // Flat sets have no detail to gain along their tangent axes. Keep them at
    // most two levels coarser than the requested surface lattice, within the
    // DMC transition encoding's 4:1 bound. Box edges, joints, fractional edits,
    // curved props and foliage still descend to the finest cells.
    if(level<nodeEdges.length-3||touching.some(b=>!b.axisAlignedBox&&!b.horizontalSlab))return false;
    for(let axis=0;axis<3;axis++){
      if(touching.some(b=>b.horizontalSlab&&axis!==1))continue;
      if(touching.some(b=>{const flat=b.horizontalSlab??b;return [0,1,2].some(a=>a!==axis&&(min[a]!<flat.minimum[a]!-1e-7||max[a]!>flat.maximum[a]!+1e-7));}))continue;
      // A single monotone crossing survives a coarser lattice. Two faces of a
      // thin slab (or a gap) do not: both can fall between the same samples.
      let flat=true, previous=0, transitions=0, solidStart:number|undefined;
      const intervals: [number,number][]=[];
      if(state===0)for(let i=lo[axis]!;i<hi[axis]!;i++){
        const a=[...lo],b=[...hi];a[axis]=i;b[axis]=i+1;
        const slice=classify(a,b);
        if(slice===0){flat=false;break;}
        const boundary=sourceOrigin[axis]!+i*sourceCell[axis]!;
        if(slice===1&&previous!==1)solidStart=boundary;
        if(slice===-1&&previous===1)intervals.push([solidStart!,boundary]);
        if(previous!==0&&slice!==previous&&++transitions>1){flat=false;break;}
        previous=slice;
      }
      if(!flat)continue;
      if(previous===1)intervals.push([solidStart!,sourceOrigin[axis]!+hi[axis]!*sourceCell[axis]!]);
      for(const b of touching){const flat=b.horizontalSlab??b;intervals.push([flat.minimum[axis]!,flat.maximum[axis]!]);}
      intervals.sort((a,b)=>a[0]-b[0]);
      const merged:[number,number][]=[];
      for(const span of intervals){
        const last=merged.at(-1);
        if(last&&span[0]<=last[1]+1e-7)last[1]=Math.max(last[1],span[1]);
        else merged.push([...span]);
      }
      // A patio slab joined to the source ground is one half-space. An
      // isolated slab or a gap still has two crossings and must stay fine.
      let crossings=0;
      for(const span of merged)for(const boundary of span){
        if(boundary>min[axis]!-1e-7&&boundary<max[axis]!+1e-7)crossings++;
      }
      if(crossings<=1)return true;
    }
    return false;
  };
}
