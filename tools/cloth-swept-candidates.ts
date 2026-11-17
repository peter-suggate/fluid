/** CPU diagnostic broadphase for complete affine sweeps. Independent of OGC's
 * envelopes/candidate cache. No fixed queue capacity and no silent truncation. */
import type {AffineVertex} from './cloth-affine-ccd';
type Box={lo:number[];hi:number[]};
type Node=Box & {left?:Node;right?:Node;ids?:number[]};
const roundingBits=new DataView(new ArrayBuffer(8));
function nextUp(x:number):number {
  if(x===Infinity)return x;if(x===0)return Number.MIN_VALUE;
  roundingBits.setFloat64(0,x,false);const bits=roundingBits.getBigUint64(0,false);
  roundingBits.setBigUint64(0,x>0?bits+1n:bits-1n,false);return roundingBits.getFloat64(0,false);
}
/** Outward bounds on the exact affine endpoint sum and thickness expansion.
 * Each rounded arithmetic operation is expanded by one representable value. */
export function affineAxisBounds(x:number,d:number,thickness:number):readonly [number,number] {
  if(!Number.isFinite(x)||!Number.isFinite(d)||!Number.isFinite(thickness)||thickness<0)throw new Error('Nonfinite affine sweep');
  const y=x+d,lo=-nextUp(-(Math.min(x,-nextUp(-y))-thickness)),hi=nextUp(Math.max(x,nextUp(y))+thickness);
  if(!Number.isFinite(lo)||!Number.isFinite(hi))throw new Error('Unrepresentable swept bound');
  return [lo,hi];
}
const empty=():Box=>({lo:[Infinity,Infinity,Infinity],hi:[-Infinity,-Infinity,-Infinity]});
function include(a:Box,b:Box){for(let k=0;k<3;k++){a.lo[k]=Math.min(a.lo[k]!,b.lo[k]!);a.hi[k]=Math.max(a.hi[k]!,b.hi[k]!);}}
function overlap(a:Box,b:Box){for(let k=0;k<3;k++)if(a.lo[k]!>b.hi[k]!||b.lo[k]!>a.hi[k]!)return false;return true;}
function tree(boxes:Box[],ids:number[]):Node {
  const bounds=empty();for(const i of ids)include(bounds,boxes[i]!);
  if(ids.length<=8)return {...bounds,ids};
  let axis=0;for(let k=1;k<3;k++)if(bounds.hi[k]!-bounds.lo[k]!>bounds.hi[axis]!-bounds.lo[axis]!)axis=k;
  ids.sort((i,j)=>(boxes[i]!.lo[axis]!+boxes[i]!.hi[axis]!)-(boxes[j]!.lo[axis]!+boxes[j]!.hi[axis]!));
  const mid=ids.length>>>1;
  return {...bounds,left:tree(boxes,ids.slice(0,mid)),right:tree(boxes,ids.slice(mid))};
}
function* query(root:Node,b:Box,boxes:Box[]):Generator<number>{
  const stack=[root];while(stack.length){const n=stack.pop()!;if(!overlap(n,b))continue;
    if(n.ids){for(const i of n.ids)if(overlap(boxes[i]!,b))yield i;}
    else stack.push(n.left!,n.right!);
  }
}
export type SweptCandidate={kind:'vt'|'ee';ids:readonly [number,number,number,number]};

/** Topology arrays use tightly packed triples/pairs. Only incident primitives
 * are excluded. Each EE pair is emitted once; every directed VT is covered. */
export function* clothSweptCandidates(vertices:readonly AffineVertex[],triangles:readonly number[],edges:readonly number[],thickness:number):Generator<SweptCandidate>{
  if(!Number.isFinite(thickness)||thickness<0||triangles.length%3||edges.length%2)throw new Error('Invalid swept candidate input');
  for(const i of [...triangles,...edges])if(!Number.isInteger(i)||i<0||i>=vertices.length)throw new Error('Invalid swept topology index');
  const vertexBoxes=vertices.map(p=>{
    const b=empty();for(let k=0;k<3;k++){
      [b.lo[k],b.hi[k]]=affineAxisBounds(p.x[k]!,p.d[k]!,thickness);
    }return b;
  });
  const boxes=(indices:readonly number[],arity:number)=>Array.from({length:indices.length/arity},(_,i)=>{
    const b=empty();for(let k=0;k<arity;k++)include(b,vertexBoxes[indices[arity*i+k]!]!);return b;
  });
  const tb=boxes(triangles,3),eb=boxes(edges,2),tt=tree(tb,tb.map((_,i)=>i)),et=tree(eb,eb.map((_,i)=>i));
  for(let v=0;v<vertices.length;v++)for(const t of query(tt,vertexBoxes[v]!,tb)){
    const a=triangles[3*t]!,b=triangles[3*t+1]!,c=triangles[3*t+2]!;
    if(v!==a&&v!==b&&v!==c)yield {kind:'vt',ids:[v,a,b,c]};
  }
  for(let i=0;i<eb.length;i++)for(const j of query(et,eb[i]!,eb)){
    if(j<=i)continue;
    const a=edges[2*i]!,b=edges[2*i+1]!,c=edges[2*j]!,d=edges[2*j+1]!;
    if(a!==c&&a!==d&&b!==c&&b!==d)yield {kind:'ee',ids:[a,b,c,d]};
  }
}
