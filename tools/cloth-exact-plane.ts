/** Independent exact support comparisons for OFFLINE affine CCD validation.
 * All finite IEEE inputs are dyadic rationals. BigInt arithmetic avoids an
 * assumed floating-point distance/normalization error bound. */
import type {AffineVertex,Vec3} from './cloth-affine-ccd';
type D={n:bigint;e:number};
const zero:D={n:0n,e:0};
const bits=new DataView(new ArrayBuffer(8));
function exact(x:number):D {
  if(!Number.isFinite(x))throw new Error('Nonfinite exact plane input');
  bits.setFloat64(0,x,false);const word=bits.getBigUint64(0,false),exponent=Number((word>>52n)&2047n),fraction=word&((1n<<52n)-1n);
  const n=(exponent?fraction|(1n<<52n):fraction)*(word>>63n?-1n:1n);
  return n===0n?zero:{n,e:exponent?exponent-1023-52:-1074};
}
const add=(a:D,b:D):D=>{if(a.n===0n)return b;if(b.n===0n)return a;const e=Math.min(a.e,b.e);return {n:(a.n<<BigInt(a.e-e))+(b.n<<BigInt(b.e-e)),e};};
const neg=(a:D):D=>({n:-a.n,e:a.e});
const sub=(a:D,b:D)=>add(a,neg(b));
const mul=(a:D,b:D):D=>({n:a.n*b.n,e:a.e+b.e});
const compare=(a:D,b:D)=>sub(a,b).n;
const min=(values:D[])=>values.reduce((a,b)=>compare(a,b)<=0n?a:b);
const max=(values:D[])=>values.reduce((a,b)=>compare(a,b)>=0n?a:b);
const dot=(a:D[],b:D[])=>a.reduce((sum,v,i)=>add(sum,mul(v,b[i]!)),zero);

/** Axis selection may use floating point: ANY finite nonzero axis can serve as
 * a witness. Original x+t*d inputs, not rounded relative positions, are tested.
 * Positive gap is required before squaring, and orientation is consistent. */
export function prepareExactAffinePlaneCheck(a:readonly AffineVertex[],b:readonly AffineVertex[],thickness:number,sameTimeEndpoints=false){
  if(!a.length||!b.length||!Number.isFinite(thickness)||thickness<0)throw new Error('Invalid exact plane input');
  const pack=(p:AffineVertex)=>({x:p.x.map(exact),d:p.d.map(exact)}),aa=a.map(pack),bb=b.map(pack),r=exact(thickness),rr=mul(r,r);
  return (axis:Vec3,lo:number,hi:number):boolean=>{
    if(axis.some(v=>!Number.isFinite(v))||!Number.isFinite(lo)||!Number.isFinite(hi)||lo>hi)return false;
    const n=axis.map(exact),nn=dot(n,n);if(nn.n===0n)return false;
    const times=[exact(lo),exact(hi)];
    const supports=(points:typeof aa)=>times.map(t=>points.map(p=>add(dot(n,p.x),mul(t,dot(n,p.d)))));
    const av=supports(aa),bv=supports(bb);let gap:D;
    if(sameTimeEndpoints){
      const ab=times.map((_,i)=>sub(min(bv[i]!),max(av[i]!))),ba=times.map((_,i)=>sub(min(av[i]!),max(bv[i]!)));
      gap=max([min(ab),min(ba)]);
    }else gap=max([sub(min(bv.flat()),max(av.flat())),sub(min(av.flat()),max(bv.flat()))]);
    return gap.n>0n && compare(mul(gap,gap),mul(rr,nn))>0n;
  };
}
