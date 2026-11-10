/* Lane P: CPU Float64 prototype, residual reduction per launch of the pressure
 * cycle's STRUCTURE against launch-lean candidates. Not the solver: an
 * analogue (every liquid cell an h row, no band/native composite, no seams,
 * no active set, no BULK16). What it shares with the arena
 * (lib/methods/uniform/uniform-pressure-arena.ts): the six levels h..32h, sum
 * restriction, trilinear renormalised prolongation, lumped 7-point aggregated
 * operators (x0.5; x0.75 for the 16h and 32h aggregates) with Dirichlet row
 * sums kept, level 0 red-black sweeps Jacobi across two-tile workgroups, the
 * overlapping 8^3 box smoothers of levels 1 and 2 (halo 2, outer ring fixed,
 * core kept), three tail cycles over 8h/16h/32h, flexible GCR(1).
 *
 *   npx tsx tools/experiments/lane-p/pressure-launch-proto.ts [--case=shallow|deep|jet] [--json=path]
 */
import {writeFileSync} from "node:fs";

interface Level{
 n:number;dims:[number,number,number];map:Int32Array;cx:Int32Array;cy:Int32Array;cz:Int32Array;
 nb:Int32Array;w:Float64Array;diag:Float64Array;colour:Uint8Array;
 parent?:Int32Array;pIdx?:Int32Array;pW?:Float64Array;
 bricks:Map<string,Bricks>;
}
interface Bricks{count:number;red:Int32Array;black:Int32Array;core:Int32Array;redAt:Int32Array;blackAt:Int32Array;coreAt:Int32Array}
const DIRS=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]] as const;

interface Scene{name:string;dims:[number,number,number];phi:(x:number,y:number,z:number)=>number;solid:(i:number,j:number,k:number)=>0|1|2;
 /** u* on a non-solid face: axis, and the lower cell's coordinates. */
 face:(axis:number,i:number,j:number,k:number)=>number;bodyVelocity:[number,number,number]}

function hash(a:number,b:number,c:number,d:number,seed:number):number{
 let h=(a*374761393+b*668265263+c*2147483647+d*1274126177+seed*974711)|0;h=Math.imul(h^(h>>>13),1274126177);h^=h>>>16;
 return ((h>>>0)/4294967295)*2-1;
}

function buildFine(s:Scene):{level:Level;b:Float64Array}{
 const [nx,ny,nz]=s.dims,map=new Int32Array(nx*ny*nz).fill(-1);
 const at=(i:number,j:number,k:number)=>i+nx*(j+ny*k);
 const liquid=(i:number,j:number,k:number)=>i>=0&&j>=0&&k>=0&&i<nx&&j<ny&&k<nz&&s.solid(i,j,k)===0&&s.phi(i+0.5,j+0.5,k+0.5)<0;
 let n=0;const xs:number[]=[],ys:number[]=[],zs:number[]=[];
 for(let k=0;k<nz;k++)for(let j=0;j<ny;j++)for(let i=0;i<nx;i++)if(liquid(i,j,k)){map[at(i,j,k)]=n++;xs.push(i);ys.push(j);zs.push(k);}
 const L:Level={n,dims:s.dims,map,cx:Int32Array.from(xs),cy:Int32Array.from(ys),cz:Int32Array.from(zs),nb:new Int32Array(6*n).fill(-1),w:new Float64Array(6*n),diag:new Float64Array(n),colour:new Uint8Array(n),bricks:new Map()};
 const b=new Float64Array(n);
 for(let c=0;c<n;c++){
  const i=xs[c]!,j=ys[c]!,k=zs[c]!;L.colour[c]=(i+j+k)&1;const pc=s.phi(i+0.5,j+0.5,k+0.5);
  for(let d=0;d<6;d++){
   const [dx,dy,dz]=DIRS[d]!,a=i+dx,e=j+dy,f=k+dz,axis=d>>1,sign=d&1?-1:1;
   const inside=a>=0&&e>=0&&f>=0&&a<nx&&e<ny&&f<nz;
   const kind=inside?s.solid(a,e,f):(d===2?0:1);// past the box: a wall, open above
   if(kind!==0){b[c]-=sign*(kind===2?s.bodyVelocity[axis]!:0);continue;}
   // A non-solid face: u* there.
   const u=d&1?s.face(axis,a,e,f):s.face(axis,i,j,k);b[c]-=sign*u;
   const q=inside?map[at(a,e,f)]!:-1;
   if(q>=0){L.nb[6*c+d]=q;L.w[6*c+d]=1;L.diag[c]+=1;}
   else{const pn=inside?s.phi(a+0.5,e+0.5,f+0.5):1;const theta=Math.min(1,Math.max(0.1,pc/(pc-Math.max(pn,1e-9))));L.diag[c]+=1/theta;}
  }
 }
 return {level:L,b};
}

function coarsen(F:Level,scale:number):Level{
 const [fx,fy,fz]=F.dims,dims:[number,number,number]=[(fx+1)>>1,(fy+1)>>1,(fz+1)>>1],[nx,ny]=dims;
 const map=new Int32Array(dims[0]*dims[1]*dims[2]).fill(-1),at=(i:number,j:number,k:number)=>i+nx*(j+ny*k);
 const parent=new Int32Array(F.n);let n=0;const xs:number[]=[],ys:number[]=[],zs:number[]=[];
 for(let c=0;c<F.n;c++){const i=F.cx[c]!>>1,j=F.cy[c]!>>1,k=F.cz[c]!>>1,m=at(i,j,k);if(map[m]!<0){map[m]=n++;xs.push(i);ys.push(j);zs.push(k);}parent[c]=map[m]!;}
 const C:Level={n,dims,map,cx:Int32Array.from(xs),cy:Int32Array.from(ys),cz:Int32Array.from(zs),nb:new Int32Array(6*n).fill(-1),w:new Float64Array(6*n),diag:new Float64Array(n),colour:new Uint8Array(n),bricks:new Map()};
 for(let c=0;c<n;c++)C.colour[c]=(xs[c]!+ys[c]!+zs[c]!)&1;
 for(let c=0;c<F.n;c++){
  const p=parent[c]!;let links=0;
  for(let d=0;d<6;d++){const q=F.nb[6*c+d]!;if(q<0)continue;links+=F.w[6*c+d]!;const pq=parent[q]!;if(pq!==p){C.nb[6*p+d]=pq;C.w[6*p+d]+=scale*F.w[6*c+d]!;}}
  C.diag[p]+=F.diag[c]!-links;// the Dirichlet part, kept whole
 }
 for(let p=0;p<n;p++)for(let d=0;d<6;d++)C.diag[p]+=C.w[6*p+d]!;
 // Trilinear over the parents that exist, renormalised.
 const pIdx=new Int32Array(8*F.n).fill(-1),pW=new Float64Array(8*F.n);
 for(let c=0;c<F.n;c++){
  const i=F.cx[c]!,j=F.cy[c]!,k=F.cz[c]!,I=i>>1,J=j>>1,K=k>>1,oi=i&1?1:-1,oj=j&1?1:-1,ok=k&1?1:-1;let sum=0;
  for(let m=0;m<8;m++){
   const a=I+(m&1?oi:0),e=J+(m&2?oj:0),f=K+(m&4?ok:0);
   if(a<0||e<0||f<0||a>=dims[0]||e>=dims[1]||f>=dims[2])continue;const q=map[at(a,e,f)]!;if(q<0)continue;
   const wt=(m&1?0.25:0.75)*(m&2?0.25:0.75)*(m&4?0.25:0.75);pIdx[8*c+m]=q;pW[8*c+m]=wt;sum+=wt;
  }
  for(let m=0;m<8;m++)pW[8*c+m]/=sum;
 }
 F.parent=parent;F.pIdx=pIdx;F.pW=pW;
 return C;
}

function bricksOf(L:Level,bx:number,by:number,bz:number,ring:number):Bricks{
 const key=`${bx},${by},${bz},${ring}`;let B=L.bricks.get(key);if(B)return B;
 const [nx,ny,nz]=L.dims,gx=Math.ceil(nx/bx),gy=Math.ceil(ny/by),gz=Math.ceil(nz/bz);
 const red:number[]=[],black:number[]=[],core:number[]=[],redAt=[0],blackAt=[0],coreAt=[0];
 for(let K=0;K<gz;K++)for(let J=0;J<gy;J++)for(let I=0;I<gx;I++){
  let any=false;
  for(let k=K*bz;k<Math.min(nz,(K+1)*bz)&&!any;k++)for(let j=J*by;j<Math.min(ny,(J+1)*by)&&!any;j++)for(let i=I*bx;i<Math.min(nx,(I+1)*bx);i++)if(L.map[i+nx*(j+ny*k)]!>=0){any=true;break;}
  if(!any)continue;
  for(let k=Math.max(0,K*bz-ring);k<Math.min(nz,(K+1)*bz+ring);k++)for(let j=Math.max(0,J*by-ring);j<Math.min(ny,(J+1)*by+ring);j++)for(let i=Math.max(0,I*bx-ring);i<Math.min(nx,(I+1)*bx+ring);i++){
   const c=L.map[i+nx*(j+ny*k)]!;if(c<0)continue;((i+j+k)&1?black:red).push(c);
   if(i>=I*bx&&i<(I+1)*bx&&j>=J*by&&j<(J+1)*by&&k>=K*bz&&k<(K+1)*bz)core.push(c);
  }
  redAt.push(red.length);blackAt.push(black.length);coreAt.push(core.length);
 }
 B={count:redAt.length-1,red:Int32Array.from(red),black:Int32Array.from(black),core:Int32Array.from(core),redAt:Int32Array.from(redAt),blackAt:Int32Array.from(blackAt),coreAt:Int32Array.from(coreAt)};
 L.bricks.set(key,B);return B;
}

const relax=(L:Level,x:Float64Array,b:Float64Array,c:number):void=>{
 const o=6*c,nb=L.nb,w=L.w;let s=b[c]!;
 for(let d=0;d<6;d++){const q=nb[o+d]!;if(q>=0)s+=w[o+d]!*x[q]!;}
 if(L.diag[c]!>0)x[c]=s/L.diag[c]!;
};
/** One launch: every brick sweeps its active cells (core plus ring) k times
 * red-black from the launch's input iterate and keeps its core (Jacobi
 * across bricks, as workgroups of one dispatch are). */
function blockSmooth(L:Level,x:Float64Array,b:Float64Array,B:Bricks,k:number):void{
 const loc=Float64Array.from(x),out=Float64Array.from(x);
 for(let g=0;g<B.count;g++){
  for(let s=0;s<k;s++){
   for(let m=B.redAt[g]!;m<B.redAt[g+1]!;m++)relax(L,loc,b,B.red[m]!);
   for(let m=B.blackAt[g]!;m<B.blackAt[g+1]!;m++)relax(L,loc,b,B.black[m]!);
  }
  for(let m=B.coreAt[g]!;m<B.coreAt[g+1]!;m++){const c=B.core[m]!;out[c]=loc[c]!;}
  for(let m=B.redAt[g]!;m<B.redAt[g+1]!;m++){const c=B.red[m]!;loc[c]=x[c]!;}
  for(let m=B.blackAt[g]!;m<B.blackAt[g+1]!;m++){const c=B.black[m]!;loc[c]=x[c]!;}
 }
 x.set(out);
}
function globalRB(L:Level,x:Float64Array,b:Float64Array,sweeps:number):void{
 for(let s=0;s<sweeps;s++)for(let colour=0;colour<2;colour++)for(let c=0;c<L.n;c++)if(L.colour[c]===colour)relax(L,x,b,c);
}
function apply(L:Level,x:Float64Array,out:Float64Array):void{
 for(let c=0;c<L.n;c++){let s=L.diag[c]!*x[c]!;const o=6*c;for(let d=0;d<6;d++){const q=L.nb[o+d]!;if(q>=0)s-=L.w[o+d]!*x[q]!;}out[c]=s;}
}
function residual(L:Level,x:Float64Array,b:Float64Array):Float64Array{const r=new Float64Array(L.n);apply(L,x,r);for(let c=0;c<L.n;c++)r[c]=b[c]!-r[c]!;return r;}
function restrict(F:Level,r:Float64Array,n:number):Float64Array{const out=new Float64Array(n);for(let c=0;c<F.n;c++)out[F.parent![c]!]+=r[c]!;return out;}
function prolongAdd(F:Level,x:Float64Array,coarse:Float64Array):void{
 for(let c=0;c<F.n;c++){let s=0;for(let m=0;m<8;m++){const q=F.pIdx![8*c+m]!;if(q>=0)s+=F.pW![8*c+m]!*coarse[q]!;}x[c]+=s;}
}
const maxAbs=(v:Float64Array)=>{let m=0;for(const e of v)m=Math.max(m,Math.abs(e));return m;};
const dot=(a:Float64Array,b:Float64Array)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]!*b[i]!;return s;};

export interface Scheme{
 name:string;
 /** Level 0: "pair" = a red-black sweep a launch, Jacobi across two-tile workgroups (today);
  * "box" = one launch of k sweeps over an 8^3 brick's overlapping box; "global" = exact red-black sweeps. */
 l0:{kind:"pair";sweeps:number}|{kind:"global";sweeps:number}|{kind:"box";k:number;launches:number};
 k1:number;k2:number;tailCycles:number;tailSweeps:number;krylov:boolean;
 /** Launches a cycle on the real composite (see launchesPerCycle). */
 fused:{tail:boolean;krylov:boolean;native:boolean;restrict:boolean};
}
/** Launches a V-cycle on the arena's composite (paddleboats: natives, BULK16
 * lift, Krylov), by the same items as uniform-pressure-arena-launches.ts. */
export function launchesPerCycle(s:Scheme):number{
 const l0=s.l0.kind==="box"?s.l0.launches:s.l0.sweeps;
 const pre=l0,post=s.l0.kind==="box"?l0-1:l0;// a box post-smoothing takes the prolongation into its first launch
 const native=s.fused.native?2:4,residual=1,restrictRebase=s.fused.restrict?1:2,nativeOperator=1,level1=2+1,level2=2+1+1;
 const tail=s.fused.tail?3*s.tailCycles:5*s.tailCycles+1,prolong=1,krylov=s.krylov?(s.fused.krylov?3:5):0;
 return pre+post+native+residual+restrictRebase+nativeOperator+level1+level2+tail+prolong+krylov;
}

function cycle(H:Level[],x:Float64Array,b:Float64Array,s:Scheme):void{
 const L0=H[0]!,L1=H[1]!,L2=H[2]!;
 const l0=s.l0;
 const smooth0=()=>{
  if(l0.kind==="pair"){const B=bricksOf(L0,4,4,8,0);for(let k=0;k<l0.sweeps;k++)blockSmooth(L0,x,b,B,1);}
  else if(l0.kind==="global")globalRB(L0,x,b,l0.sweeps);
  else{const B=bricksOf(L0,8,8,8,1);for(let k=0;k<l0.launches;k++)blockSmooth(L0,x,b,B,l0.k);}
 };
 smooth0();
 const b1=restrict(L0,residual(L0,x,b),L1.n),x1=new Float64Array(L1.n),B1=bricksOf(L1,8,8,8,1);
 blockSmooth(L1,x1,b1,B1,s.k1);
 const b2=restrict(L1,residual(L1,x1,b1),L2.n),x2=new Float64Array(L2.n),B2=bricksOf(L2,8,8,8,1);
 blockSmooth(L2,x2,b2,B2,s.k2);
 const b3=restrict(L2,residual(L2,x2,b2),H[3]!.n),x3=new Float64Array(H[3]!.n);
 // The page tail: 8h and 16h smooth a page at a time (a page is 4^3 8h rows, 2^3 16h rows), the 32h rows in one group.
 const tail=(l:number,xl:Float64Array,bl:Float64Array):void=>{
  const L=H[l]!;
  if(l===H.length-1){globalRB(L,xl,bl,32);return;}
  const side=32>>(l),B=bricksOf(L,side,side,side,0);
  blockSmooth(L,xl,bl,B,s.tailSweeps);
  const bc=restrict(L,residual(L,xl,bl),H[l+1]!.n),xc=new Float64Array(H[l+1]!.n);
  tail(l+1,xc,bc);prolongAdd(L,xl,xc);blockSmooth(L,xl,bl,B,s.tailSweeps);
 };
 for(let m=0;m<s.tailCycles;m++)tail(3,x3,b3);
 prolongAdd(L2,x2,x3);blockSmooth(L2,x2,b2,B2,s.k2);
 prolongAdd(L1,x1,x2);blockSmooth(L1,x1,b1,B1,s.k1);
 prolongAdd(L0,x,x1);smooth0();
}

/** The solve: cycles of the scheme under flexible GCR(1) (the first cycle plain). Returns max |r| after each cycle. */
export function solve(H:Level[],b:Float64Array,start:Float64Array,s:Scheme,cycles:number):{history:number[];l2:number[];x:Float64Array}{
 const L=H[0]!,x=Float64Array.from(start);let r=residual(L,x,b);const history=[maxAbs(r)],l2=[Math.sqrt(dot(r,r))];
 let pPrev:Float64Array|undefined,qPrev:Float64Array|undefined;
 for(let c=0;c<cycles;c++){
  const y=Float64Array.from(x);cycle(H,y,b,s);
  if(!s.krylov){x.set(y);r=residual(L,x,b);}
  else{
   const z=new Float64Array(L.n);for(let i=0;i<L.n;i++)z[i]=y[i]!-x[i]!;const q=new Float64Array(L.n);apply(L,z,q);
   if(c===0){x.set(y);r=residual(L,x,b);}
   else{
    if(pPrev&&qPrev){const beta=dot(q,qPrev)/Math.max(dot(qPrev,qPrev),1e-300);for(let i=0;i<L.n;i++){z[i]-=beta*pPrev[i]!;q[i]-=beta*qPrev[i]!;}}
    const alpha=dot(q,r)/Math.max(dot(q,q),1e-300);for(let i=0;i<L.n;i++){x[i]+=alpha*z[i]!;r[i]-=alpha*q[i]!;}
   }
   pPrev=z;qPrev=q;
  }
  history.push(maxAbs(r));l2.push(Math.sqrt(dot(r,r)));
 }
 return {history,l2,x};
}

function hierarchy(fine:Level):Level[]{
 const H=[fine];for(let l=1;l<6;l++)H.push(coarsen(H[l-1]!,l>=4?0.75:0.5));return H;
}

// ---- scenes (h = 1; G = g dt / h in cells a step: 9.81/60/60/0.025 = 0.109 at h = 2.5 cm) ----
const G=0.109;
interface SceneSpec{name:string;dims:[number,number,number];depth:number;box:[number,number,number,number,number,number];splash:[number,number,number];jet?:{x:number;z:number;radius:number;speed:number}}
function scene(spec:SceneSpec,t:number):Scene{
 // t = 0: last step's state; t = 1: this step's. The wave travels, the splash lump moves and grows, the box accelerates.
 const [sx,sy,sz]=spec.splash,cx=sx+0.6*t,cy=sy+0.5*t,amp=(0.7+0.3*t)*5*G;
 const surface=(x:number,z:number)=>spec.depth+1.5*Math.sin(2*Math.PI*x/40+0.15*t)+Math.cos(2*Math.PI*z/30-0.1*t);
 const phi=(x:number,y:number,z:number)=>{
  let p=y-surface(x,z);p=Math.min(p,Math.hypot(x-cx,y-cy,z-sz)-4);
  if(spec.jet)p=Math.min(p,Math.hypot(x-spec.jet.x,z-spec.jet.z)-spec.jet.radius);
  return p;
 };
 const [b0,b1,b2,b3,b4,b5]=spec.box;
 return {name:spec.name,dims:spec.dims,phi,
  solid:(i,j,k)=>i>=b0&&i<b1&&j>=b2&&j<b3&&k>=b4&&k<b5?2:0,
  bodyVelocity:[3*G*(0.9+0.1*t),-2*G*(0.9+0.1*t),0],
  face:(axis,i,j,k)=>{
   let u=axis===1?-G:0;
   const d=Math.hypot(i+0.5-cx,j+0.5-cy,k+0.5-sz);if(d<6)u+=amp*hash(i,j,k,axis,7)*(1-d/6);
   if(spec.jet&&axis===1&&j>=spec.depth+2&&Math.hypot(i+0.5-spec.jet.x,k+0.5-spec.jet.z)<spec.jet.radius+1)u-=spec.jet.speed;
   return u;
  }};
}
const CASES:Record<string,SceneSpec>={
 // A quarter-length paddleboat pool: 24 cells deep, a hull 4 cells into the water, a splash lump at the surface.
 shallow:{name:"shallow tank 128x40x48, depth 24",dims:[128,40,48],depth:24,box:[56,72,20,28,18,30],splash:[30,26,12]},
 deep:{name:"deep tank 48x72x48, depth 56",dims:[48,72,48],depth:56,box:[16,28,52,60,18,30],splash:[36,58,12]},
 jet:{name:"thin jet 40x96x40: radius 2.5 column into a 12-deep pool",dims:[40,96,40],depth:12,box:[6,14,9,13,6,14],splash:[28,13,28],jet:{x:20,z:20,radius:2.5,speed:12*G}},
};

const TODAY:Scheme={name:"today: pair x3, box k3/k6, tail x3, GCR",l0:{kind:"pair",sweeps:3},k1:3,k2:6,tailCycles:3,tailSweeps:3,krylov:true,fused:{tail:false,krylov:false,native:false,restrict:false}};
const lean=(name:string,k:number,launches:number,tailCycles:number,more:Partial<Scheme>={}):Scheme=>({name,l0:{kind:"box",k,launches},k1:3,k2:6,tailCycles,tailSweeps:3,krylov:true,fused:{tail:true,krylov:true,native:true,restrict:true},...more});
const SCHEMES:Scheme[]=[
 TODAY,
 {...TODAY,name:"today, tail x1 (unfused)",tailCycles:1},
 {...TODAY,name:"today, launches fused only (tail x3)",fused:{tail:true,krylov:true,native:true,restrict:true}},
 {...TODAY,name:"today, launches fused, tail x1",tailCycles:1,fused:{tail:true,krylov:true,native:true,restrict:true}},
 lean("lean: L0 box k3 x1, tail x3",3,1,3),
 lean("lean: L0 box k3 x1, tail x1",3,1,1),
 lean("lean: L0 box k6 x1, tail x1",6,1,1),
 lean("lean: L0 box k3 x2, tail x1",3,2,1),
 lean("lean: L0 box k6 x1, k1 6, k2 12, tail x1, tail sweeps 6",6,1,1,{k1:6,k2:12,tailSweeps:6}),
 lean("lean: L0 box k6 x1, tail x3",6,1,3),
 lean("lean: L0 box k6 x1, k1 6, k2 12, tail x2, tail sweeps 6",6,1,2,{k1:6,k2:12,tailSweeps:6}),
];

/** Cycles (log-interpolated) for max |r| to fall by `factor` from the start; Infinity if not within the history. */
function cyclesTo(history:number[],factor:number):number{
 const target=history[0]!/factor;
 for(let c=1;c<history.length;c++)if(history[c]!<=target){const a=Math.log(history[c-1]!),e=Math.log(history[c]!);return c-1+(a-Math.log(target))/(a-e);}
 return Number.POSITIVE_INFINITY;
}

function run(name:string){
 const spec=CASES[name]!,previous=buildFine(scene(spec,0)),now=buildFine(scene(spec,1));
 const Hp=hierarchy(previous.level),H=hierarchy(now.level);
 // Last step's pressure, solved to round-off, carried to this step's cells (0 where a cell is new).
 const before=solve(Hp,previous.b,new Float64Array(previous.level.n),TODAY,30);
 const warm=new Float64Array(now.level.n);
 for(let c=0;c<now.level.n;c++){const q=previous.level.map[now.level.cx[c]!+spec.dims[0]*(now.level.cy[c]!+spec.dims[1]*now.level.cz[c]!)]!;if(q>=0)warm[c]=before.x[q]!;}
 const zero=new Float64Array(now.level.n);
 console.log(`\n## ${spec.name}: ${now.level.n} liquid cells, levels ${H.map(l=>l.n).join(" / ")}; last step's solve reached ${(before.history.at(-1)!/before.history[0]!).toExponential(1)}`);
 const rows:object[]=[];
 console.log("| scheme | launches/cycle | warm x/cycle, cycles 1-3 | 3-6 | warm: cycles to 1/400 | launches | vs today | warm: cycles to 1e-5 | launches | vs today | cold: cycles to 1/400 | launches | vs today |");
 console.log("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
 const base:number[]=[];
 for(const s of SCHEMES){
  const per=launchesPerCycle(s),w=solve(H,now.b,warm,s,12),c=solve(H,now.b,zero,s,12);
  const early=Math.pow(w.history[0]!/w.history[3]!,1/3),late=Math.pow(w.history[3]!/w.history[6]!,1/3);
  const counts=[cyclesTo(w.history,400),cyclesTo(w.history,1e5),cyclesTo(c.history,400)];
  if(s===TODAY)for(const n of counts)base.push(n*per);
  rows.push({scene:name,scheme:s.name,launchesPerCycle:per,warmHistory:w.history,coldHistory:c.history,warmL2:w.l2,cycles:counts});
  console.log(`| ${s.name} | ${per} | ${early.toFixed(1)} | ${late.toFixed(1)} | ${counts.map((n,i)=>`${n.toFixed(2)} | ${(n*per).toFixed(0)} | ${(base[i]!/(n*per)).toFixed(2)}x`).join(" | ")} |`);
 }
 const w=solve(H,now.b,warm,TODAY,1),c=solve(H,now.b,zero,TODAY,1);
 console.log(`start max |r|: warm ${w.history[0]!.toExponential(2)}, cold ${c.history[0]!.toExponential(2)} (warm start is ${(c.history[0]!/w.history[0]!).toFixed(1)}x lower)`);
 return rows;
}

const only=process.argv.find(a=>a.startsWith("--case="))?.slice(7),json=process.argv.find(a=>a.startsWith("--json="))?.slice(7);
const all:object[]=[];
for(const name of Object.keys(CASES))if(!only||only===name)all.push(...run(name));
if(json)writeFileSync(json,JSON.stringify(all,null,1));
