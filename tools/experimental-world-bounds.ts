/** Rejected runtime world-policy boundary specialization. Kept tool-local
 * for reproducible parity/performance probes; production uses its measured
 * faster generic helpers. */
export function experimentalWorldBounds(code:string,buffer:string,table:string):string {
 const bounds=`// Policies 1..3 derive every SOLID page from immutable world planes;
 // allocation and retirement only exchange AIR and resident pages. Policy 0
 // is an explicit directory and retains the general lookup below. A point
 // on or inside a SOLID page also keeps the exact general-side selection.
 if(umWindowPolicy()!=0u){
  let world=page+umFramePage();let worldLow=umWindowSolidLow();let worldHigh=umWindowSolidHigh();
  if(all(world>=worldLow)&&all(world<worldHigh)){
   b.lo=select(b.lo,low,world==worldLow);
   b.hi=select(b.hi,high,world==worldHigh-vec3i(1));return b;
  }
 }`;
 const incident=`// An incident non-SOLID cell under a world policy is the preferred
 // cell with just its blocked face axes flipped. AIR needs no owner lookup.
 // Off the world's boundary, or when no incident cell can reach its open
 // span, return the preferred cell exactly as the directory walk does.
 if(!any(on)){return c;}
 if(umWindowPolicy()!=0u){
  let world=(c>>vec3u(5u))+umFramePage();let lo=umWindowSolidLow();let hi=umWindowSolidHigh();
  let low=world<lo;let high=world>=hi;
  if(below){
   let flip=low&on&(world==lo-vec3i(1));
   if(!any(high)&&all(flip==low)){return c+vec3i(flip);}
  }else{
   let flip=high&on&(world==hi);
   if(!any(low)&&all(flip==high)){return c-vec3i(flip);}
  }
  return c;
 }`;
 const a=' let high=low+vec3f(32.0);\n let solidBelow=vec3<bool>';
 const b=' let on=(p&vec3i(31))==vec3i(0);let c=p-select(vec3i(0),vec3i(1),below);';
 if(!code.includes(a)||!code.includes(b))throw new Error('World-bound experiment requires the generic point bounds and incident-cell helpers');
 code=code.replace(a,' let high=low+vec3f(32.0);\n '+bounds+'\n let solidBelow=vec3<bool>');
 code=code.replace(b,b+'\n '+incident);
 const word=(i:number)=>`${buffer}[${table}+${64**3+i}u]`;
 const vector=(i:number)=>`bitcast<vec3i>(vec3u(${word(i)},${word(i+1)},${word(i+2)}))`;
 return code+`\nfn umWindowPolicy()->u32{return ${word(12)};}
fn umWindowSolidLow()->vec3i{return ${vector(24)};}
fn umWindowSolidHigh()->vec3i{return ${vector(27)};}`;
}
