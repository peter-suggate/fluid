import assert from "node:assert/strict";
import test from "node:test";
import {compileUniformStencil,uniformVertexIncidentMasks,UNIFORM_COMPILED_TOPOLOGY} from "../lib/methods/uniform/uniform-compiled-topology";

test("compiled coarse incidence matches the vertex walk for every absent/fine/coarse incident combination",()=>{
 const bit=(x:number,y:number,z:number)=>1<<(x+1+3*(y+1+3*(z+1)));
 // 0/1/4 tile widths; missing incident pages and mixed low/high walls are
 // represented by absent neighbors. Local 4 is a vertex on a high wall.
 const locals:number[][]=[];
 for(const x of [0,1,4])for(const y of [0,1,4])for(const z of [0,1,4])if(x===1||y===1||z===1)locals.push([x,y,z]);
 for(let configuration=0;configuration<3**8;configuration++){
  let state=configuration,fine=0,valid=0;
  for(let k=0;k<8;k++){
   const width=state%3;state=Math.floor(state/3);const mask=bit((k&1)-1,((k>>1)&1)-1,(k>>2)-1);
   if(width!==0)valid|=mask;if(width===1)fine|=mask;
  }
  const incident=(compileUniformStencil(fine,valid)>>>UNIFORM_COMPILED_TOPOLOGY.incident)&255;
  for(const local of locals){
   let walked=false;
   for(let k=0;k<8;k++){
    const back=[k&1,(k>>1)&1,k>>2];
    if(back.some((b,a)=>b!==0&&local[a]!%4!==0))continue;
    const at=local.map((v,a)=>Math.floor(v/4)-back[a]!);
    const mask=bit(at[0]!,at[1]!,at[2]!);
    if((valid&mask)!==0&&(fine&mask)===0){walked=true;break;}
   }
   const required=Number(local[0]!==0)|(Number(local[1]!==0)<<1)|(Number(local[2]!==0)<<2);
   const compiled=(incident&uniformVertexIncidentMasks[required]!)!==0;
   assert.equal(compiled,walked,`configuration ${configuration}, vertex ${local}`);
  }
 }
});
