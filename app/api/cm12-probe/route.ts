import {mkdir,writeFile} from "node:fs/promises";
export async function POST(request:Request){
 const step=Number(new URL(request.url).searchParams.get("step"));
 if(!Number.isInteger(step)||step<1||step>180)return new Response("bad step",{status:400});
 const data=await request.arrayBuffer();if(data.byteLength!==32768*9)return new Response("bad size",{status:400});
 await mkdir("/tmp/cm12-browser-holes",{recursive:true});
 await writeFile(`/tmp/cm12-browser-holes/step-${step}.bin`,new Uint8Array(data));return new Response("ok");
}
