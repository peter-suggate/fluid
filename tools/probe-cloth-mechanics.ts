/** Reproducible CPU source-size mechanics timings; no fluid or scene substitutions. */
import {readFile} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import {performance} from 'node:perf_hooks';
import {decodeClothTopology} from '../lib/cloth/source-assets';
import {sourceClothDocument,wetClothSource} from '../lib/cloth/scenes/sources';
import {createClothState,prepareClothTopology} from '../lib/cloth/topology';
import {stepCloth} from '../lib/cloth/dynamics';
import {resolveClothContacts} from '../lib/cloth/contact';
const id=process.argv[2]??'wetcloth-source-simple-cloth',source=wetClothSource(id),document=sourceClothDocument(id);
const raw=await readFile(`public${source.asset.url}`),data=decodeClothTopology(gunzipSync(raw));
let start=performance.now(); const topology=prepareClothTopology(data,document.materials),state=createClothState(topology);
console.log(JSON.stringify({id,phase:'prepare',vertices:topology.vertexCount,triangles:topology.triangles.length,rods:topology.rods.length,ms:performance.now()-start}));
const before=state.positions_m.slice();start=performance.now();
const options={gravity_m_s2:source.gravity_m_s2,selfCollision:false,iterations:Number(process.env.CLOTH_ITERATIONS??64),tolerance_m:1e-8};
let report;
if(process.argv.includes('--gpu')) {
 const {GPUClothMechanics}=await import('../lib/cloth/gpu-mechanics');
 const {withUniformDevice}=await import('../tests/helpers/uniform-geometric');
 await withUniformDevice('source cloth mechanics benchmark',async device=>{const gpu=new GPUClothMechanics(device,topology);try{await gpu.initialize();start=performance.now();report=await gpu.step(state,source.dt_s,options);}finally{gpu.destroy();}});
} else report=stepCloth(topology,state,source.dt_s,options);
console.log(JSON.stringify({id,phase:'mechanics',ms:performance.now()-start,...report}));
if(!process.argv.includes('--skip-contact')) {start=performance.now();const contact=resolveClothContacts(topology,state,before,source.dt_s,[],true);console.log(JSON.stringify({id,phase:'contact',ms:performance.now()-start,...contact}));}
