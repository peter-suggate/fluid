/** Opt-in native 2D swept-extension experiment, using the UI's actual seed. */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { createSparseCM12ComplexityScene, findSceneDefinition } from "../../lib/core/scenes";
import { sceneDocument } from "../../lib/core/scene-definition";
import { uniformLabSeed, UNIFORM_LAB_VALUES } from "../../lib/physics-wasm/uniform-controller";
const args = new Map(process.argv.slice(2).map((a):[string,string]=>{const [k,...v]=a.replace(/^--/,"").split("=");return [k!,v.join("=")];}));
const out=args.get("out")??"docs/research/uniform-geometric-swept-extension-2026-09-20";
mkdirSync(out,{recursive:true});
const scene=args.has("scene")?sceneDocument(findSceneDefinition(args.get("scene")!)!):createSparseCM12ComplexityScene("long-dam");
const seed=uniformLabSeed(scene);
const dt=Number(args.get("dt")??1/30), seconds=Number(args.get("seconds")??4);
const build=spawnSync("cargo",["build","--release","--manifest-path","rust/Cargo.toml","-p","fluid-core","--example","uniform_geometric_scene"],{stdio:"inherit"});assert.equal(build.status,0);
for(const mode of (args.get("mode")??"off,support,local,coarse,combined").split(",")) {
  const config={mode,areaGuard:args.has("guard"),localSweeps:Number(args.get("local")??4),coarseSweeps:Number(args.get("coarse")??24),cycles:Number(args.get("cycles")??1),traceSteps:Number(args.get("traces")??1),agreementGain:Number(args.get("gain")??0.7),agreementClamp:Number(args.get("clamp")??0.5),agreementIterations:Number(args.get("iterations")??2)};
  const request={...seed,options:UNIFORM_LAB_VALUES,dt,frames:Math.round(seconds/dt),auditStages:!args.has("timing"),sweptExtension:config};
  const id=args.get("id")??mode;
  console.log(`Running ${id}: ${request.frames} frames, ${seed.dimensions.join("x")}`);
  const run=spawnSync("rust/target/release/examples/uniform_geometric_scene",[],{input:JSON.stringify(request),encoding:"utf8",maxBuffer:128*1024*1024});assert.equal(run.status,0,run.stderr);
  const r=JSON.parse(run.stdout);
  writeFileSync(`${out}/${id}-input.json.gz`,gzipSync(JSON.stringify(request)));
  writeFileSync(`${out}/${id}.json.gz`,gzipSync(JSON.stringify(r)));
  const summary={id,config,scene:scene.sceneId,seconds,dt,initialVolume:r.initialVolume,final:r.finalHighResolutionMetrics??r.receipts.at(-1),
    elapsedMs:r.elapsedMs,msPerStep:r.elapsedMs/request.frames,
    maxPressureResidual:Math.max(...r.receipts.map((x:any)=>x.pressure.residual)),
    rejected:r.receipts.filter((x:any)=>!x.sweptExtension.accepted&&!["off","support","conforming-only"].includes(mode)).length,
    maxCorrection:Math.max(...r.receipts.map((x:any)=>x.sweptExtension.surface?.maxNormalShift ?? x.sweptExtension.maxCorrection)),
    activeMean:r.receipts.reduce((s:number,x:any)=>s+x.sweptExtension.activeCells,0)/request.frames,
    divergenceBefore:r.receipts.reduce((s:number,x:any)=>s+x.sweptExtension.divergenceBefore,0),
    divergenceAfter:r.receipts.reduce((s:number,x:any)=>s+x.sweptExtension.divergenceAfter,0),
  };
  writeFileSync(`${out}/${id}-summary.json`,JSON.stringify(summary,null,2)+"\n");console.log(JSON.stringify(summary));
}
