/** The dynamic vector-write lowering (brick-lower.ts, a diagnostic switch, off by default): a corpus of every width and
 * operator, the forms left alone, the guard, the switch, and the dam's whole schedule lowered with nothing left (the
 * scanner's count is reported, with no upper limit). CPU only. */
import test from "node:test";
import assert from "node:assert/strict";
import { brickLowerChecked, brickLowerOn, brickModuleCode, dynAxisWGSL, lowerDynamicVectorWrites, scanDynamicVectorWrites } from "./brick-lower";

const lowerBody = (decl: string, stmt: string) => lowerDynamicVectorWrites(`fn f(i:u32){${decl}${stmt}}`, { helpers: false }).code.slice(`fn f(i:u32){${decl}`.length, -1);

test("every width and operator lowers to a select of the declared width", () => {
  const ops = ["+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"];
  for (const [n, t] of [[2, "vec2<u32>"], [3, "vec3<i32>"], [4, "vec4f"]] as const) {
    for (const op of ops) assert.equal(lowerBody(`var v:${t};`, `v[i]${op}=x+1u;`), `v=select(v,v${op}vec${n}(x+1u),nbDynAxis${n}(u32(i)));`, `${t} ${op}=`);
    assert.equal(lowerBody(`var v:${t};`, "v[i]=y;"), `v=select(v,vec${n}(y),nbDynAxis${n}(u32(i)));`);
    assert.equal(lowerBody(`var v:${t};`, "v[i]++;"), `v=select(v,v+vec${n}(1),nbDynAxis${n}(u32(i)));`);
    assert.equal(lowerBody(`var v:${t};`, "v[i]--;"), `v=select(v,v-vec${n}(1),nbDynAxis${n}(u32(i)));`);
    // The width from an initializer, a copy, and a parameter.
    assert.equal(lowerBody(`var v=vec${n}<f32>();`, "v[i]=0.;"), `v=select(v,vec${n}(0.),nbDynAxis${n}(u32(i)));`);
    assert.equal(lowerDynamicVectorWrites(`fn g(p:${t},k:u32)->${t}{var v=p;v[k]+=2;return v;}`, { helpers: false }).code, `fn g(p:${t},k:u32)->${t}{var v=p;v=select(v,v+vec${n}(2),nbDynAxis${n}(u32(k)));return v;}`);
  }
});

test("the nearest declaration in the function decides, and field paths take the field's type", () => {
  const code = "var<private> v:vec4f;\nfn a(i:u32){var v:vec2f;v[i]=1.;}\nfn b(i:u32){v[i]=1.;}\nstruct S{lo:vec3i,hi:vec3i}\nstruct T{lo:array<u32,3>}\nfn c(i:u32,s:ptr<function,S>){(*s).lo[i]-=1;}\nfn d(i:u32){var t=T(array<u32,3>());t.lo[i]=1u;var u=S(vec3i(),vec3i());{var u:vec2f;}u.lo[i]=2;}";
  const out = lowerDynamicVectorWrites(code).code;
  assert.ok(out.includes("v=select(v,vec2(1.),nbDynAxis2(u32(i)));}"));
  assert.ok(out.includes("fn b(i:u32){v=select(v,vec4(1.),nbDynAxis4(u32(i)));}"));
  assert.ok(out.includes("(*s).lo=select((*s).lo,(*s).lo-vec3(1),nbDynAxis3(u32(i)));"));
  assert.ok(out.includes("t.lo[i]=1u;"), "an array field is left alone");
  const calls = lowerDynamicVectorWrites("fn mk()->array<u32,6>{return array<u32,6>();}\nfn mv()->vec2u{return vec2u();}\nfn f(i:u32){var k=mk();k[i]=1u;var m=mv();m[i]=2u;}", { helpers: false }).code;
  assert.ok(calls.includes("k[i]=1u;") && calls.includes("m=select(m,vec2(2u),nbDynAxis2(u32(i)));"), "a call's return type decides");
  assert.ok(out.includes("{var u:vec2f;}u.lo=select(u.lo,vec3(2),nbDynAxis3(u32(i)));"), "a closed block's declaration is not visible");
  assert.equal(scanDynamicVectorWrites(out).length, 0);
});

test("arrays, atomics, storage buffers, literal indices and comparisons are left alone", () => {
  const code = [
    "@group(0) @binding(0) var<storage,read_write> buf:array<u32>;",
    "var<workgroup> wg:array<atomic<u32>,64>;",
    "fn f(i:u32,v:vec3f)->bool{",
    " var a:array<f32,4>; var b=array<u32,3>(0u,0u,0u); var w=v;",
    " buf[i]=1u; buf[i]+=2u; a[i]*=2.; b[i]|=1u; w[0]=1.; w[2u]+=1.;",
    " let c=w[i]<=1.; let d=w[i]==1.; let e=w[i]>=1.; let g=w[i]!=1.;",
    " return c&&d&&e&&g;}",
  ].join("\n");
  const r = lowerDynamicVectorWrites(code);
  assert.equal(r.sites, 0);
  assert.equal(r.code, code);
});

test("the right side runs to the statement's end, brackets and calls included, and a for-header's update lowers", () => {
  assert.equal(lowerBody("var v:vec3f;", "v[i]=h[(i+1u)%3u]*f(a,(b));"), "v=select(v,vec3(h[(i+1u)%3u]*f(a,(b))),nbDynAxis3(u32(i)));");
  assert.equal(lowerBody("var v:vec3u;", "for(var k=0u;k<3u;v[k]++){k++;}"), "for(var k=0u;k<3u;v=select(v,v+vec3(1),nbDynAxis3(u32(k)))){k++;}");
  assert.equal(lowerBody("var v:vec3u;", "v[i+j[k]]=1u;"), "v=select(v,vec3(1u),nbDynAxis3(u32(i+j[k])));");
});

test("the axis masks: one a width, appended once", () => {
  const r = lowerDynamicVectorWrites("fn f(i:u32){var a:vec2f;var b:vec3f;var c:vec3f;a[i]=1.;b[i]=1.;c[i]=1.;}");
  assert.deepEqual(r.widths, [2, 3]);
  assert.equal(r.code.match(/fn nbDynAxis3/g)?.length, 1);
  assert.ok(r.code.endsWith(dynAxisWGSL([2, 3]) + "\n"));
  assert.equal(dynAxisWGSL([4]), "fn nbDynAxis4(a:u32)->vec4<bool>{return vec4<bool>(a==0u,a==1u,a==2u,a==3u);}");
});

test("the guard: a dynamic write that survives the lowering fails, naming the kernel", () => {
  assert.doesNotThrow(() => brickLowerChecked("ok", "fn f(i:u32){var v:vec3f;v[i]=1.;}"));
  // A field declared with two types cannot be lowered: an explicit failure, never a guess.
  assert.throws(() => brickLowerChecked("twoTypes", "struct A{p:vec2f}\nstruct B{p:vec3f}\nfn f(i:u32){var a=mk();a.p[i]=1.;}"), /field p has several declared types/);
});

test("the switch: off, a module is created exactly as written; on, lowered and checked", () => {
  const code = "fn f(i:u32){var v:vec3f;v[i]=1.;}";
  assert.equal(brickModuleCode("k", code, false), code);
  assert.equal(brickModuleCode("k", code, true), brickLowerChecked("k", code));
  assert.notEqual(brickModuleCode("k", code, true), code);
  assert.equal(brickLowerOn(true), true);
  assert.equal(brickLowerOn(false), false);
  assert.equal(brickLowerOn(), process.env.NB_BRICK_LOWER === "1");
});

test("the dam's whole schedule lowers with no dynamic write left", async () => {
  const abi = await import("./brick-abi");
  const scenes = await import("../../core/scenes");
  const { brickSceneSetup } = await import("./brick-scene");
  const { BrickEngine } = await import("./brick-engine");
  const stagesFactory = (await import("./brick-fluid-stages")).brickFluidStages, solverFactory = (await import("./brick-pressure-solver")).brickPressureSolver;
  assert.ok(stagesFactory && solverFactory, "the brick stages and solver are registered");
  const { getMethod } = await import("../../core/method-registry");
  const { resolveMethodValues } = await import("../../core/method-contract");
  const { sceneDocument } = await import("../../core/scene-definition");
  await import("../index");
  const METHOD = "uniform-narrow-band-flip", method = getMethod(METHOD), definition = scenes.getSceneDefinition("high-resolution-dam-break")!;
  const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
  const { setup } = brickSceneSetup(sceneDocument(definition), values);
  assert.ok(setup);
  const L = new abi.BrickLayout({ lattice: setup.lattice, farPhi: setup.farPhi, ...setup.capacities });
  const stages = stagesFactory({ layout: L, dt_s: setup.dt_s, h_m: setup.h_m, gravity_mps2: setup.gravity_mps2, values, fluid: { density_kg_m3: setup.density_kg_m3, viscosity_Pa_s: setup.viscosity_Pa_s, surfaceTension_N_m: setup.surfaceTension_N_m }, particles: { bandDepth: setup.particleBandDepth } });
  const s = BrickEngine.scheduleFor(L, solverFactory(L), stages, setup.initial ? -1 : setup.liquid.length);
  const codes = new Set([...s.boot, ...s.part1, ...s.part2].map(p => p.kernel.code));
  let before = 0;
  for (const code of codes) { before += scanDynamicVectorWrites(code).length; assert.equal(scanDynamicVectorWrites(brickLowerChecked("dam", code)).length, 0); }
  assert.ok(before > 0, "the dam's kernels hold dynamic writes to lower");
  console.log(`dam: ${codes.size} modules, ${before} dynamic vector-component writes before lowering, 0 after`);
});
