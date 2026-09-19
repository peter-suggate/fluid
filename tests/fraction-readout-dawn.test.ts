/**
 * The shader's V/K readout against the lab's.
 *
 * The 3-D slice writes V/K with a WGSL formatter and a bitmap font, because a
 * fragment shader has no `fillText`. Both are second implementations of
 * something `lib/core` already defines — the string is `fractionReadout`, the
 * letterforms are `FRACTION_READOUT_FONT` — so this runs the shader code on
 * Dawn and compares it with those definitions rather than with a transcription
 * of them.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  FRACTION_FLOOR, FRACTION_OVERFULL, FRACTION_READOUT_ALPHABET, FRACTION_READOUT_CEILING,
  FRACTION_READOUT_HUNDREDTH, FRACTION_READOUT_WHOLE, fractionReadout, fractionViewShaderConstants,
} from "../lib/core/fluid-fraction-view";
import {
  FRACTION_READOUT_FONT, FRACTION_READOUT_GLYPH_HEIGHT, fractionReadoutShaderLibrary,
} from "../lib/core/fraction-readout.wgsl";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";

const RASTER_WIDTH = 120, RASTER_HEIGHT = 32;

const probeShader = `${fractionViewShaderConstants}
${fractionReadoutShaderLibrary}
@group(0) @binding(0) var<storage,read> fills:array<f32>;
@group(0) @binding(1) var<storage,read_write> glyphs:array<vec2u>;
@compute @workgroup_size(64) fn formatFills(@builtin(global_invocation_id) id:vec3u){
  if(id.x<arrayLength(&fills)){glyphs[id.x]=fractionReadoutGlyphs(fills[id.x]);}
}
struct RasterCase { text:vec2u, scale:i32, pad:u32, centre:vec2f, pad2:vec2f }
@group(0) @binding(2) var<storage,read> rasterCase:RasterCase;
@group(0) @binding(3) var<storage,read_write> raster:array<vec2f>;
@compute @workgroup_size(64) fn rasterText(@builtin(global_invocation_id) id:vec3u){
  if(id.x>=${RASTER_WIDTH * RASTER_HEIGHT}u){return;}
  let pixel=vec2f(f32(id.x%${RASTER_WIDTH}u)+0.5,f32(id.x/${RASTER_WIDTH}u)+0.5);
  raster[id.x]=fractionReadoutInk(pixel,rasterCase.centre,rasterCase.text,rasterCase.scale);
}`;

function decode(packed: number, length: number): string {
  let text = "";
  for (let slot = 0; slot < length; slot += 1) {
    text += FRACTION_READOUT_ALPHABET[(packed >>> (4 * slot)) & 15] ?? "?";
  }
  return text;
}

function encode(text: string): [number, number] {
  let packed = 0;
  [...text].forEach((character, slot) => {
    packed = (packed | (FRACTION_READOUT_ALPHABET.indexOf(character) << (4 * slot))) >>> 0;
  });
  return [packed, text.length];
}

/**
 * Where f32 arithmetic and the lab's f64 may honestly round apart: a
 * hundredths tie, a half-decade, or a band threshold hit to within a few ulps.
 * A readout that differs there moves its last digit by one and nothing else.
 */
function nearAmbiguity(fill: number): boolean {
  const near = (value: number, target: number) => Math.abs(value - target) <= 1e-5 * target;
  if ([FRACTION_FLOOR, FRACTION_OVERFULL, FRACTION_READOUT_WHOLE, FRACTION_READOUT_HUNDREDTH,
    FRACTION_READOUT_CEILING].some(threshold => near(fill, threshold))) return true;
  if (fill >= FRACTION_READOUT_CEILING) return false;
  if (fill >= FRACTION_READOUT_HUNDREDTH) {
    const hundredths = fill * 100;
    return Math.abs(hundredths - Math.floor(hundredths) - 0.5) < 1e-6 + 3e-7 * hundredths;
  }
  const decade = Math.log10(fill);
  return Math.abs(decade - Math.floor(decade) - 0.5) < 1e-4;
}

/** The ink and casing the font says a readout draws, at whole-pixel scale. */
function expectedRaster(text: string, scale: number, centre: [number, number]): Float32Array {
  const width = (text.length * 6 - 1) * scale, height = FRACTION_READOUT_GLYPH_HEIGHT * scale;
  const left = Math.round(centre[0] - width / 2), top = Math.round(centre[1] - height / 2);
  const ink = (x: number, y: number): boolean => {
    const lx = x - left, ly = y - top;
    if (lx < 0 || ly < 0) return false;
    const fx = Math.floor(lx / scale), fy = Math.floor(ly / scale);
    const slot = Math.floor(fx / 6), column = fx % 6;
    if (slot >= text.length || column >= 5 || fy >= FRACTION_READOUT_GLYPH_HEIGHT) return false;
    return FRACTION_READOUT_FONT[text[slot]!]![fy]![column] === "#";
  };
  const result = new Float32Array(RASTER_WIDTH * RASTER_HEIGHT * 2);
  for (let y = 0; y < RASTER_HEIGHT; y += 1) for (let x = 0; x < RASTER_WIDTH; x += 1) {
    const at = 2 * (x + RASTER_WIDTH * y);
    if (ink(x, y)) { result[at] = 1; continue; }
    let cased = false;
    for (let dy = -1; dy <= 1 && !cased; dy += 1) for (let dx = -1; dx <= 1; dx += 1) {
      if (ink(x + dx, y + dy)) { cased = true; break; }
    }
    result[at + 1] = cased ? 1 : 0;
  }
  return result;
}

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("the shader writes V/K as fractionReadout does, in FRACTION_READOUT_FONT", { timeout: 60_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "fraction-readout");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal", "enable-dawn-features=disable_blob_cache"]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    const shaderModule = device.createShaderModule({ code: probeShader });
    const compilation = await shaderModule.getCompilationInfo();
    assert.deepEqual(compilation.messages.filter(m => m.type === "error").map(m => `${m.lineNum}: ${m.message}`), []);

    const readBack = async (buffer: GPUBuffer, bytes: number): Promise<ArrayBuffer> => {
      const staging = device!.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = device!.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
      device!.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const copy = staging.getMappedRange().slice(0);
      staging.unmap(); staging.destroy();
      return copy;
    };
    const storage = (data: ArrayBufferView<ArrayBuffer> | number): GPUBuffer => {
      const size = typeof data === "number" ? data : data.byteLength;
      const buffer = device!.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      if (typeof data !== "number") device!.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    const run = async (entryPoint: string, entries: GPUBindGroupEntry[], invocations: number) => {
      const pipeline = await device!.createComputePipelineAsync({ layout: "auto", compute: { module: shaderModule, entryPoint } });
      const encoder = device!.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, device!.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
      pass.dispatchWorkgroups(Math.ceil(invocations / 64)); pass.end();
      device!.queue.submit([encoder.finish()]);
    };

    // Formatting: a log sweep across every band and the ceiling, plus the
    // values the lab refuses to write.
    const sweep: number[] = [0, -0.25, 1e-7, Number.NaN, Number.POSITIVE_INFINITY, 1, 0.5, 0.25, 1.04, 9999.99];
    for (let i = 0; i <= 3000; i += 1) sweep.push(10 ** (-7 + (11.5 * i) / 3000));
    const fills = new Float32Array(sweep);
    const fillBuffer = storage(fills), glyphBuffer = storage(fills.length * 8);
    await run("formatFills", [
      { binding: 0, resource: { buffer: fillBuffer } }, { binding: 1, resource: { buffer: glyphBuffer } },
    ], fills.length);
    const words = new Uint32Array(await readBack(glyphBuffer, fills.length * 8));
    let compared = 0;
    for (let i = 0; i < fills.length; i += 1) {
      const fill = fills[i]!;
      if (Number.isFinite(fill) && fill > 0 && nearAmbiguity(fill)) continue;
      const lab = fractionReadout(fill);
      // The lab writes nothing for vacuum; the shader also writes nothing past its ceiling.
      const expected = !(fill > FRACTION_FLOOR) || !(fill < FRACTION_READOUT_CEILING)
        || !Number.isFinite(fill) ? "" : lab;
      assert.equal(decode(words[2 * i]!, words[2 * i + 1]!), expected, `fill ${fill}`);
      compared += 1;
    }
    assert.ok(compared > 2900, `only ${compared} of ${fills.length} fills were comparable`);

    // Drawing: every glyph of the alphabet, at one and two pixels per font pixel.
    const caseBuffer = storage(32), rasterBuffer = storage(RASTER_WIDTH * RASTER_HEIGHT * 8);
    for (const [text, scale, centre] of [
      ["01234567", 1, [40.25, 9.75]], ["89.e-", 2, [60.25, 16.25]], ["1.04", 2, [30.75, 12.25]],
    ] as const) {
      const [packed, length] = encode(text);
      const params = new ArrayBuffer(32);
      new Uint32Array(params, 0, 2).set([packed, length]);
      new Int32Array(params, 8, 1).set([scale]);
      new Float32Array(params, 16, 2).set(centre);
      device.queue.writeBuffer(caseBuffer, 0, params);
      await run("rasterText", [
        { binding: 2, resource: { buffer: caseBuffer } }, { binding: 3, resource: { buffer: rasterBuffer } },
      ], RASTER_WIDTH * RASTER_HEIGHT);
      const drawn = new Float32Array(await readBack(rasterBuffer, RASTER_WIDTH * RASTER_HEIGHT * 8));
      const expected = expectedRaster(text, scale, [...centre]);
      const mismatches: string[] = [];
      for (let p = 0; p < RASTER_WIDTH * RASTER_HEIGHT; p += 1) {
        if (drawn[2 * p] !== expected[2 * p] || drawn[2 * p + 1] !== expected[2 * p + 1]) {
          mismatches.push(`(${p % RASTER_WIDTH},${Math.floor(p / RASTER_WIDTH)}) drew ${drawn[2 * p]},${drawn[2 * p + 1]} expected ${expected[2 * p]},${expected[2 * p + 1]}`);
        }
      }
      assert.deepEqual(mismatches.slice(0, 8), [], `"${text}" at scale ${scale}`);
      assert.ok(expected.some(v => v === 1), "the reference raster drew nothing");
    }
    for (const buffer of [fillBuffer, glyphBuffer, caseBuffer, rasterBuffer]) buffer.destroy();
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
