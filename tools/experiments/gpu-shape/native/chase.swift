// Native Metal ceiling for M0(a) (docs/plans/gpu-shape/m0a-shape.md): N independent latency-bound dispatches in ONE
// compute encoder with MTLDispatchType.serial vs .concurrent (and concurrent with a buffer barrier between every
// dispatch), against one packed dispatch of N x the workgroups. Same pointer-chase kernel as the WebGPU sweep.
//   swiftc -O chase.swift -o chase && tools/gpu/fluid-gpu-run.sh timing ./chase <out.jsonl>
// Timing: each measured command buffer holds only the measured encoder; its GPU time is gpuEndTime - gpuStartTime.
// A heater command buffer precedes it and a filler follows (the GPU never idles, so the clock stays in its top
// state); a probe command buffer (one 32-lane dependent FMA chain) follows every measurement as a clock check.
import Foundation
import Metal

let src = """
#include <metal_stdlib>
using namespace metal;
static uint hsh(uint x0) { uint x = x0 * 747796405u + 2891336453u; x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u; return (x >> 22u) ^ x; }
kernel void chase(device const uint* nxt [[buffer(0)]], device uint* outb [[buffer(1)]], constant uint4& prm [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
  uint i = hsh(gid + prm.w + prm.z) & prm.x;
  for (uint d = 0; d < prm.y; d++) { i = nxt[i]; }
  outb[gid] = i;
}
kernel void alu(device float* o [[buffer(0)]], constant uint4& it [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  float x = float(gid) * 1e-6f; float y = x + 1.0f;
  for (uint i = 0; i < it.x; i++) { x = fma(x, 0.999f, 0.25f); y = fma(y, x, -0.5f); }
  o[gid] = x + y;
}
"""
let outPath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "native.jsonl"
FileManager.default.createFile(atPath: outPath, contents: nil)
let outFile = FileHandle(forWritingAtPath: outPath)!
func emit(_ d: [String: Any]) {
  let j = try! JSONSerialization.data(withJSONObject: d, options: [.sortedKeys]); outFile.write(j); outFile.write("\n".data(using: .utf8)!)
}

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let chase = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "chase")!)
let alu = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "alu")!)
print("device \(dev.name), chase maxThreads \(chase.maxTotalThreadsPerThreadgroup)")
emit(["kind": "setup", "device": dev.name, "chaseMaxThreads": chase.maxTotalThreadsPerThreadgroup])

func permutation(_ bytes: Int) -> MTLBuffer {
  let n = bytes / 4
  let buf = dev.makeBuffer(length: bytes, options: .storageModeShared)!
  let a = buf.contents().bindMemory(to: UInt32.self, capacity: n)
  var p = [UInt32](repeating: 0, count: n); for i in 0..<n { p[i] = UInt32(i) }
  var g = SystemRandomNumberGenerator()
  var i = n - 1; while i > 0 { let j = Int.random(in: 0...i, using: &g); p.swapAt(i, j); i -= 1 }
  for k in 0..<n { a[Int(p[k])] = p[(k + 1) % n] }
  return buf
}
let SMALL = 8 << 20, BIG = 256 << 20
let perm = [SMALL: permutation(SMALL), BIG: permutation(BIG)]
let outBuf = dev.makeBuffer(length: 4 << 22, options: .storageModePrivate)!
func prm(_ bytes: Int, _ depth: Int, _ seed: Int, _ base: Int) -> MTLBuffer {
  var v: [UInt32] = [UInt32(bytes / 4 - 1), UInt32(depth), UInt32(seed), UInt32(base)]
  return dev.makeBuffer(bytes: &v, length: 16, options: .storageModeShared)!
}
func itBuf(_ n: Int) -> MTLBuffer { var v: [UInt32] = [UInt32(n), 0, 0, 0]; return dev.makeBuffer(bytes: &v, length: 16, options: .storageModeShared)! }
let aluOut = dev.makeBuffer(length: 4 * 65536, options: .storageModePrivate)!
var heatIt = itBuf(10000)
let probeIt = itBuf(2048)

func aluCB(_ groups: Int, _ lanes: Int, _ it: MTLBuffer) -> MTLCommandBuffer {
  let cb = queue.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(alu); e.setBuffer(aluOut, offset: 0, index: 0); e.setBuffer(it, offset: 0, index: 1)
  e.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: lanes, height: 1, depth: 1))
  e.endEncoding(); return cb
}
func gpuUs(_ cb: MTLCommandBuffer) -> Double { (cb.gpuEndTime - cb.gpuStartTime) * 1e6 }

struct Disp { let bytes: Int; let depth: Int; let seed: Int; let base: Int; let L: Int; let G: Int; let out: Int }
enum Mode: String { case serial, concurrent, concurrentBarrier }
var prmCache: [String: MTLBuffer] = [:]
func prmC(_ d: Disp) -> MTLBuffer { let k = "\(d.bytes)/\(d.depth)/\(d.seed)/\(d.base)"; if let b = prmCache[k] { return b }; let b = prm(d.bytes, d.depth, d.seed, d.base); prmCache[k] = b; return b }

// One measurement: heater cb, then REPS measured cbs (back-to-back), filler, probe. Returns min measured time.
func measure(_ list: [Disp], _ mode: Mode, reps: Int = 5) -> (min: Double, probe: Double) {
  aluCB(256, 256, heatIt).commit()
  var cbs: [MTLCommandBuffer] = []
  for _ in 0...reps {
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder(dispatchType: mode == .serial ? .serial : .concurrent)!
    e.setComputePipelineState(chase)
    for (i, d) in list.enumerated() {
      if mode == .concurrentBarrier && i > 0 { e.memoryBarrier(scope: .buffers) }
      e.setBuffer(perm[d.bytes]!, offset: 0, index: 0); e.setBuffer(outBuf, offset: 4 * d.out, index: 1); e.setBuffer(prmC(d), offset: 0, index: 2)
      e.dispatchThreadgroups(MTLSize(width: d.G, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: d.L, height: 1, depth: 1))
    }
    e.endEncoding(); cb.commit(); cbs.append(cb)
  }
  let probe = aluCB(1, 32, probeIt); probe.commit()
  aluCB(256, 256, heatIt).commit() // filler
  probe.waitUntilCompleted()
  let ts = cbs.dropFirst().map(gpuUs)
  return (ts.min()!, gpuUs(probe))
}

// Ramp the clock: sustained ALU load until the probe stops improving.
func ramp() {
  var best = Double.infinity, stable = 0; let t = Date()
  while Date().timeIntervalSince(t) < 5 {
    for _ in 0..<4 { aluCB(256, 256, heatIt).commit() }
    let p = aluCB(1, 32, probeIt); p.commit(); p.waitUntilCompleted()
    let v = gpuUs(p); if v > 0.97 * best { stable += 1 } else { stable = 0 }; best = min(best, v)
    if stable >= 4 && Date().timeIntervalSince(t) > 0.3 { break }
  }
  print("ramp: probe \(best) us"); emit(["kind": "ramp", "probe": best])
}
ramp()
// Heater at about 1.5 ms.
do { let cb = aluCB(256, 256, heatIt); cb.commit(); cb.waitUntilCompleted(); let n = max(64, Int(10000 * 1500 / gpuUs(cb))); heatIt = itBuf(n); print("heater iters \(n)") }

for bytes in [SMALL, BIG] {
  for (L, G, D) in [(64, 8, 16), (256, 40, 16), (32, 1, 1), (64, 8, 4)] {
    for N in [1, 2, 4, 8, 16, 32, 64] {
      let list = (0..<N).map { b in Disp(bytes: bytes, depth: D, seed: 1 + b, base: b * G * L, L: L, G: G, out: b * G * L) }
      let s = measure(list, .serial), c = measure(list, .concurrent), cb = measure(list, .concurrentBarrier)
      let packed = measure([Disp(bytes: bytes, depth: D, seed: 1, base: 0, L: L, G: G * N, out: 0)], .serial)
      emit(["kind": "native", "bytes": bytes, "L": L, "G": G, "D": D, "N": N, "serial": s.min, "concurrent": c.min, "concurrentBarrier": cb.min, "packed": packed.min, "probe": s.probe])
      print(String(format: "%3d MB L%-4d G%-3d D%-2d N%-2d serial %8.1f  concurrent %8.1f  conc+barrier %8.1f  packed %8.1f  probe %.1f", bytes >> 20, L, G, D, N, s.min, c.min, cb.min, packed.min, s.probe))
    }
  }
}
// Empty command buffer: the gpuStart/EndTime floor of a cb with one encoder and one trivial dispatch.
let floor = measure([Disp(bytes: SMALL, depth: 0, seed: 1, base: 0, L: 32, G: 1, out: 0)], .serial)
emit(["kind": "nativeFloor", "min": floor.min, "probe": floor.probe])
print("floor (one D=0 dispatch) \(floor.min) us")
