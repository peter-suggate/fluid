// psoinfo a.metal ...: per file "name maxTotalThreadsPerThreadgroup staticThreadgroupMemoryLength threadExecutionWidth"
// (the MSL compiled by Metal; with the size attribute stripped the first number is the register-bound thread cap).
import Metal
import Foundation
let dev = MTLCreateSystemDefaultDevice()!
let opts = MTLCompileOptions(); opts.mathMode = .safe
for path in CommandLine.arguments.dropFirst() {
  let name = (path as NSString).lastPathComponent.replacingOccurrences(of: ".metal", with: "")
  do {
    let src = try String(contentsOfFile: path, encoding: .utf8)
    let lib = try dev.makeLibrary(source: src, options: opts)
    guard let fn = lib.makeFunction(name: "dawn_entry_point") else { print(name, "nofn"); continue }
    let pso = try dev.makeComputePipelineState(function: fn)
    print(name, pso.maxTotalThreadsPerThreadgroup, pso.staticThreadgroupMemoryLength, pso.threadExecutionWidth)
  } catch { print(name, "ERR", String(describing: error).prefix(200)) }
}
