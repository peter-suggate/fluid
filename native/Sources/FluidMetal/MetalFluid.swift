import Foundation
import Metal
import QuartzCore
import simd

struct GridInfo: Sendable {
    let nx, ny, nz: Int
    let cellSize: SIMD3<Float>
    var count: Int { nx * ny * nz }
    var label: String { "\(nx) × \(ny) × \(nz) · \(count.formatted()) cells" }
}

struct FluidMetrics: Sendable {
    var simulationMS = 0.0
    var renderMS = 0.0
    var frameMS = 0.0
    var simulationTime: Float = 0
}

private struct SolverParams {
    var dimsDt: SIMD4<Float>
    var cellGravity: SIMD4<Float>
    var containerDensity: SIMD4<Float>
    var cameraPosition: SIMD4<Float>
    var cameraTarget: SIMD4<Float>
    var viewport: SIMD4<Float>
}

final class MetalFluidSolver: @unchecked Sendable {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let scene: SceneDescription
    let quality: Quality
    let grid: GridInfo

    private let velocityA, velocityB: MTLBuffer
    private let pressureA, pressureB: MTLBuffer
    private let volumeA, volumeB: MTLBuffer
    private let params: MTLBuffer
    private let advect, jacobi, project, render: MTLComputePipelineState
    private(set) var metrics = FluidMetrics()
    private var lastFrame = CACurrentMediaTime()
    private var azimuth: Float = 0.72
    private var elevation: Float = 0.42
    private var distance: Float = 2.65
    var isRunning = true

    init(device: MTLDevice, scene: SceneDescription, quality: Quality) throws {
        self.device = device
        self.scene = scene
        self.quality = quality
        guard let queue = device.makeCommandQueue() else { throw MetalError.resource("command queue") }
        self.queue = queue

        let c = scene.container
        let h = pow(c.width_m * c.height_m * c.depth_m / Float(quality.targetCells), 1 / 3)
        let nx = max(8, Int((c.width_m / h).rounded()))
        let ny = max(8, Int((c.height_m / h).rounded()))
        let nz = max(8, Int((c.depth_m / h).rounded()))
        grid = GridInfo(nx: nx, ny: ny, nz: nz, cellSize: SIMD3(c.width_m / Float(nx), c.height_m / Float(ny), c.depth_m / Float(nz)))

        func buffer(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard let value = device.makeBuffer(length: bytes, options: .storageModePrivate) else { throw MetalError.resource(label) }
            value.label = label
            return value
        }
        velocityA = try buffer(grid.count * MemoryLayout<SIMD4<Float>>.stride, "velocity A")
        velocityB = try buffer(grid.count * MemoryLayout<SIMD4<Float>>.stride, "velocity B")
        pressureA = try buffer(grid.count * MemoryLayout<Float>.stride, "pressure A")
        pressureB = try buffer(grid.count * MemoryLayout<Float>.stride, "pressure B")
        volumeA = try buffer(grid.count * MemoryLayout<Float>.stride, "volume A")
        volumeB = try buffer(grid.count * MemoryLayout<Float>.stride, "volume B")
        guard let params = device.makeBuffer(length: MemoryLayout<SolverParams>.stride, options: .storageModeShared) else { throw MetalError.resource("parameters") }
        self.params = params

        guard let shaderURL = Bundle.module.url(forResource: "FluidKernels", withExtension: "metal") else { throw MetalError.shader("FluidKernels.metal missing") }
        let source = try String(contentsOf: shaderURL, encoding: .utf8)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: source, options: options)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw MetalError.shader("kernel \(name) missing") }
            return try device.makeComputePipelineState(function: function)
        }
        advect = try pipeline("advect")
        jacobi = try pipeline("jacobi")
        project = try pipeline("projectAndCommit")
        render = try pipeline("raymarch")
        try initializeVolume()
    }

    private func initializeVolume() throws {
        let staging = device.makeBuffer(length: grid.count * 4, options: .storageModeShared)!
        let values = staging.contents().bindMemory(to: Float.self, capacity: grid.count)
        let fill = scene.container.fillFraction
        let damHeight = max(0.92, fill)
        let damWidth = sqrt(fill / max(damHeight, 0.01))
        for z in 0..<grid.nz { for y in 0..<grid.ny { for x in 0..<grid.nx {
            let wet: Bool
            if scene.fluid.initialCondition == "dam-break" {
                wet = Float(x) + 0.5 <= damWidth * Float(grid.nx) && Float(y) + 0.5 <= damHeight * Float(grid.ny) && Float(z) + 0.5 <= damWidth * Float(grid.nz)
            } else { wet = Float(y) + 0.5 <= fill * Float(grid.ny) }
            values[x + grid.nx * (y + grid.ny * z)] = wet ? 1 : 0
        } } }
        guard let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder() else { throw MetalError.resource("initial upload") }
        blit.copy(from: staging, sourceOffset: 0, to: volumeA, destinationOffset: 0, size: grid.count * 4)
        blit.copy(from: staging, sourceOffset: 0, to: volumeB, destinationOffset: 0, size: grid.count * 4)
        blit.fill(buffer: velocityA, range: 0..<velocityA.length, value: 0)
        blit.fill(buffer: velocityB, range: 0..<velocityB.length, value: 0)
        blit.fill(buffer: pressureA, range: 0..<pressureA.length, value: 0)
        blit.fill(buffer: pressureB, range: 0..<pressureB.length, value: 0)
        blit.endEncoding(); command.commit(); command.waitUntilCompleted()
    }

    func reset() throws { metrics.simulationTime = 0; try initializeVolume() }
    func orbit(dx: Float, dy: Float) { azimuth -= dx * 0.006; elevation = max(-1.3, min(1.3, elevation + dy * 0.006)) }
    func zoom(delta: Float) { distance = max(1.1, min(6, distance * exp(delta * 0.001))) }

    func encode(to drawable: CAMetalDrawable) {
        let texture = drawable.texture
        let frameStart = CACurrentMediaTime()
        let target = SIMD3<Float>(0, scene.container.height_m * 0.42, 0)
        let camera = target + SIMD3(cos(elevation) * sin(azimuth), sin(elevation), cos(elevation) * cos(azimuth)) * distance
        params.contents().storeBytes(of: SolverParams(
            dimsDt: SIMD4(Float(grid.nx), Float(grid.ny), Float(grid.nz), scene.numerics.fixedDt_s),
            cellGravity: SIMD4(grid.cellSize.x, grid.cellSize.y, grid.cellSize.z, scene.fluid.gravity_m_s2.y),
            containerDensity: SIMD4(scene.container.width_m, scene.container.height_m, scene.container.depth_m, scene.fluid.density_kg_m3),
            cameraPosition: SIMD4(camera, 0), cameraTarget: SIMD4(target, 0),
            viewport: SIMD4(Float(texture.width), Float(texture.height), metrics.simulationTime, 0)), as: SolverParams.self)
        guard let command = queue.makeCommandBuffer() else { return }
        command.label = "Fluid frame"
        let simulationStart = CACurrentMediaTime()
        if isRunning {
            encode3D(command, pipeline: advect, buffers: [velocityA, velocityB, volumeA, volumeB, params])
            for iteration in 0..<quality.pressureIterations {
                encode3D(command, pipeline: jacobi, buffers: [velocityB, volumeB, iteration.isMultiple(of: 2) ? pressureA : pressureB, iteration.isMultiple(of: 2) ? pressureB : pressureA, params])
            }
            let finalPressure = quality.pressureIterations.isMultiple(of: 2) ? pressureA : pressureB
            encode3D(command, pipeline: project, buffers: [velocityB, velocityA, finalPressure, volumeB, volumeA, params])
            metrics.simulationTime += scene.numerics.fixedDt_s
        }
        let simulationEncoded = CACurrentMediaTime()
        guard let encoder = command.makeComputeCommandEncoder() else { return }
        encoder.label = "Volume raymarch"
        encoder.setComputePipelineState(render)
        encoder.setBuffer(volumeA, offset: 0, index: 0)
        encoder.setBuffer(params, offset: 0, index: 1)
        encoder.setTexture(texture, index: 0)
        let tw = render.threadExecutionWidth
        let th = max(1, render.maxTotalThreadsPerThreadgroup / tw)
        encoder.dispatchThreads(MTLSize(width: texture.width, height: texture.height, depth: 1), threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
        encoder.endEncoding()
        command.present(drawable)
        command.addCompletedHandler { [weak self] buffer in
            guard let self else { return }
            metrics.simulationMS = max(0, (simulationEncoded - simulationStart) * 1000)
            if buffer.gpuEndTime > buffer.gpuStartTime { metrics.renderMS = (buffer.gpuEndTime - buffer.gpuStartTime) * 1000 }
        }
        command.commit()
        let now = CACurrentMediaTime(); metrics.frameMS = (now - lastFrame) * 1000; lastFrame = now
        _ = frameStart
    }

    private func encode3D(_ command: MTLCommandBuffer, pipeline: MTLComputePipelineState, buffers: [MTLBuffer]) {
        guard let encoder = command.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        let width = min(8, pipeline.threadExecutionWidth)
        let tg = MTLSize(width: width, height: 4, depth: 4)
        encoder.dispatchThreads(MTLSize(width: grid.nx, height: grid.ny, depth: grid.nz), threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

enum MetalError: LocalizedError {
    case resource(String), shader(String)
    var errorDescription: String? { switch self { case .resource(let x): "Could not allocate Metal \(x)"; case .shader(let x): "Metal shader error: \(x)" } }
}
