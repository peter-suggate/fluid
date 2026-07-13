import Foundation
@preconcurrency import Metal
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
    var volumeDrift: Float = 0
    var maxSpeed: Float = 0
    var simulatedPerWallSecond: Float = 0
}

private struct SolverParams {
    var dimsDt: SIMD4<Float>
    var cellGravity: SIMD4<Float>
    var containerDensity: SIMD4<Float>
    var cameraPosition: SIMD4<Float>
    var cameraTarget: SIMD4<Float>
    var viewport: SIMD4<Float>
    var physical: SIMD4<Float>
    var boundary: SIMD4<Float>
}

private struct SimulationRecordingFrame: Sendable {
    let time: Float
    let bodies: [RigidBodyState]
}

final class MetalFluidSolver: @unchecked Sendable {
    let device: MTLDevice
    let queue: MTLCommandQueue
    private(set) var scene: SceneDescription
    let quality: Quality
    let grid: GridInfo

    private let velocityA, velocityB: MTLBuffer
    private let pressureA, pressureB: MTLBuffer
    private let volumeA, volumeB: MTLBuffer
    private let auxiliary, remapScale: MTLBuffer
    private let solidCell, solidFace: MTLBuffer
    private let params: MTLBuffer
    private let buildSolid, prepareRemap, limitRemap, applyRemap, buildAuxiliary, advect, jacobi, project, pressureTraction, clearExchange, integrateBodies, reduce, render: MTLComputePipelineState
    private let bodyBuffers: [MTLBuffer]
    private let exchangeBuffers: [MTLBuffer]
    private let reductionBuffers: [MTLBuffer]
    private let inFlight = DispatchSemaphore(value: 1)
    private let stateLock = NSLock()
    private var frameIndex = 0
    private var accumulator: Double = 0
    private let maximumSimulationStepsPerFrame = 32
    private var lastWallTime = CACurrentMediaTime()
    private var initialVolumeSum: Float = 1
    private(set) var bodies: [RigidBodyState]
    private(set) var metrics = FluidMetrics()
    private var activelyDraggedBody: Int?
    private var lastDragTime = CACurrentMediaTime()
    private var azimuth: Float = 0.72
    private var elevation: Float = 0.42
    private var distance: Float = 2.65
    var isRunning = true
    private(set) var isPlayingRecording = false
    private(set) var recordingDuration: Float = 0
    private var recordingFrames: [SimulationRecordingFrame] = []
    private var playbackStartedAt = CACurrentMediaTime()
    private var singleStepRequested = false
    var scientificView = false
    var glassVisible = true
    var selectedBodyIndex = 0

    init(device: MTLDevice, scene: SceneDescription, quality: Quality) throws {
        self.device = device
        self.scene = scene
        self.quality = quality
        bodies = scene.rigidBodies.prefix(12).map(RigidBodyState.init)
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
        auxiliary = try buffer(grid.count * MemoryLayout<SIMD4<Float>>.stride, "VOF limiter and curvature")
        remapScale = try buffer(grid.count * MemoryLayout<Float>.stride, "cut-cell remap receiver scale")
        solidCell = try buffer(grid.count * MemoryLayout<SIMD4<Float>>.stride, "cut-cell open volume and owner")
        solidFace = try buffer(grid.count * MemoryLayout<SIMD4<Float>>.stride, "cut-cell open face apertures")
        guard let params = device.makeBuffer(length: MemoryLayout<SolverParams>.stride, options: .storageModeShared) else { throw MetalError.resource("parameters") }
        self.params = params
        func sharedBuffer(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard let value = device.makeBuffer(length: bytes, options: .storageModeShared) else { throw MetalError.resource(label) }
            value.label = label
            return value
        }
        bodyBuffers = try (0..<3).map { try sharedBuffer(12 * MemoryLayout<BodyGPU>.stride, "body state \($0)") }
        exchangeBuffers = try (0..<3).map { try sharedBuffer(12 * 8 * 4, "body exchange \($0)") }
        reductionBuffers = try (0..<3).map { try sharedBuffer(4 * 4, "diagnostics \($0)") }

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
        buildSolid = try pipeline("buildSolidGeometry")
        prepareRemap = try pipeline("prepareSolidRemap")
        limitRemap = try pipeline("limitSolidRemap")
        applyRemap = try pipeline("applySolidRemap")
        buildAuxiliary = try pipeline("buildAuxiliary")
        advect = try pipeline("advect")
        jacobi = try pipeline("jacobi")
        project = try pipeline("projectAndCommit")
        pressureTraction = try pipeline("accumulatePressureTraction")
        clearExchange = try pipeline("clearRigidExchange")
        integrateBodies = try pipeline("integrateRigidBodies")
        reduce = try pipeline("reduceDiagnostics")
        render = try pipeline("raymarch")
        try initializeVolume()
        recordingFrames = [SimulationRecordingFrame(time: 0, bodies: bodies)]
    }

    private func initializeVolume() throws {
        let staging = device.makeBuffer(length: grid.count * 4, options: .storageModeShared)!
        let values = staging.contents().bindMemory(to: Float.self, capacity: grid.count)
        let fill = scene.container.fillFraction
        let damHeight = max(0.92, fill)
        let damWidth = sqrt(fill / max(damHeight, 0.01))
        var initial: Float = 0
        for z in 0..<grid.nz { for y in 0..<grid.ny { for x in 0..<grid.nx {
            let wet: Bool
            if scene.fluid.initialCondition == "dam-break" {
                wet = Float(x) + 0.5 <= damWidth * Float(grid.nx) && Float(y) + 0.5 <= damHeight * Float(grid.ny) && Float(z) + 0.5 <= damWidth * Float(grid.nz)
            } else { wet = Float(y) + 0.5 <= fill * Float(grid.ny) }
            values[x + grid.nx * (y + grid.ny * z)] = wet ? 1 : 0
            if wet { initial += 1 }
        } } }
        initialVolumeSum = max(initial, 1)
        guard let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder() else { throw MetalError.resource("initial upload") }
        blit.copy(from: staging, sourceOffset: 0, to: volumeA, destinationOffset: 0, size: grid.count * 4)
        blit.copy(from: staging, sourceOffset: 0, to: volumeB, destinationOffset: 0, size: grid.count * 4)
        blit.fill(buffer: velocityA, range: 0..<velocityA.length, value: 0)
        blit.fill(buffer: velocityB, range: 0..<velocityB.length, value: 0)
        blit.fill(buffer: pressureA, range: 0..<pressureA.length, value: 0)
        blit.fill(buffer: pressureB, range: 0..<pressureB.length, value: 0)
        blit.endEncoding(); command.commit(); command.waitUntilCompleted()
    }

    func reset() throws {
        metrics = FluidMetrics(); accumulator = 0; lastWallTime = CACurrentMediaTime()
        bodies = scene.rigidBodies.prefix(12).map(RigidBodyState.init)
        isPlayingRecording = false
        recordingDuration = 0
        recordingFrames = [SimulationRecordingFrame(time: 0, bodies: bodies)]
        try initializeVolume()
    }
    var canReplay: Bool { metrics.simulationTime > 0 || recordingDuration > 0 }
    func requestSingleStep() { if !isPlayingRecording { singleStepRequested = true } }

    func startPlayback() throws {
        inFlight.wait()
        defer { inFlight.signal() }
        guard metrics.simulationTime > 0 else { return }
        stateLock.lock()
        let endpoint = SimulationRecordingFrame(time: metrics.simulationTime, bodies: bodies)
        if abs(metrics.simulationTime - recordingDuration) < 1e-6, !recordingFrames.isEmpty { recordingFrames[recordingFrames.count - 1] = endpoint }
        else if metrics.simulationTime > recordingDuration { recordingFrames.append(endpoint) }
        recordingDuration = max(recordingDuration, metrics.simulationTime)
        stateLock.unlock()
        guard canReplay else { return }
        isRunning = false
        isPlayingRecording = true
        activelyDraggedBody = nil
        metrics = FluidMetrics()
        accumulator = 0
        bodies = recordingFrames[0].bodies
        playbackStartedAt = CACurrentMediaTime()
        lastWallTime = playbackStartedAt
        try initializeVolume()
    }

    private func playbackBodies(at time: Float) -> [RigidBodyState] {
        guard let first = recordingFrames.first else { return bodies }
        if time <= first.time { return first.bodies }
        guard let last = recordingFrames.last else { return first.bodies }
        if time >= last.time { return last.bodies }
        var low = 0, high = recordingFrames.count - 1
        while low + 1 < high {
            let middle = (low + high) / 2
            if recordingFrames[middle].time <= time { low = middle } else { high = middle }
        }
        let a = recordingFrames[low], b = recordingFrames[high]
        guard a.bodies.count == b.bodies.count, b.time > a.time else { return a.bodies }
        let amount = max(0, min(1, (time - a.time) / (b.time - a.time)))
        return zip(a.bodies, b.bodies).map { lower, upper in
            guard lower.description.id == upper.description.id else { return lower }
            var value = lower
            value.position = simd_mix(lower.position, upper.position, SIMD3(repeating: amount))
            value.orientation = simd_slerp(lower.orientation, upper.orientation, amount)
            value.linearVelocity = simd_mix(lower.linearVelocity, upper.linearVelocity, SIMD3(repeating: amount))
            value.angularVelocity = simd_mix(lower.angularVelocity, upper.angularVelocity, SIMD3(repeating: amount))
            value.displacedVolume = lower.displacedVolume + (upper.displacedVolume - lower.displacedVolume) * amount
            return value
        }
    }
    func orbit(dx: Float, dy: Float) { azimuth -= dx * 0.006; elevation = max(-1.3, min(1.3, elevation + dy * 0.006)) }
    func zoom(delta: Float) { distance = max(1.1, min(6, distance * exp(delta * 0.001))) }
    func moveBody(index: Int, to position: SIMD3<Float>, velocity: SIMD3<Float> = .zero) {
        guard bodies.indices.contains(index) else { return }
        bodies[index].position = position; bodies[index].linearVelocity = velocity
    }
    func dragSelectedBody(dx: Float, dy: Float) {
        guard !isPlayingRecording, bodies.indices.contains(selectedBodyIndex) else { return }
        let right = SIMD3<Float>(cos(azimuth), 0, -sin(azimuth))
        let up = SIMD3<Float>(-sin(elevation)*sin(azimuth), cos(elevation), -sin(elevation)*cos(azimuth))
        let displacement = (right * dx + up * dy) * 0.0025
        let now = CACurrentMediaTime(), dt = Float(max(1.0 / 240.0, min(now - lastDragTime, 1.0 / 20.0)))
        lastDragTime = now
        stateLock.lock()
        bodies[selectedBodyIndex].position += displacement
        let sweptVelocity = displacement / dt
        let speed = simd_length(sweptVelocity)
        bodies[selectedBodyIndex].linearVelocity = speed > 8 ? sweptVelocity * (8 / speed) : sweptVelocity
        stateLock.unlock()
    }
    func pickBody(ndc: SIMD2<Float>, aspect: Float) -> Bool {
        guard !isPlayingRecording else { return false }
        let target = SIMD3<Float>(0, scene.container.height_m * 0.42, 0)
        let origin = target + SIMD3(cos(elevation) * sin(azimuth), sin(elevation), cos(elevation) * cos(azimuth)) * distance
        let forward = simd_normalize(target - origin)
        let right = simd_normalize(simd_cross(forward, SIMD3<Float>(0, 1, 0)))
        let up = simd_normalize(simd_cross(right, forward))
        let direction = simd_normalize(forward + right * ndc.x * aspect * 0.72 + up * ndc.y * 0.72)
        var bestDistance = Float.greatestFiniteMagnitude
        var bestIndex: Int?
        for index in bodies.indices {
            let offset = origin - bodies[index].position
            let projected = simd_dot(offset, direction)
            let discriminant = projected * projected - simd_dot(offset, offset) + bodies[index].supportRadius * bodies[index].supportRadius
            guard discriminant >= 0 else { continue }
            let distance = -projected - sqrt(discriminant)
            if distance > 0, distance < bestDistance { bestDistance = distance; bestIndex = index }
        }
        guard let bestIndex else { return false }
        selectedBodyIndex = bestIndex; activelyDraggedBody = bestIndex; lastDragTime = CACurrentMediaTime()
        return true
    }
    func endBodyDrag() { activelyDraggedBody = nil }
    func cameraPreset(_ name: String) {
        switch name {
        case "front": azimuth = 0; elevation = 0.08
        case "side": azimuth = .pi/2; elevation = 0.08
        case "top": azimuth = 0; elevation = 1.34; distance = 2.25
        default: azimuth = 0.72; elevation = 0.42; distance = 2.65
        }
    }
    func dropSelectedBody() {
        guard !isPlayingRecording else { return }
        guard bodies.indices.contains(selectedBodyIndex) else { return }
        bodies[selectedBodyIndex].position.y = scene.container.height_m + bodies[selectedBodyIndex].supportRadius + 0.2
        bodies[selectedBodyIndex].linearVelocity = .zero
        bodies[selectedBodyIndex].angularVelocity = SIMD3(0.8, 0.35, -0.5)
    }
    func addBody(shape: String) {
        guard !isPlayingRecording, bodies.count < 12 else { return }
        let index = bodies.count + 1, sphere = shape == "sphere"
        let value = RigidBodyDescription(
            id: "native-\(shape)-\(index)", name: "\(shape.capitalized) \(index)", shape: shape,
            dimensions_m: sphere ? Vec3(x: 0.07, y: 0.07, z: 0.07) : Vec3(x: 0.13, y: 0.11, z: 0.12),
            density_kg_m3: sphere ? 500 : 1100, position_m: Vec3(x: 0, y: scene.container.height_m + 0.25, z: 0),
            orientation: Quaternion(w: 1, x: 0, y: 0, z: 0), linearVelocity_m_s: Vec3(x: 0, y: 0, z: 0),
            angularVelocity_rad_s: Vec3(x: 0.4, y: 0.2, z: 0.7), restitution: 0.3, friction: 0.45
        )
        scene.rigidBodies.append(value); bodies.append(RigidBodyState(value)); selectedBodyIndex = bodies.count - 1
    }
    func removeSelectedBody() {
        guard !isPlayingRecording, bodies.indices.contains(selectedBodyIndex) else { return }
        bodies.remove(at: selectedBodyIndex); scene.rigidBodies.remove(at: selectedBodyIndex)
        selectedBodyIndex = min(selectedBodyIndex, max(0, bodies.count - 1))
    }
    func updateSelectedBody(density: Float, size: Float) {
        guard !isPlayingRecording, bodies.indices.contains(selectedBodyIndex) else { return }
        var value = bodies[selectedBodyIndex].description
        value.density_kg_m3 = max(1, density)
        let old = max(value.dimensions_m.x, 1e-6), ratio = max(0.01, size) / old
        value.dimensions_m = Vec3(x: size, y: value.dimensions_m.y * ratio, z: value.dimensions_m.z * ratio)
        value.position_m = Vec3(x: bodies[selectedBodyIndex].position.x, y: bodies[selectedBodyIndex].position.y, z: bodies[selectedBodyIndex].position.z)
        scene.rigidBodies[selectedBodyIndex] = value; bodies[selectedBodyIndex] = RigidBodyState(value)
    }

    func encode(to drawable: CAMetalDrawable) {
        let texture = drawable.texture
        guard inFlight.wait(timeout: .now()) == .success else { return }
        let frameStart = CACurrentMediaTime()
        let frame = frameIndex % 3; frameIndex += 1
        let now = CACurrentMediaTime()
        let wallDelta = min(max(now - lastWallTime, 0), 0.25); lastWallTime = now
        let playingRecording = isPlayingRecording
        if playingRecording {
            let playbackTime = min(recordingDuration, Float(now - playbackStartedAt))
            bodies = playbackBodies(at: playbackTime)
            accumulator = min(accumulator + wallDelta, 0.25)
        }
        if singleStepRequested { accumulator = max(accumulator, Double(scene.numerics.fixedDt_s)); singleStepRequested = false }
        else if isRunning && !playingRecording {
            // Preserve the stability-limited dt and catch up with GPU substeps.
            accumulator = min(accumulator + wallDelta, 0.25)
        }
        let target = SIMD3<Float>(0, scene.container.height_m * 0.42, 0)
        let camera = target + SIMD3(cos(elevation) * sin(azimuth), sin(elevation), cos(elevation) * cos(azimuth)) * distance
        let minimumCell = min(grid.cellSize.x, grid.cellSize.y, grid.cellSize.z)
        let waveSpeed = sqrt(abs(scene.fluid.gravity_m_s2.y) * scene.container.height_m * max(scene.container.fillFraction, 0.01))
        let bodySpeed = bodies.reduce(Float.zero) { max($0, simd_length($1.linearVelocity)) }
        let gravitySpeedGrowth = abs(scene.fluid.gravity_m_s2.y) * Float(min(accumulator, 0.25))
        let characteristicSpeed = max(waveSpeed, metrics.maxSpeed + gravitySpeedGrowth, bodySpeed, 0.1)
        let advectiveDt = 0.45 * minimumCell / characteristicSpeed
        let capillaryDt = scene.fluid.surfaceTension_N_m > 0 ? 0.4 * sqrt(scene.fluid.density_kg_m3 * minimumCell * minimumCell * minimumCell / (.pi * scene.fluid.surfaceTension_N_m)) : scene.numerics.fixedDt_s
        let stepDt = min(scene.numerics.fixedDt_s, advectiveDt, capillaryDt)
        params.contents().storeBytes(of: SolverParams(
            dimsDt: SIMD4(Float(grid.nx), Float(grid.ny), Float(grid.nz), stepDt),
            cellGravity: SIMD4(grid.cellSize.x, grid.cellSize.y, grid.cellSize.z, scene.fluid.gravity_m_s2.y),
            containerDensity: SIMD4(scene.container.width_m, scene.container.height_m, scene.container.depth_m, scene.fluid.density_kg_m3),
            cameraPosition: SIMD4(camera, 0), cameraTarget: SIMD4(target, 0),
            viewport: SIMD4(Float(texture.width), Float(texture.height), metrics.simulationTime, 0),
            physical: SIMD4(scene.fluid.dynamicViscosity_Pa_s, scene.fluid.surfaceTension_N_m, glassVisible ? 1 : 0, Float(selectedBodyIndex)),
            boundary: SIMD4(scene.container.fluidWallMode == "no-slip" ? 1 : 0, scene.container.top == "closed" ? 1 : 0, Float(bodies.count), scientificView ? 1 : 0)
        ), as: SolverParams.self)
        let bodyBuffer = bodyBuffers[frame], exchange = exchangeBuffers[frame], reductions = reductionBuffers[frame]
        let bodyPointer = bodyBuffer.contents().bindMemory(to: BodyGPU.self, capacity: 12)
        stateLock.lock()
        let draggedBody = activelyDraggedBody
        for index in bodies.indices {
            var value = bodies[index].gpuValue
            value.angularVelocity.w = playingRecording || index == draggedBody ? 1 : 0
            bodyPointer[index] = value
        }
        stateLock.unlock()
        guard let command = queue.makeCommandBuffer() else { inFlight.signal(); return }
        command.label = "Fluid frame"
        if let blit = command.makeBlitCommandEncoder() {
            blit.fill(buffer: reductions, range: 0..<reductions.length, value: 0)
            blit.endEncoding()
        }
        guard let encoder = command.makeComputeCommandEncoder() else { inFlight.signal(); return }
        encoder.label = "Fluid simulation and raymarch"
        var encodedSteps = 0
        if isRunning || playingRecording || accumulator >= Double(stepDt) {
            let playbackEnd = playingRecording ? recordingDuration : scene.duration_s
            while accumulator >= Double(stepDt), metrics.simulationTime + stepDt <= playbackEnd + 1e-6, encodedSteps < maximumSimulationStepsPerFrame {
                if !bodies.isEmpty { dispatchBodies(encoder, pipeline: clearExchange, buffers: [exchange, params], count: bodies.count) }
                dispatch3D(encoder, pipeline: buildSolid, buffers: [bodyBuffer, solidCell, solidFace, params])
                // Two conservative cover/uncover sweeps move displaced liquid through
                // neighboring open cells without atomics or volume loss.
                dispatch3D(encoder, pipeline: prepareRemap, buffers: [volumeA, solidCell, auxiliary, params])
                dispatch3D(encoder, pipeline: limitRemap, buffers: [auxiliary, remapScale, params])
                dispatch3D(encoder, pipeline: applyRemap, buffers: [volumeA, volumeB, auxiliary, remapScale, params])
                dispatch3D(encoder, pipeline: prepareRemap, buffers: [volumeB, solidCell, auxiliary, params])
                dispatch3D(encoder, pipeline: limitRemap, buffers: [auxiliary, remapScale, params])
                dispatch3D(encoder, pipeline: applyRemap, buffers: [volumeB, volumeA, auxiliary, remapScale, params])
                dispatch3D(encoder, pipeline: buildAuxiliary, buffers: [velocityA, volumeA, auxiliary, solidCell, solidFace, params])
                dispatch3D(encoder, pipeline: advect, buffers: [velocityA, velocityB, volumeA, volumeB, auxiliary, solidFace, params])
                for iteration in 0..<quality.pressureIterations {
                    dispatch3D(encoder, pipeline: jacobi, buffers: [velocityB, volumeB, iteration.isMultiple(of: 2) ? pressureA : pressureB, iteration.isMultiple(of: 2) ? pressureB : pressureA, solidCell, solidFace, bodyBuffer, params])
                }
                let finalPressure = quality.pressureIterations.isMultiple(of: 2) ? pressureA : pressureB
                dispatch3D(encoder, pipeline: project, buffers: [velocityB, velocityA, finalPressure, volumeB, volumeA, solidCell, solidFace, bodyBuffer, params])
                if !bodies.isEmpty { dispatch3D(encoder, pipeline: pressureTraction, buffers: [finalPressure, volumeA, solidCell, solidFace, bodyBuffer, exchange, params]) }
                if !bodies.isEmpty { dispatchBodies(encoder, pipeline: integrateBodies, buffers: [bodyBuffer, exchange, params], count: bodies.count) }
                metrics.simulationTime += stepDt
                if !playingRecording, metrics.simulationTime >= scene.duration_s { isRunning = false }
                accumulator -= Double(stepDt); encodedSteps += 1
            }
            if playingRecording, metrics.simulationTime + stepDt > recordingDuration {
                metrics.simulationTime = recordingDuration
                accumulator = 0
                isPlayingRecording = false
            }
        }
        dispatch3D(encoder, pipeline: reduce, buffers: [velocityA, volumeA, reductions, params])
        encoder.setComputePipelineState(render)
        encoder.setBuffer(volumeA, offset: 0, index: 0)
        encoder.setBuffer(params, offset: 0, index: 1)
        encoder.setBuffer(bodyBuffer, offset: 0, index: 2)
        encoder.setTexture(texture, index: 0)
        let tw = render.threadExecutionWidth
        let th = max(1, render.maxTotalThreadsPerThreadgroup / tw)
        encoder.dispatchThreads(MTLSize(width: texture.width, height: texture.height, depth: 1), threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
        encoder.endEncoding()
        command.present(drawable)
        let completedSteps = encodedSteps, bodyCount = bodies.count, completedSimulationTime = metrics.simulationTime
        command.addCompletedHandler { [weak self] buffer in
            guard let self else { return }
            if buffer.gpuEndTime > buffer.gpuStartTime { metrics.renderMS = (buffer.gpuEndTime - buffer.gpuStartTime) * 1000 }
            let reduction = reductions.contents().bindMemory(to: UInt32.self, capacity: 4)
            metrics.volumeDrift = (Float(reduction[0]) / 2048 - initialVolumeSum) / initialVolumeSum
            metrics.maxSpeed = Float(bitPattern: reduction[1])
            metrics.simulatedPerWallSecond = wallDelta > 0 ? Float(completedSteps) * stepDt / Float(wallDelta) : 0
            if bodyCount > 0 {
                let words = exchange.contents().bindMemory(to: Int32.self, capacity: 12 * 8)
                let gpuBodies = bodyBuffer.contents().bindMemory(to: BodyGPU.self, capacity: 12)
                stateLock.lock()
                for index in 0..<bodyCount {
                    let base = index * 8
                    if index != activelyDraggedBody { bodies[index].synchronize(from: gpuBodies[index]) }
                    bodies[index].displacedVolume = Float(words[base+6]) / 65536 * grid.cellSize.x * grid.cellSize.y * grid.cellSize.z
                }
                stateLock.unlock()
            }
            if !playingRecording, completedSteps > 0,
               (completedSimulationTime - recordingDuration >= 1.0 / 60.0 || completedSimulationTime >= scene.duration_s) {
                stateLock.lock()
                if abs(completedSimulationTime - recordingDuration) < 1e-6, !recordingFrames.isEmpty {
                    recordingFrames[recordingFrames.count - 1] = SimulationRecordingFrame(time: completedSimulationTime, bodies: bodies)
                } else {
                    recordingFrames.append(SimulationRecordingFrame(time: completedSimulationTime, bodies: bodies))
                }
                recordingDuration = max(recordingDuration, completedSimulationTime)
                stateLock.unlock()
            }
            inFlight.signal()
        }
        command.commit()
        let frameNow = CACurrentMediaTime()
        metrics.simulationMS = Double(encodedSteps) * Double(stepDt) * 1000
        metrics.frameMS = (frameNow - frameStart) * 1000
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

    private func dispatch3D(_ encoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, buffers: [MTLBuffer]) {
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        let width = min(8, pipeline.threadExecutionWidth)
        encoder.dispatchThreads(MTLSize(width: grid.nx, height: grid.ny, depth: grid.nz), threadsPerThreadgroup: MTLSize(width: width, height: 4, depth: 4))
    }

    private func dispatchBodies(_ encoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, buffers: [MTLBuffer], count: Int) {
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(max(1, count), pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
    }

    private func encodeBodies(_ command: MTLCommandBuffer, pipeline: MTLComputePipelineState, buffers: [MTLBuffer], count: Int) {
        guard let encoder = command.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(max(1, count), pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }

    func runHeadlessValidation(steps: Int = 3) throws -> (volumeDrift: Float, maxSpeed: Float) {
        let bodyBuffer = bodyBuffers[0], exchange = exchangeBuffers[0], reductions = reductionBuffers[0]
        let bodyPointer = bodyBuffer.contents().bindMemory(to: BodyGPU.self, capacity: 12)
        for index in bodies.indices { bodyPointer[index] = bodies[index].gpuValue }
        params.contents().storeBytes(of: SolverParams(
            dimsDt: SIMD4(Float(grid.nx), Float(grid.ny), Float(grid.nz), scene.numerics.fixedDt_s),
            cellGravity: SIMD4(grid.cellSize.x, grid.cellSize.y, grid.cellSize.z, scene.fluid.gravity_m_s2.y),
            containerDensity: SIMD4(scene.container.width_m, scene.container.height_m, scene.container.depth_m, scene.fluid.density_kg_m3),
            cameraPosition: SIMD4(0, 0, 0, 0), cameraTarget: SIMD4(0, 0, 0, 0), viewport: SIMD4(1, 1, 0, 0),
            physical: SIMD4(scene.fluid.dynamicViscosity_Pa_s, scene.fluid.surfaceTension_N_m, 1, 0),
            boundary: SIMD4(scene.container.fluidWallMode == "no-slip" ? 1 : 0, scene.container.top == "closed" ? 1 : 0, Float(bodies.count), 0)
        ), as: SolverParams.self)
        for _ in 0..<max(1, steps) {
            guard let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder() else { throw MetalError.resource("validation command") }
            blit.fill(buffer: exchange, range: 0..<exchange.length, value: 0); blit.fill(buffer: reductions, range: 0..<reductions.length, value: 0); blit.endEncoding()
            encode3D(command, pipeline: buildSolid, buffers: [bodyBuffer, solidCell, solidFace, params])
            encode3D(command, pipeline: prepareRemap, buffers: [volumeA, solidCell, auxiliary, params])
            encode3D(command, pipeline: limitRemap, buffers: [auxiliary, remapScale, params])
            encode3D(command, pipeline: applyRemap, buffers: [volumeA, volumeB, auxiliary, remapScale, params])
            encode3D(command, pipeline: prepareRemap, buffers: [volumeB, solidCell, auxiliary, params])
            encode3D(command, pipeline: limitRemap, buffers: [auxiliary, remapScale, params])
            encode3D(command, pipeline: applyRemap, buffers: [volumeB, volumeA, auxiliary, remapScale, params])
            encode3D(command, pipeline: buildAuxiliary, buffers: [velocityA, volumeA, auxiliary, solidCell, solidFace, params])
            encode3D(command, pipeline: advect, buffers: [velocityA, velocityB, volumeA, volumeB, auxiliary, solidFace, params])
            for iteration in 0..<quality.pressureIterations { encode3D(command, pipeline: jacobi, buffers: [velocityB, volumeB, iteration.isMultiple(of: 2) ? pressureA : pressureB, iteration.isMultiple(of: 2) ? pressureB : pressureA, solidCell, solidFace, bodyBuffer, params]) }
            let finalPressure = quality.pressureIterations.isMultiple(of: 2) ? pressureA : pressureB
            encode3D(command, pipeline: project, buffers: [velocityB, velocityA, finalPressure, volumeB, volumeA, solidCell, solidFace, bodyBuffer, params])
            if !bodies.isEmpty { encode3D(command, pipeline: pressureTraction, buffers: [finalPressure, volumeA, solidCell, solidFace, bodyBuffer, exchange, params]) }
            if !bodies.isEmpty { encodeBodies(command, pipeline: integrateBodies, buffers: [bodyBuffer, exchange, params], count: bodies.count) }
            encode3D(command, pipeline: reduce, buffers: [velocityA, volumeA, reductions, params])
            command.commit(); command.waitUntilCompleted()
            if command.status == .error { throw command.error ?? MetalError.resource("validation GPU execution") }
        }
        stateLock.lock()
        let gpuBodies = bodyBuffer.contents().bindMemory(to: BodyGPU.self, capacity: 12)
        for index in bodies.indices { bodies[index].synchronize(from: gpuBodies[index]) }
        stateLock.unlock()
        let values = reductions.contents().bindMemory(to: UInt32.self, capacity: 4)
        let drift = (Float(values[0]) / 2048 - initialVolumeSum) / initialVolumeSum
        return (drift, Float(bitPattern: values[1]))
    }
}

enum MetalError: LocalizedError {
    case resource(String), shader(String)
    var errorDescription: String? { switch self { case .resource(let x): "Could not allocate Metal \(x)"; case .shader(let x): "Metal shader error: \(x)" } }
}
