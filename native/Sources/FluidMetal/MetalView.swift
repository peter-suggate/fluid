import AppKit
import MetalKit

final class FluidMetalView: MTKView, MTKViewDelegate {
    private(set) var solver: MetalFluidSolver?
    var metricsChanged: ((FluidMetrics, GridInfo) -> Void)?
    private var lastPoint: NSPoint?
    private var timer = 0

    init(frame: CGRect, device: MTLDevice, scene: SceneDescription, quality: Quality) throws {
        super.init(frame: frame, device: device)
        colorPixelFormat = .bgra8Unorm
        framebufferOnly = false
        preferredFramesPerSecond = 60
        enableSetNeedsDisplay = false
        isPaused = false
        layer?.isOpaque = true
        delegate = self
        solver = try MetalFluidSolver(device: device, scene: scene, quality: quality)
    }
    required init(coder: NSCoder) { fatalError("init(coder:) is unsupported") }
    func draw(in view: MTKView) {
        guard let drawable = currentDrawable else { return }
        solver?.encode(to: drawable)
        timer += 1
        if timer.isMultiple(of: 15), let solver { metricsChanged?(solver.metrics, solver.grid) }
    }
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    override func mouseDown(with event: NSEvent) { lastPoint = convert(event.locationInWindow, from: nil) }
    override func mouseDragged(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        if let lastPoint { solver?.orbit(dx: Float(point.x-lastPoint.x), dy: Float(point.y-lastPoint.y)) }
        lastPoint = point
    }
    override func mouseUp(with event: NSEvent) { lastPoint = nil }
    override func scrollWheel(with event: NSEvent) { solver?.zoom(delta: Float(event.scrollingDeltaY)) }
}
