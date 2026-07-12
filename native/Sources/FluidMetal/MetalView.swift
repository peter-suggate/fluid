import AppKit
import MetalKit

final class FluidMetalView: MTKView, MTKViewDelegate {
    private(set) var solver: MetalFluidSolver?
    var metricsChanged: ((FluidMetrics, GridInfo) -> Void)?
    private var lastPoint: NSPoint?
    private var draggingBody = false
    private var timer = 0
    private var renderScale: CGFloat = 1

    init(frame: CGRect, device: MTLDevice, scene: SceneDescription, quality: Quality) throws {
        super.init(frame: frame, device: device)
        renderScale = quality == .balanced ? 1 : quality == .high ? 0.9 : 0.75
        colorPixelFormat = .bgra8Unorm
        framebufferOnly = false
        autoResizeDrawable = false
        preferredFramesPerSecond = 60
        enableSetNeedsDisplay = false
        isPaused = false
        layer?.isOpaque = true
        delegate = self
        solver = try MetalFluidSolver(device: device, scene: scene, quality: quality)
    }
    required init(coder: NSCoder) { fatalError("init(coder:) is unsupported") }
    override func layout() {
        super.layout()
        let scale = window?.backingScaleFactor ?? 2
        drawableSize = CGSize(width: max(1, bounds.width * scale * renderScale), height: max(1, bounds.height * scale * renderScale))
    }
    func draw(in view: MTKView) {
        guard let drawable = currentDrawable else { return }
        solver?.encode(to: drawable)
        timer += 1
        if timer.isMultiple(of: 15), let solver { metricsChanged?(solver.metrics, solver.grid) }
    }
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    override func mouseDown(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        lastPoint = point
        let ndc = SIMD2<Float>(Float(point.x / max(bounds.width, 1) * 2 - 1), Float(point.y / max(bounds.height, 1) * 2 - 1))
        draggingBody = solver?.pickBody(ndc: ndc, aspect: Float(bounds.width / max(bounds.height, 1))) ?? false
        if event.modifierFlags.contains(.option) { draggingBody = true }
    }
    override func mouseDragged(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        if let lastPoint {
            let dx = Float(point.x-lastPoint.x), dy = Float(point.y-lastPoint.y)
            if draggingBody { solver?.dragSelectedBody(dx: dx, dy: dy) } else { solver?.orbit(dx: dx, dy: dy) }
        }
        lastPoint = point
    }
    override func mouseUp(with event: NSEvent) { lastPoint = nil; draggingBody = false }
    override func scrollWheel(with event: NSEvent) { solver?.zoom(delta: Float(event.scrollingDeltaY)) }
}
