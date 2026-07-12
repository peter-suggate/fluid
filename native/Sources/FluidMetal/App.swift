import AppKit
import Metal

@MainActor
@main
enum FluidMetalMain {
    static func main() {
        if CommandLine.arguments.contains("--self-test") {
            SelfTest.run()
            return
        }
        if CommandLine.arguments.contains("--benchmark") {
            Benchmark.run()
            return
        }
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.setActivationPolicy(.regular)
        app.run()
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private var metalView: FluidMetalView!
    private let metrics = NSTextField(labelWithString: "Starting Metal…")
    private let status = NSTextField(labelWithString: "")
    private var quality: Quality = .m1Max
    private var scene = SceneDescription.browserDefault

    func applicationDidFinishLaunching(_ notification: Notification) {
        do { try createWindow() }
        catch { NSAlert(error: error).runModal(); NSApp.terminate(nil); return }
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    private func createWindow() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw MetalError.resource("device") }
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1440, height: 900),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "Fluid Lab · Metal"
        window.titlebarAppearsTransparent = true
        window.center()

        let root = NSView()
        root.wantsLayer = true
        root.layer?.backgroundColor = NSColor(calibratedRed: 0.025, green: 0.035, blue: 0.055, alpha: 1).cgColor
        window.contentView = root
        metalView = try FluidMetalView(frame: .zero, device: device, scene: scene, quality: quality)
        metalView.translatesAutoresizingMaskIntoConstraints = false
        root.addSubview(metalView)
        let side = sidebar(device: device)
        side.translatesAutoresizingMaskIntoConstraints = false
        root.addSubview(side)
        NSLayoutConstraint.activate([
            metalView.leadingAnchor.constraint(equalTo: root.leadingAnchor),
            metalView.topAnchor.constraint(equalTo: root.topAnchor),
            metalView.bottomAnchor.constraint(equalTo: root.bottomAnchor),
            metalView.trailingAnchor.constraint(equalTo: side.leadingAnchor),
            side.trailingAnchor.constraint(equalTo: root.trailingAnchor),
            side.topAnchor.constraint(equalTo: root.topAnchor),
            side.bottomAnchor.constraint(equalTo: root.bottomAnchor),
            side.widthAnchor.constraint(equalToConstant: 290)
        ])
        metalView.metricsChanged = { [weak self] value, grid in
            DispatchQueue.main.async {
                self?.metrics.stringValue = String(
                    format: "GPU frame  %.2f ms\nCPU frame  %.2f ms\nSimulation  %.3f s\n\n%@",
                    value.renderMS, value.frameMS, value.simulationTime, grid.label
                )
            }
        }
        window.makeKeyAndOrderFront(nil)
    }

    private func sidebar(device: MTLDevice) -> NSView {
        let panel = NSVisualEffectView()
        panel.material = .sidebar
        panel.blendingMode = .behindWindow
        let title = NSTextField(labelWithString: "FLUID LAB")
        title.font = .systemFont(ofSize: 22, weight: .bold)
        let subtitle = NSTextField(labelWithString: "Native Metal · Apple Silicon")
        subtitle.textColor = .secondaryLabelColor
        let play = NSButton(title: "Pause", target: self, action: #selector(toggle(_:)))
        play.bezelStyle = .rounded
        let reset = NSButton(title: "Reset", target: self, action: #selector(resetSimulation))
        reset.bezelStyle = .rounded
        let qualityLabel = NSTextField(labelWithString: "Quality")
        let popup = NSPopUpButton()
        Quality.allCases.forEach { popup.addItem(withTitle: $0.title) }
        popup.selectItem(withTitle: quality.title)
        popup.target = self
        popup.action = #selector(changeQuality(_:))
        metrics.font = .monospacedDigitSystemFont(ofSize: 13, weight: .medium)
        metrics.maximumNumberOfLines = 0
        status.stringValue = "\(device.name)\nPrivate unified-memory buffers\nFast math · 128-thread 3D groups\nRuntime native MSL compilation"
        status.maximumNumberOfLines = 0
        status.textColor = .secondaryLabelColor
        let hint = NSTextField(wrappingLabelWithString: "Drag to orbit · scroll to zoom\nOpen a browser scene JSON with ⌘O")
        let buttons = NSStackView(views: [play, reset])
        buttons.orientation = .horizontal
        buttons.distribution = .fillEqually
        buttons.spacing = 8
        let stack = NSStackView(views: [title, subtitle, separator(), buttons, qualityLabel, popup, separator(), metrics, separator(), status, hint])
        stack.orientation = .vertical
        stack.spacing = 12
        stack.alignment = .leading
        stack.translatesAutoresizingMaskIntoConstraints = false
        panel.addSubview(stack)
        for view in stack.views { view.widthAnchor.constraint(lessThanOrEqualToConstant: 250).isActive = true }
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: panel.leadingAnchor, constant: 20),
            stack.trailingAnchor.constraint(equalTo: panel.trailingAnchor, constant: -20),
            stack.topAnchor.constraint(equalTo: panel.topAnchor, constant: 54)
        ])
        let open = NSMenuItem(title: "Open Scene…", action: #selector(openScene), keyEquivalent: "o")
        open.target = self
        let menu = NSMenu()
        let appItem = NSMenuItem()
        menu.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Quit Fluid Lab", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        let file = NSMenuItem()
        menu.addItem(file)
        let fileMenu = NSMenu(title: "File")
        fileMenu.addItem(open)
        file.submenu = fileMenu
        NSApp.mainMenu = menu
        return panel
    }

    private func separator() -> NSBox { let box = NSBox(); box.boxType = .separator; return box }

    @objc private func toggle(_ sender: NSButton) {
        guard let solver = metalView.solver else { return }
        solver.isRunning.toggle()
        sender.title = solver.isRunning ? "Pause" : "Run"
    }
    @objc private func resetSimulation() { try? metalView.solver?.reset() }
    @objc private func changeQuality(_ sender: NSPopUpButton) {
        guard let next = Quality.allCases.first(where: { $0.title == sender.titleOfSelectedItem }) else { return }
        quality = next
        rebuild()
    }
    @objc private func openScene() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        if panel.runModal() == .OK, let url = panel.url {
            do { scene = try SceneDescription.load(url: url); rebuild() }
            catch { NSAlert(error: error).runModal() }
        }
    }
    private func rebuild() {
        guard let device = metalView.device, let parent = metalView.superview,
              let side = parent.subviews.first(where: { $0 is NSVisualEffectView }) else { return }
        do {
            let replacement = try FluidMetalView(frame: .zero, device: device, scene: scene, quality: quality)
            replacement.translatesAutoresizingMaskIntoConstraints = false
            replacement.metricsChanged = metalView.metricsChanged
            parent.replaceSubview(metalView, with: replacement)
            NSLayoutConstraint.activate([
                replacement.leadingAnchor.constraint(equalTo: parent.leadingAnchor),
                replacement.topAnchor.constraint(equalTo: parent.topAnchor),
                replacement.bottomAnchor.constraint(equalTo: parent.bottomAnchor),
                replacement.trailingAnchor.constraint(equalTo: side.leadingAnchor)
            ])
            metalView = replacement
        } catch { NSAlert(error: error).runModal() }
    }
}

enum Benchmark {
    static func run() {
        guard let device = MTLCreateSystemDefaultDevice() else { fputs("Metal unavailable\n", stderr); exit(1) }
        do {
            let solver = try MetalFluidSolver(device: device, scene: .browserDefault, quality: .balanced)
            print("FluidMetal smoke PASS · \(device.name) · \(solver.grid.label)")
        } catch { fputs("\(error)\n", stderr); exit(1) }
    }
}

enum SelfTest {
    static func run() {
        let scene = SceneDescription.browserDefault
        let valid = scene.schemaVersion == "1.0.0"
            && scene.sceneId == "interactive-water-box"
            && abs(scene.container.width_m - 1.2) < 1e-6
            && abs(scene.fluid.density_kg_m3 - 998.2) < 1e-3
            && scene.rigidBodies.count == 2
            && Quality.m1Max.targetCells > Quality.ultra.targetCells
        guard valid else { fputs("FluidMetal contract tests FAILED\n", stderr); exit(1) }
        print("FluidMetal contract tests PASS")
    }
}
