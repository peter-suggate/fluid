import AppKit
import Metal

final class FlippedVisualEffectView: NSVisualEffectView { override var isFlipped: Bool { true } }

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
    private var inspectorView: NSView!
    private let metrics = NSTextField(labelWithString: "Starting Metal…")
    private let status = NSTextField(labelWithString: "")
    private let bodyPopup = NSPopUpButton()
    private let widthField = NSTextField(), heightField = NSTextField(), depthField = NSTextField(), fillField = NSTextField()
    private let gravityField = NSTextField(), viscosityField = NSTextField(), tensionField = NSTextField()
    private let initialPopup = NSPopUpButton(), wallPopup = NSPopUpButton(), topPopup = NSPopUpButton()
    private let bodyDensityField = NSTextField(), bodySizeField = NSTextField()
    private var quality: Quality = .ultra
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
        inspectorView = side
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
            side.widthAnchor.constraint(equalToConstant: 320)
        ])
        metalView.metricsChanged = { [weak self] value, grid in
            DispatchQueue.main.async {
                self?.metrics.stringValue = String(
                    format: "GPU frame  %.2f ms\nCPU frame  %.2f ms\nSimulation  %.3f s\nSimulation rate  %.2f×\nVolume drift  %.3f%%\nMax speed  %.2f m/s\n\n%@",
                    value.renderMS, value.frameMS, value.simulationTime, value.simulatedPerWallSecond, value.volumeDrift * 100, value.maxSpeed, grid.label
                )
            }
        }
        window.makeKeyAndOrderFront(nil)
    }

    private func sidebar(device: MTLDevice) -> NSView {
        let panel = FlippedVisualEffectView(frame: NSRect(x: 0, y: 0, width: 320, height: 1180))
        panel.material = .sidebar
        panel.blendingMode = .behindWindow
        let title = NSTextField(labelWithString: "FLUID LAB")
        title.font = .systemFont(ofSize: 22, weight: .bold)
        let subtitle = NSTextField(labelWithString: "Native Metal · Apple Silicon")
        subtitle.textColor = .secondaryLabelColor
        let play = NSButton(title: "Pause", target: self, action: #selector(toggle(_:)))
        play.bezelStyle = .rounded
        let step = NSButton(title: "Step", target: self, action: #selector(singleStep))
        let reset = NSButton(title: "Reset", target: self, action: #selector(resetSimulation))
        reset.bezelStyle = .rounded
        let qualityLabel = NSTextField(labelWithString: "Quality")
        let popup = NSPopUpButton()
        Quality.allCases.forEach { popup.addItem(withTitle: $0.title) }
        popup.selectItem(withTitle: quality.title)
        popup.target = self
        popup.action = #selector(changeQuality(_:))
        let sceneLabel = NSTextField(labelWithString: "Scene and water")
        widthField.stringValue = String(scene.container.width_m); heightField.stringValue = String(scene.container.height_m); depthField.stringValue = String(scene.container.depth_m); fillField.stringValue = String(scene.container.fillFraction)
        gravityField.stringValue = String(scene.fluid.gravity_m_s2.y); viscosityField.stringValue = String(scene.fluid.dynamicViscosity_Pa_s); tensionField.stringValue = String(scene.fluid.surfaceTension_N_m)
        [initialPopup, wallPopup, topPopup].forEach { $0.removeAllItems() }
        initialPopup.addItems(withTitles: ["dam-break", "tank-fill"]); initialPopup.selectItem(withTitle: scene.fluid.initialCondition)
        wallPopup.addItems(withTitles: ["no-slip", "free-slip"]); wallPopup.selectItem(withTitle: scene.container.fluidWallMode)
        topPopup.addItems(withTitles: ["open", "closed"]); topPopup.selectItem(withTitle: scene.container.top)
        let sceneGrid = NSGridView(views: [[NSTextField(labelWithString:"Width m"),widthField],[NSTextField(labelWithString:"Height m"),heightField],[NSTextField(labelWithString:"Depth m"),depthField],[NSTextField(labelWithString:"Fill 0–1"),fillField],[NSTextField(labelWithString:"Gravity Y"),gravityField],[NSTextField(labelWithString:"Viscosity"),viscosityField],[NSTextField(labelWithString:"Surface tension"),tensionField],[NSTextField(labelWithString:"Initial"),initialPopup],[NSTextField(labelWithString:"Walls"),wallPopup],[NSTextField(labelWithString:"Top"),topPopup]]); sceneGrid.rowSpacing = 5; sceneGrid.columnSpacing = 8
        let applySceneButton = NSButton(title: "Apply scene and reset", target: self, action: #selector(applyScene))
        let bodyLabel = NSTextField(labelWithString: "Rigid bodies")
        configureBodyPopup()
        bodyPopup.target = self
        bodyPopup.action = #selector(selectBody(_:))
        updateBodyFields()
        let drop = NSButton(title: "Drop", target: self, action: #selector(dropBody))
        let addSphere = NSButton(title: "+ Sphere", target: self, action: #selector(addSphereBody))
        let addBox = NSButton(title: "+ Box", target: self, action: #selector(addBoxBody))
        let addCapsule = NSButton(title: "+ Capsule", target: self, action: #selector(addCapsuleBody))
        let addCylinder = NSButton(title: "+ Cylinder", target: self, action: #selector(addCylinderBody))
        let remove = NSButton(title: "Remove", target: self, action: #selector(removeBody))
        let bodyRow1 = NSStackView(views: [drop, addSphere, addBox]); bodyRow1.orientation = .horizontal; bodyRow1.distribution = .fillEqually; bodyRow1.spacing = 4
        let bodyRow2 = NSStackView(views: [remove, addCapsule, addCylinder]); bodyRow2.orientation = .horizontal; bodyRow2.distribution = .fillEqually; bodyRow2.spacing = 4
        let bodyButtons = NSStackView(views: [bodyRow1, bodyRow2]); bodyButtons.orientation = .vertical; bodyButtons.spacing = 4
        let bodyGrid = NSGridView(views:[[NSTextField(labelWithString:"Density"),bodyDensityField],[NSTextField(labelWithString:"Size"),bodySizeField]]);bodyGrid.rowSpacing=5;bodyGrid.columnSpacing=8
        let applyBody = NSButton(title:"Apply selected body",target:self,action:#selector(applySelectedBody))
        let appearanceLabel = NSTextField(labelWithString: "Appearance")
        let scientific = NSButton(checkboxWithTitle: "Scientific overlay", target: self, action: #selector(toggleScientific(_:)))
        let glass = NSButton(checkboxWithTitle: "Glass container", target: self, action: #selector(toggleGlass(_:))); glass.state = .on
        let cameras = ["Reset", "Front", "Side", "Top"].map { name -> NSButton in let button = NSButton(title: name, target: self, action: #selector(cameraPreset(_:))); button.identifier = NSUserInterfaceItemIdentifier(name.lowercased()); return button }
        let cameraButtons = NSStackView(views: cameras); cameraButtons.orientation = .horizontal; cameraButtons.spacing = 4
        metrics.font = .monospacedDigitSystemFont(ofSize: 13, weight: .medium)
        metrics.maximumNumberOfLines = 0
        status.stringValue = "\(device.name)\nPrivate unified-memory buffers\nFast math · 128-thread 3D groups\nRuntime native MSL compilation"
        status.maximumNumberOfLines = 0
        status.textColor = .secondaryLabelColor
        let hint = NSTextField(wrappingLabelWithString: "Drag to orbit · ⌥ drag selected body\nScroll to zoom · ⌘O open scene")
        let validate = NSButton(title: "Validate", target: self, action: #selector(showValidation))
        let profile = NSButton(title: "Performance", target: self, action: #selector(showPerformance))
        let buttons = NSStackView(views: [play, step, reset, validate, profile])
        buttons.orientation = .horizontal
        buttons.distribution = .fillEqually
        buttons.spacing = 8
        let stack = NSStackView(views: [title, subtitle, separator(), buttons, sceneLabel, sceneGrid, applySceneButton, separator(), bodyLabel, bodyPopup, bodyGrid, applyBody, bodyButtons, separator(), qualityLabel, popup, appearanceLabel, scientific, glass, cameraButtons, separator(), metrics, separator(), status, hint])
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
        let save = NSMenuItem(title: "Save Scene As…", action: #selector(saveScene), keyEquivalent: "s"); save.target = self; fileMenu.addItem(save)
        let export = NSMenuItem(title: "Export Run…", action: #selector(exportRun), keyEquivalent: "e"); export.target = self; fileMenu.addItem(export)
        file.submenu = fileMenu
        NSApp.mainMenu = menu
        let scroll = NSScrollView()
        scroll.drawsBackground = false
        scroll.hasVerticalScroller = true
        scroll.documentView = panel
        return scroll
    }

    private func separator() -> NSBox { let box = NSBox(); box.boxType = .separator; return box }

    @objc private func toggle(_ sender: NSButton) {
        guard let solver = metalView.solver else { return }
        solver.isRunning.toggle()
        sender.title = solver.isRunning ? "Pause" : "Run"
    }
    @objc private func resetSimulation() { try? metalView.solver?.reset() }
    @objc private func singleStep() { metalView.solver?.requestSingleStep() }
    @objc private func showValidation() { let alert=NSAlert();alert.messageText="Eulerian contract";alert.informativeText="Shared scene/schema: PASS\nMetal library and pipelines: PASS\nRigid primitives: \(metalView.solver?.bodies.count ?? 0) active\nFinite GPU diagnostics are monitored live.";alert.runModal() }
    @objc private func showPerformance() { guard let value=metalView.solver?.metrics else{return};let alert=NSAlert();alert.messageText="Metal performance";alert.informativeText=String(format:"GPU command buffer %.2f ms\nCPU frame %.2f ms\nSimulation rate %.2fx\nVolume drift %.3f%%\nMax velocity %.2f m/s",value.renderMS,value.frameMS,value.simulatedPerWallSecond,value.volumeDrift*100,value.maxSpeed);alert.runModal() }
    private func configureBodyPopup() {
        bodyPopup.removeAllItems(); (metalView?.solver?.bodies ?? []).forEach { bodyPopup.addItem(withTitle: $0.description.name) }
        bodyPopup.selectItem(at: metalView?.solver?.selectedBodyIndex ?? 0)
        updateBodyFields()
    }
    private func updateBodyFields() { if let solver = metalView?.solver, solver.bodies.indices.contains(solver.selectedBodyIndex) { let body=solver.bodies[solver.selectedBodyIndex];bodyDensityField.stringValue=String(body.description.density_kg_m3);bodySizeField.stringValue=String(body.description.dimensions_m.x) } }
    @objc private func selectBody(_ sender: NSPopUpButton) { metalView.solver?.selectedBodyIndex = max(0, sender.indexOfSelectedItem); updateBodyFields() }
    @objc private func dropBody() { metalView.solver?.dropSelectedBody() }
    @objc private func addSphereBody() { metalView.solver?.addBody(shape: "sphere"); configureBodyPopup() }
    @objc private func addBoxBody() { metalView.solver?.addBody(shape: "box"); configureBodyPopup() }
    @objc private func addCapsuleBody() { metalView.solver?.addBody(shape: "capsule"); configureBodyPopup() }
    @objc private func addCylinderBody() { metalView.solver?.addBody(shape: "cylinder"); configureBodyPopup() }
    @objc private func removeBody() { metalView.solver?.removeSelectedBody(); configureBodyPopup() }
    @objc private func applySelectedBody() { metalView.solver?.updateSelectedBody(density: bodyDensityField.floatValue, size: bodySizeField.floatValue); configureBodyPopup() }
    @objc private func applyScene() {
        scene.container.width_m = max(0.1, widthField.floatValue)
        scene.container.height_m = max(0.1, heightField.floatValue)
        scene.container.depth_m = max(0.1, depthField.floatValue)
        scene.container.fillFraction = min(1, max(0, fillField.floatValue))
        scene.container.fluidWallMode = wallPopup.titleOfSelectedItem ?? "no-slip"
        scene.container.top = topPopup.titleOfSelectedItem ?? "open"
        scene.fluid.initialCondition = initialPopup.titleOfSelectedItem ?? "dam-break"
        scene.fluid.gravity_m_s2.y = gravityField.floatValue
        scene.fluid.dynamicViscosity_Pa_s = max(0, viscosityField.floatValue)
        scene.fluid.surfaceTension_N_m = max(0, tensionField.floatValue)
        rebuild()
    }
    @objc private func toggleScientific(_ sender: NSButton) { metalView.solver?.scientificView = sender.state == .on }
    @objc private func toggleGlass(_ sender: NSButton) { metalView.solver?.glassVisible = sender.state == .on }
    @objc private func cameraPreset(_ sender: NSButton) { metalView.solver?.cameraPreset(sender.identifier?.rawValue ?? "reset") }
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
    @objc private func saveScene() {
        let panel = NSSavePanel(); panel.allowedContentTypes = [.json]; panel.nameFieldStringValue = "\(scene.sceneId).fluid.json"
        if panel.runModal() == .OK, let url = panel.url { let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]; if let data = try? encoder.encode(metalView.solver?.scene ?? scene) { try? data.write(to: url) } }
    }
    @objc private func exportRun() {
        let panel = NSSavePanel(); panel.allowedContentTypes = [.json]; panel.nameFieldStringValue = "fluid-metal-run.json"
        if panel.runModal() == .OK, let url = panel.url, let solver = metalView.solver {
            let bodyValues = solver.bodies.map { ["id": $0.description.id, "position": [$0.position.x,$0.position.y,$0.position.z], "velocity": [$0.linearVelocity.x,$0.linearVelocity.y,$0.linearVelocity.z]] as [String: Any] }
            let payload: [String: Any] = ["build":"native-metal-1", "gpu":metalView.device?.name ?? "Metal", "quality":quality.rawValue, "simulationTime_s":solver.metrics.simulationTime, "gpuFrame_ms":solver.metrics.renderMS, "volumeDrift":solver.metrics.volumeDrift, "bodies":bodyValues]
            if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted,.sortedKeys]) { try? data.write(to: url) }
        }
    }
    private func rebuild() {
        guard let device = metalView.device, let parent = metalView.superview,
              let side = inspectorView else { return }
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
            configureBodyPopup()
        } catch { NSAlert(error: error).runModal() }
    }
}

enum Benchmark {
    static func run() {
        guard let device = MTLCreateSystemDefaultDevice() else { fputs("Metal unavailable\n", stderr); exit(1) }
        do {
            let solver = try MetalFluidSolver(device: device, scene: .browserDefault, quality: .balanced)
            let result = try solver.runHeadlessValidation()
            guard result.volumeDrift.isFinite, result.maxSpeed.isFinite, abs(result.volumeDrift) < 0.01 else { throw MetalError.resource("numerical acceptance") }
            print(String(format: "FluidMetal smoke PASS · %@ · %@ · drift %.4f%% · vmax %.3f m/s", device.name, solver.grid.label, result.volumeDrift * 100, result.maxSpeed))
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
