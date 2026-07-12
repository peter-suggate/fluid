// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "FluidMetal",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "FluidMetal", targets: ["FluidMetal"])],
    targets: [
        .executableTarget(
            name: "FluidMetal",
            resources: [.process("Resources")],
            swiftSettings: [.unsafeFlags(["-Ounchecked"], .when(configuration: .release))]
        )
    ]
)
