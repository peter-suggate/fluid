import Foundation
import simd

struct Vec3: Codable, Sendable {
    var x: Float
    var y: Float
    var z: Float
    var simd: SIMD3<Float> { SIMD3(x, y, z) }
}

struct Quaternion: Codable, Sendable {
    var w: Float
    var x: Float
    var y: Float
    var z: Float
}

struct RigidBodyDescription: Codable, Sendable {
    var id: String
    var name: String
    var shape: String
    var dimensions_m: Vec3
    var density_kg_m3: Float
    var position_m: Vec3
    var orientation: Quaternion
    var linearVelocity_m_s: Vec3
    var angularVelocity_rad_s: Vec3
    var restitution: Float
    var friction: Float
}

struct SceneDescription: Codable, Sendable {
    struct Container: Codable, Sendable {
        var width_m: Float
        var height_m: Float
        var depth_m: Float
        var fillFraction: Float
        var top: String
        var fluidWallMode: String
    }
    struct Fluid: Codable, Sendable {
        var density_kg_m3: Float
        var dynamicViscosity_Pa_s: Float
        var surfaceTension_N_m: Float
        var gravity_m_s2: Vec3
        var initialCondition: String
    }
    struct Resolution: Codable, Sendable { var length_m: Float }
    struct Numerics: Codable, Sendable {
        var fixedDt_s: Float
        var maxDt_s: Float
        var pressureRelativeTolerance: Float
        var pressureMaxIterations: Int
    }

    var schemaVersion: String
    var sceneId: String
    var randomSeed: Int
    var duration_s: Float
    var container: Container
    var fluid: Fluid
    var nominalResolution: Resolution
    var numerics: Numerics
    var rigidBodies: [RigidBodyDescription]

    static func load(url: URL) throws -> SceneDescription {
        let result = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        guard result.schemaVersion == "1.0.0" else { throw SceneError.unsupportedSchema(result.schemaVersion) }
        guard result.container.width_m > 0, result.container.height_m > 0, result.container.depth_m > 0,
              (0...1).contains(result.container.fillFraction), result.fluid.density_kg_m3 > 0 else {
            throw SceneError.invalidValues
        }
        return result
    }

    static let browserDefault: SceneDescription = {
        let url = Bundle.module.url(forResource: "default-scene", withExtension: "json")!
        return try! SceneDescription.load(url: url)
    }()
}

enum SceneError: LocalizedError {
    case unsupportedSchema(String), invalidValues
    var errorDescription: String? {
        switch self {
        case .unsupportedSchema(let version): "Unsupported scene schema \(version)"
        case .invalidValues: "Scene dimensions, density, or fill fraction are invalid"
        }
    }
}

enum Quality: String, CaseIterable {
    case balanced, high, ultra, m1Max
    var targetCells: Int {
        switch self { case .balanced: 110_000; case .high: 500_000; case .ultra: 1_200_000; case .m1Max: 2_000_000 }
    }
    var pressureIterations: Int {
        switch self { case .balanced: 48; case .high: 64; case .ultra: 80; case .m1Max: 96 }
    }
    var title: String { self == .m1Max ? "M1 Max" : rawValue.capitalized }
}
