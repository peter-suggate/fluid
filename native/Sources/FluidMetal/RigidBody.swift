import Foundation
import simd

struct RigidBodyState: Sendable {
    let description: RigidBodyDescription
    var position: SIMD3<Float>
    var orientation: simd_quatf
    var linearVelocity: SIMD3<Float>
    var angularVelocity: SIMD3<Float>
    var force: SIMD3<Float> = .zero
    var torque: SIMD3<Float> = .zero
    var displacedVolume: Float = 0
    let mass: Float
    let inverseMass: Float
    let inverseInertia: SIMD3<Float>

    init(_ value: RigidBodyDescription) {
        description = value
        position = value.position_m.simd
        orientation = simd_normalize(simd_quatf(ix: value.orientation.x, iy: value.orientation.y, iz: value.orientation.z, r: value.orientation.w))
        linearVelocity = value.linearVelocity_m_s.simd
        angularVelocity = value.angularVelocity_rad_s.simd
        let d = value.dimensions_m.simd
        let volume: Float
        switch value.shape {
        case "sphere": volume = 4 / 3 * .pi * d.x * d.x * d.x
        case "box": volume = d.x * d.y * d.z
        case "cylinder": volume = .pi * d.x * d.x * d.y
        default: volume = .pi * d.x * d.x * d.y + 4 / 3 * .pi * d.x * d.x * d.x
        }
        mass = max(volume * value.density_kg_m3, 1e-6)
        inverseMass = 1 / mass
        let inertia: SIMD3<Float>
        if value.shape == "sphere" {
            inertia = SIMD3(repeating: 0.4 * mass * d.x * d.x)
        } else if value.shape == "box" {
            inertia = SIMD3(mass * (d.y*d.y+d.z*d.z), mass * (d.x*d.x+d.z*d.z), mass * (d.x*d.x+d.y*d.y)) / 12
        } else {
            inertia = SIMD3(mass * (3*d.x*d.x+d.y*d.y)/12, 0.5*mass*d.x*d.x, mass * (3*d.x*d.x+d.y*d.y)/12)
        }
        inverseInertia = 1 / max(inertia, SIMD3(repeating: 1e-8))
    }

    var supportRadius: Float {
        let d = description.dimensions_m.simd
        switch description.shape {
        case "sphere": return d.x
        case "box": return 0.5 * simd_length(d)
        case "cylinder": return hypot(d.x, d.y * 0.5)
        default: return d.x + d.y * 0.5
        }
    }

    mutating func applyGPUImpulse(linear: SIMD3<Float>, angular: SIMD3<Float>, displaced: Float) {
        linearVelocity += linear * inverseMass
        let localTorque = orientation.inverse.act(angular)
        angularVelocity += orientation.act(localTorque * inverseInertia)
        displacedVolume = displaced
    }

    mutating func step(dt: Float, scene: SceneDescription) {
        force = scene.fluid.gravity_m_s2.simd * mass
        linearVelocity += force * inverseMass * dt
        position += linearVelocity * dt
        let spin = simd_length(angularVelocity)
        if spin > 1e-7 { orientation = simd_normalize(simd_quatf(angle: spin * dt, axis: angularVelocity / spin) * orientation) }

        let radius = supportRadius
        func bounce(_ normal: SIMD3<Float>, penetration: Float) {
            guard penetration > 0 else { return }
            position += normal * penetration
            let speed = simd_dot(linearVelocity, normal)
            if speed < 0 { linearVelocity -= normal * ((1 + description.restitution) * speed) }
            let tangent = linearVelocity - normal * simd_dot(linearVelocity, normal)
            linearVelocity -= tangent * min(1, description.friction * dt * 30)
        }
        bounce(SIMD3(0,1,0), penetration: radius-position.y)
        bounce(SIMD3(1,0,0), penetration: -scene.container.width_m/2-(position.x-radius))
        bounce(SIMD3(-1,0,0), penetration: position.x+radius-scene.container.width_m/2)
        bounce(SIMD3(0,0,1), penetration: -scene.container.depth_m/2-(position.z-radius))
        bounce(SIMD3(0,0,-1), penetration: position.z+radius-scene.container.depth_m/2)
        if scene.container.top == "closed" { bounce(SIMD3(0,-1,0), penetration: position.y+radius-scene.container.height_m) }
    }
}

struct BodyGPU {
    var positionShape: SIMD4<Float>
    var dimensions: SIMD4<Float>
    var orientation: SIMD4<Float>
    var linearVelocity: SIMD4<Float>
    var angularVelocity: SIMD4<Float>
    var inverseMassRestitutionFriction: SIMD4<Float>
    var inverseInertia: SIMD4<Float>
}

extension RigidBodyState {
    var gpuValue: BodyGPU {
        let shape: Float = ["sphere":0, "box":1, "capsule":2, "cylinder":3][description.shape] ?? 0
        let q = orientation.vector
        return BodyGPU(
            positionShape: SIMD4(position, shape), dimensions: SIMD4(description.dimensions_m.simd, 0),
            orientation: SIMD4(q.w, q.x, q.y, q.z), linearVelocity: SIMD4(linearVelocity, 0), angularVelocity: SIMD4(angularVelocity, 0),
            inverseMassRestitutionFriction: SIMD4(inverseMass, description.restitution, description.friction, mass), inverseInertia: SIMD4(inverseInertia, 0)
        )
    }

    mutating func synchronize(from value: BodyGPU) {
        position = SIMD3(value.positionShape.x, value.positionShape.y, value.positionShape.z)
        orientation = simd_normalize(simd_quatf(ix: value.orientation.y, iy: value.orientation.z, iz: value.orientation.w, r: value.orientation.x))
        linearVelocity = SIMD3(value.linearVelocity.x, value.linearVelocity.y, value.linearVelocity.z)
        angularVelocity = SIMD3(value.angularVelocity.x, value.angularVelocity.y, value.angularVelocity.z)
    }
}
