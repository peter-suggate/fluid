//! Strict-f32 packed rigid integration and delayed fluid reaction.
use crate::scene_model::{
    Quaternion, RigidBodyDescription, RigidMotion, RigidShape, SceneModelError, SliceFrame, Vec3,
};
use crate::types::{Fields, Graph};
use serde::{Deserialize, Serialize};

type V = [f32; 3];
#[inline]
fn v(x: f32, y: f32, z: f32) -> V {
    [x, y, z]
}
#[inline]
fn add(a: V, b: V) -> V {
    v(a[0] + b[0], a[1] + b[1], a[2] + b[2])
}
#[inline]
fn sub(a: V, b: V) -> V {
    v(a[0] - b[0], a[1] - b[1], a[2] - b[2])
}
#[inline]
fn scale(a: V, s: f32) -> V {
    v(a[0] * s, a[1] * s, a[2] * s)
}
#[inline]
fn dot(a: V, b: V) -> f32 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}
#[inline]
fn cross(a: V, b: V) -> V {
    v(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
}
#[inline]
fn length(a: V) -> f32 {
    dot(a, a).sqrt()
}
fn qrotate(q: [f32; 4], p: V) -> V {
    let u = v(q[1], q[2], q[3]);
    let uv = cross(u, p);
    add(p, scale(add(scale(uv, q[0]), cross(u, uv)), 2.0))
}
fn qmul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let av = v(a[1], a[2], a[3]);
    let bv = v(b[1], b[2], b[3]);
    let q = add(add(scale(bv, a[0]), scale(av, b[0])), cross(av, bv));
    [a[0] * b[0] - dot(av, bv), q[0], q[1], q[2]]
}

pub fn primitive_volume(shape: RigidShape, d: Vec3) -> Result<f64, SceneModelError> {
    Ok(match shape {
        RigidShape::Sphere => (4.0 / 3.0) * std::f64::consts::PI * d.x.powi(3),
        RigidShape::Box => d.x * d.y * d.z,
        RigidShape::Capsule => {
            std::f64::consts::PI * d.x.powi(2) * d.y
                + (4.0 / 3.0) * std::f64::consts::PI * d.x.powi(3)
        }
        RigidShape::Cylinder => std::f64::consts::PI * d.x.powi(2) * d.y,
        RigidShape::Cup => {
            let t = d.z.clamp(1e-4, (0.95 * d.x).min(0.95 * d.y));
            std::f64::consts::PI * (d.x.powi(2) * d.y - (d.x - t).powi(2) * (d.y - t))
        }
    })
}
fn inertia(description: &RigidBodyDescription) -> Result<Vec3, SceneModelError> {
    let d = description.dimensions_m;
    let density = description.density_kg_m3;
    let mass = density * primitive_volume(description.shape, d)?;
    Ok(match description.shape {
        RigidShape::Sphere => {
            let q = (2.0 / 5.0) * mass * d.x.powi(2);
            Vec3 { x: q, y: q, z: q }
        }
        RigidShape::Box => Vec3 {
            x: mass * (d.y.powi(2) + d.z.powi(2)) / 12.0,
            y: mass * (d.x.powi(2) + d.z.powi(2)) / 12.0,
            z: mass * (d.x.powi(2) + d.y.powi(2)) / 12.0,
        },
        RigidShape::Capsule => {
            let r = d.x;
            let l = d.y;
            let r2 = r.powi(2);
            let l2 = l.powi(2);
            let cm = density * std::f64::consts::PI * r2 * l;
            let sm = density * (4.0 / 3.0) * std::f64::consts::PI * r.powi(3);
            let axial = 0.5 * cm * r2 + (2.0 / 5.0) * sm * r2;
            let caps = sm * ((83.0 / 320.0) * r2 + (l / 2.0 + 3.0 * r / 8.0).powi(2));
            let transverse = cm * (3.0 * r2 + l2) / 12.0 + caps;
            Vec3 {
                x: transverse,
                y: axial,
                z: transverse,
            }
        }
        RigidShape::Cylinder => Vec3 {
            x: mass * (3.0 * d.x.powi(2) + d.y.powi(2)) / 12.0,
            y: 0.5 * mass * d.x.powi(2),
            z: mass * (3.0 * d.x.powi(2) + d.y.powi(2)) / 12.0,
        },
        RigidShape::Cup => {
            let t = d.z.clamp(1e-4, (0.95 * d.x).min(0.95 * d.y));
            let ro = d.x;
            let ri = d.x - t;
            let ro2 = ro.powi(2);
            let ri2 = ri.powi(2);
            let bm = density * std::f64::consts::PI * ro2 * t;
            let bc = -0.5 * d.y + 0.5 * t;
            let wl = d.y - t;
            let wm = density * std::f64::consts::PI * (ro2 - ri2) * wl;
            let wc = 0.5 * t;
            let axial = 0.5 * bm * ro2 + 0.5 * wm * (ro2 + ri2);
            let transverse = bm * (3.0 * ro2 + t.powi(2)) / 12.0
                + bm * bc.powi(2)
                + wm * (3.0 * (ro2 + ri2) + wl.powi(2)) / 12.0
                + wm * wc.powi(2);
            Vec3 {
                x: transverse,
                y: axial,
                z: transverse,
            }
        }
    })
}
pub(crate) fn bounding_radius(d: &RigidBodyDescription) -> Result<f32, SceneModelError> {
    Ok(match d.shape {
        RigidShape::Sphere => d.dimensions_m.x,
        RigidShape::Box => {
            0.5 * (d.dimensions_m.x * d.dimensions_m.x
                + d.dimensions_m.y * d.dimensions_m.y
                + d.dimensions_m.z * d.dimensions_m.z)
                .sqrt()
        }
        RigidShape::Capsule => d.dimensions_m.x + 0.5 * d.dimensions_m.y,
        RigidShape::Cylinder | RigidShape::Cup => (d.dimensions_m.x * d.dimensions_m.x
            + 0.25 * d.dimensions_m.y * d.dimensions_m.y)
            .sqrt(),
    } as f32)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RigidBodyState {
    pub description: RigidBodyDescription,
    #[serde(rename = "position_m")]
    pub position_m: Vec3,
    pub orientation: Quaternion,
    #[serde(rename = "linearVelocity_m_s")]
    pub linear_velocity_m_s: Vec3,
    #[serde(rename = "angularVelocity_rad_s")]
    pub angular_velocity_rad_s: Vec3,
    #[serde(rename = "angularMomentum_kg_m2_s")]
    pub angular_momentum_kg_m2_s: Vec3,
    #[serde(rename = "mass_kg")]
    pub mass_kg: f64,
    /// A planar host extrudes area through its slice thickness. Ordinary 3D
    /// states leave this absent and retain the original primitive arithmetic.
    #[serde(
        default,
        rename = "dimensionalVolume_m3",
        skip_serializing_if = "Option::is_none"
    )]
    pub dimensional_volume_m3: Option<f64>,
    #[serde(rename = "inverseMass_kg")]
    pub inverse_mass_kg: f64,
    #[serde(rename = "inverseInertiaBody_kg_m2")]
    pub inverse_inertia_body_kg_m2: Vec3,
    #[serde(default)]
    pub held: bool,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FluidExchange {
    pub lanes: [i32; 12],
}

pub(crate) fn set_rigid_body_kinematics(
    body: &mut RigidBodyState,
    position_m: Vec3,
    orientation: Quaternion,
    linear_velocity_m_s: Vec3,
    angular_velocity_rad_s: Vec3,
    held: bool,
) {
    let mut q = [
        orientation.w as f32,
        orientation.x as f32,
        orientation.y as f32,
        orientation.z as f32,
    ];
    let norm = ((q[0] * q[0] + q[1] * q[1]) + q[2] * q[2] + q[3] * q[3]).sqrt();
    for lane in &mut q {
        *lane /= norm;
    }
    let omega = [
        angular_velocity_rad_s.x as f32,
        angular_velocity_rad_s.y as f32,
        angular_velocity_rad_s.z as f32,
    ];
    let local_omega = qrotate([q[0], -q[1], -q[2], -q[3]], omega);
    let inverse = body.inverse_inertia_body_kg_m2.array().map(|v| v as f32);
    let local_momentum = std::array::from_fn(|axis| {
        if inverse[axis] > 0.0 {
            local_omega[axis] / inverse[axis]
        } else {
            0.0
        }
    });
    let momentum = qrotate(q, local_momentum);
    body.position_m = Vec3 {
        x: position_m.x as f32 as f64,
        y: position_m.y as f32 as f64,
        z: position_m.z as f32 as f64,
    };
    body.orientation = Quaternion {
        w: q[0] as f64,
        x: q[1] as f64,
        y: q[2] as f64,
        z: q[3] as f64,
    };
    body.linear_velocity_m_s = Vec3 {
        x: linear_velocity_m_s.x as f32 as f64,
        y: linear_velocity_m_s.y as f32 as f64,
        z: linear_velocity_m_s.z as f32 as f64,
    };
    body.angular_velocity_rad_s = Vec3 {
        x: omega[0] as f64,
        y: omega[1] as f64,
        z: omega[2] as f64,
    };
    body.angular_momentum_kg_m2_s = Vec3 {
        x: momentum[0] as f64,
        y: momentum[1] as f64,
        z: momentum[2] as f64,
    };
    body.held = held;
}

pub fn initialize_body(
    description: RigidBodyDescription,
) -> Result<RigidBodyState, SceneModelError> {
    let volume = primitive_volume(description.shape, description.dimensions_m)?;
    let mass = description.density_kg_m3 * volume;
    let inertia = inertia(&description)?;
    let fixed = description.motion == RigidMotion::Static;
    let mut q = description.orientation;
    let norm = (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z).sqrt();
    if norm > 0.0 {
        q = Quaternion {
            w: q.w / norm,
            x: q.x / norm,
            y: q.y / norm,
            z: q.z / norm,
        }
    } else {
        q = Quaternion::default()
    }
    let omega = description.angular_velocity_rad_s;
    Ok(RigidBodyState {
        position_m: description.position_m,
        orientation: q,
        linear_velocity_m_s: description.linear_velocity_m_s,
        angular_velocity_rad_s: omega,
        angular_momentum_kg_m2_s: Vec3 {
            x: inertia.x * omega.x,
            y: inertia.y * omega.y,
            z: inertia.z * omega.z,
        },
        mass_kg: mass,
        dimensional_volume_m3: None,
        inverse_mass_kg: if fixed { 0.0 } else { 1.0 / mass },
        inverse_inertia_body_kg_m2: Vec3 {
            x: if fixed { 0.0 } else { 1.0 / inertia.x },
            y: if fixed { 0.0 } else { 1.0 / inertia.y },
            z: if fixed { 0.0 } else { 1.0 / inertia.z },
        },
        held: false,
        description,
    })
}
fn inverse_inertia(body: &[f32; 32], value: V, density: f32) -> V {
    let q = [body[8], body[9], body[10], body[11]];
    let local = qrotate([q[0], -q[1], -q[2], -q[3]], value);
    qrotate(
        q,
        v(
            local[0] * body[21] / density.max(1e-9),
            local[1] * body[22] / density.max(1e-9),
            local[2] * body[23] / density.max(1e-9),
        ),
    )
}
/// Planar mass and Z inertia, with the same state and integration path as 3D.
pub fn initialize_body_2d(
    description: RigidBodyDescription,
    depth: f64,
) -> Result<RigidBodyState, SceneModelError> {
    let mut body = initialize_body(description)?;
    let d = body.description.dimensions_m;
    let rho = body.description.density_kg_m3 * depth;
    let rectangle = |w: f64, h: f64| {
        let mass = rho * w * h;
        (mass, mass * (w * w + h * h) / 12.0)
    };
    let (mass, inertia) = match body.description.shape {
        RigidShape::Sphere => {
            let mass = rho * std::f64::consts::PI * d.x * d.x;
            (mass, 0.5 * mass * d.x * d.x)
        }
        RigidShape::Box => rectangle(d.x, d.y),
        RigidShape::Cylinder => rectangle(2.0 * d.x, d.y),
        RigidShape::Capsule => {
            let (rm, ri) = rectangle(2.0 * d.x, d.y);
            let dm = rho * std::f64::consts::PI * d.x * d.x;
            (
                rm + dm,
                ri + dm * (0.5 * d.x * d.x + 0.25 * d.y * d.y)
                    + 4.0 / 3.0 * rho * d.y * d.x.powi(3),
            )
        }
        RigidShape::Cup => {
            let t = d.z.clamp(1e-4, 0.95 * d.x.min(d.y));
            let (outer, oi) = rectangle(2.0 * d.x, d.y);
            let (inner, ii) = rectangle(2.0 * (d.x - t), d.y - t);
            (outer - inner, oi - ii - inner * 0.25 * t * t)
        }
    };
    body.mass_kg = mass;
    body.dimensional_volume_m3 = Some(mass / body.description.density_kg_m3);
    let fixed = body.description.motion == RigidMotion::Static;
    body.inverse_mass_kg = if fixed { 0.0 } else { 1.0 / mass };
    body.inverse_inertia_body_kg_m2 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: if fixed { 0.0 } else { 1.0 / inertia },
    };
    body.angular_momentum_kg_m2_s = Vec3 {
        x: 0.0,
        y: 0.0,
        z: inertia * body.angular_velocity_rad_s.z,
    };
    Ok(body)
}

fn pack(body: &RigidBodyState, density: f32) -> Result<[f32; 32], SceneModelError> {
    let mut r = [0.0; 32];
    let d = body.description.dimensions_m;
    let shape = match body.description.shape {
        RigidShape::Sphere => 0.0,
        RigidShape::Box => 1.0,
        RigidShape::Capsule => 2.0,
        RigidShape::Cylinder => 3.0,
        RigidShape::Cup => 4.0,
    };
    let radius = bounding_radius(&body.description)?;
    let inv = if body.held { 0.0 } else { body.inverse_mass_kg };
    r[..8].copy_from_slice(&[
        body.position_m.x as f32,
        body.position_m.y as f32,
        body.position_m.z as f32,
        shape,
        d.x as f32,
        d.y as f32,
        d.z as f32,
        radius,
    ]);
    r[8..16].copy_from_slice(&[
        body.orientation.w as f32,
        body.orientation.x as f32,
        body.orientation.y as f32,
        body.orientation.z as f32,
        body.linear_velocity_m_s.x as f32,
        body.linear_velocity_m_s.y as f32,
        body.linear_velocity_m_s.z as f32,
        (inv * density as f64) as f32,
    ]);
    r[16..24].copy_from_slice(&[
        body.angular_velocity_rad_s.x as f32,
        body.angular_velocity_rad_s.y as f32,
        body.angular_velocity_rad_s.z as f32,
        body.description.density_kg_m3 as f32,
        (inv * density as f64) as f32,
        ((if body.held {
            0.0
        } else {
            body.inverse_inertia_body_kg_m2.x
        }) * density as f64) as f32,
        ((if body.held {
            0.0
        } else {
            body.inverse_inertia_body_kg_m2.y
        }) * density as f64) as f32,
        ((if body.held {
            0.0
        } else {
            body.inverse_inertia_body_kg_m2.z
        }) * density as f64) as f32,
    ]);
    r[24..31].copy_from_slice(&[
        body.angular_momentum_kg_m2_s.x as f32,
        body.angular_momentum_kg_m2_s.y as f32,
        body.angular_momentum_kg_m2_s.z as f32,
        body.description.restitution as f32,
        body.description.friction as f32,
        0.0,
        if body.description.motion == RigidMotion::Dynamic {
            1.0
        } else {
            0.0
        },
    ]);
    Ok(r)
}

fn angular_term(body: &[f32; 32], arm: V, direction: V, density: f32) -> f32 {
    dot(
        cross(inverse_inertia(body, cross(arm, direction), density), arm),
        direction,
    )
}
fn velocity_at_packed(body: &[f32; 32], arm: V) -> V {
    add(
        v(body[12], body[13], body[14]),
        cross(v(body[16], body[17], body[18]), arm),
    )
}
fn apply_impulse(body: &mut [f32; 32], impulse: V, arm: V, density: f32) {
    let inv = body[20] / density.max(1e-9);
    let next = add(v(body[12], body[13], body[14]), scale(impulse, inv));
    body[12..15].copy_from_slice(&next);
    let momentum = add(v(body[24], body[25], body[26]), cross(arm, impulse));
    body[24..27].copy_from_slice(&momentum);
    let omega = inverse_inertia(body, momentum, density);
    body[16..19].copy_from_slice(&omega)
}
fn solve_body_pair(a: &mut [f32; 32], b: &mut [f32; 32], density: f32) {
    let ia = a[20] / density;
    let ib = b[20] / density;
    let it = ia + ib;
    if it <= 0.0 {
        return;
    }
    let delta = sub(v(b[0], b[1], b[2]), v(a[0], a[1], a[2]));
    let distance = length(delta);
    let normal = if distance > 1e-8 {
        scale(delta, 1.0 / distance)
    } else {
        v(1.0, 0.0, 0.0)
    };
    let penetration = a[7] + b[7] - distance;
    if penetration <= 0.0 {
        return;
    }
    let pa = sub(v(a[0], a[1], a[2]), scale(normal, penetration * ia / it));
    let pb = add(v(b[0], b[1], b[2]), scale(normal, penetration * ib / it));
    a[..3].copy_from_slice(&pa);
    b[..3].copy_from_slice(&pb);
    let arm_a = scale(normal, a[7]);
    let arm_b = scale(normal, -b[7]);
    let mut relative = sub(velocity_at_packed(b, arm_b), velocity_at_packed(a, arm_a));
    let normal_speed = dot(relative, normal);
    if normal_speed >= 0.0 {
        return;
    }
    let restitution = if -normal_speed > 0.5 {
        a[27].min(b[27])
    } else {
        0.0
    };
    let denominator =
        (it + angular_term(a, arm_a, normal, density) + angular_term(b, arm_b, normal, density))
            .max(1e-9);
    let magnitude = -(1.0 + restitution) * normal_speed / denominator;
    apply_impulse(a, scale(normal, -magnitude), arm_a, density);
    apply_impulse(b, scale(normal, magnitude), arm_b, density);
    relative = sub(velocity_at_packed(b, arm_b), velocity_at_packed(a, arm_a));
    let tangent_velocity = sub(relative, scale(normal, dot(relative, normal)));
    let tangent_speed = length(tangent_velocity);
    if tangent_speed <= 1e-8 {
        return;
    }
    let tangent = scale(tangent_velocity, 1.0 / tangent_speed);
    let td =
        (it + angular_term(a, arm_a, tangent, density) + angular_term(b, arm_b, tangent, density))
            .max(1e-9);
    let friction = (a[28] * b[28]).max(0.0).sqrt();
    let tm = (-tangent_speed / td).clamp(-friction * magnitude, friction * magnitude);
    apply_impulse(a, scale(tangent, -tm), arm_a, density);
    apply_impulse(b, scale(tangent, tm), arm_b, density)
}
fn apply_packed(state: &mut RigidBodyState, body: &[f32; 32]) {
    state.position_m = Vec3 {
        x: body[0] as f64,
        y: body[1] as f64,
        z: body[2] as f64,
    };
    state.orientation = Quaternion {
        w: body[8] as f64,
        x: body[9] as f64,
        y: body[10] as f64,
        z: body[11] as f64,
    };
    state.linear_velocity_m_s = Vec3 {
        x: body[12] as f64,
        y: body[13] as f64,
        z: body[14] as f64,
    };
    state.angular_velocity_rad_s = Vec3 {
        x: body[16] as f64,
        y: body[17] as f64,
        z: body[18] as f64,
    };
    state.angular_momentum_kg_m2_s = Vec3 {
        x: body[24] as f64,
        y: body[25] as f64,
        z: body[26] as f64,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticContactVoxel {
    pub coordinate: [i32; 3],
    pub fraction: f32,
    pub minimum_m: [f32; 3],
    pub maximum_m: [f32; 3],
}
pub trait StaticContactProvider {
    fn cell_size_m(&self) -> [f32; 3];
    fn visit_candidates(
        &self,
        centre_m: [f32; 3],
        radius_m: f32,
        visitor: &mut dyn FnMut(StaticContactVoxel),
    );
    fn signed_distance_cells(&self, coordinate: [i32; 3]) -> f32;
}
fn normalize(a: V, fallback: V) -> V {
    let n = length(a);
    if n > 1e-8 {
        scale(a, 1.0 / n)
    } else {
        fallback
    }
}
fn support_radius_packed(body: &[f32; 32], direction: V) -> f32 {
    let n = normalize(direction, v(1.0, 0.0, 0.0));
    let q = [body[8], body[9], body[10], body[11]];
    let l = qrotate([q[0], -q[1], -q[2], -q[3]], n);
    let (x, y, z) = (l[0] as f64, l[1] as f64, l[2] as f64);
    let (dx, dy, dz) = (body[4] as f64, body[5] as f64, body[6] as f64);
    match body[3].round() as i32 {
        0 => body[4],
        1 => (0.5 * (x.abs() * dx + y.abs() * dy + z.abs() * dz)) as f32,
        2 => (dx + 0.5 * dy * y.abs()) as f32,
        3 | 4 => (dx * x.hypot(z) + 0.5 * dy * y.abs()) as f32,
        _ => 0.0,
    }
}
fn resolve_static_contact(
    body: &mut [f32; 32],
    normal: V,
    penetration: f32,
    radius: f32,
    density: f32,
) {
    let inv = body[20] / density.max(1e-9);
    if !(inv > 0.0 && penetration > 0.0) {
        return;
    }
    let p = add(
        v(body[0], body[1], body[2]),
        scale(normal, penetration + 1e-7),
    );
    body[..3].copy_from_slice(&p);
    let arm = scale(normal, -radius);
    let mut relative = velocity_at_packed(body, arm);
    let normal_speed = dot(relative, normal);
    if normal_speed >= 0.0 {
        return;
    }
    let restitution = if -normal_speed > 0.5 { body[27] } else { 0.0 };
    let denominator = (inv + angular_term(body, arm, normal, density)).max(1e-9);
    let magnitude = -(1.0 + restitution) * normal_speed / denominator;
    apply_impulse(body, scale(normal, magnitude), arm, density);
    relative = velocity_at_packed(body, arm);
    let tv = sub(relative, scale(normal, dot(relative, normal)));
    let ts = length(tv);
    if ts <= 1e-8 {
        return;
    }
    let tangent = scale(tv, 1.0 / ts);
    let td = (inv + angular_term(body, arm, tangent, density)).max(1e-9);
    let tm = (-ts / td).clamp(-body[28] * magnitude, body[28] * magnitude);
    apply_impulse(body, scale(tangent, tm), arm, density)
}
fn static_contact(body: &mut [f32; 32], provider: &dyn StaticContactProvider, density: f32) {
    if body[20] / density <= 0.0 {
        return;
    }
    let position = v(body[0], body[1], body[2]);
    let broad = body[7].max(1e-7);
    let cell = provider.cell_size_m();
    let (mut best, mut best_normal, mut best_radius) = (0.0, v(0.0, 1.0, 0.0), 0.0);
    provider.visit_candidates(position, broad, &mut |voxel| {
        if voxel.fraction <= 0.0 {
            return;
        }
        let min = voxel.minimum_m;
        let max = voxel.maximum_m;
        let closest = v(
            position[0].clamp(min[0], max[0]),
            position[1].clamp(min[1], max[1]),
            position[2].clamp(min[2], max[2]),
        );
        let delta = sub(position, closest);
        let separation = length(delta);
        if separation > broad {
            return;
        }
        let (normal, penetration, radius) = if voxel.fraction < 1.0 {
            let q = voxel.coordinate;
            let sdf_q8 = |at| (provider.signed_distance_cells(at) * 256.0).clamp(-512.0, 512.0);
            let sx = (sdf_q8([q[0] + 1, q[1], q[2]]) - sdf_q8([q[0] - 1, q[1], q[2]])) / cell[0];
            let sy = (sdf_q8([q[0], q[1] + 1, q[2]]) - sdf_q8([q[0], q[1] - 1, q[2]])) / cell[1];
            let sz = (sdf_q8([q[0], q[1], q[2] + 1]) - sdf_q8([q[0], q[1], q[2] - 1])) / cell[2];
            let centre = scale(
                add(v(min[0], min[1], min[2]), v(max[0], max[1], max[2])),
                0.5,
            );
            let normal = normalize(
                v(sx, sy, sz),
                normalize(sub(position, centre), v(0.0, 1.0, 0.0)),
            );
            let sdf = (sdf_q8(q) / 256.0) * cell[0].min(cell[1]).min(cell[2]);
            let surface = sub(centre, scale(normal, sdf));
            let radius = support_radius_packed(body, normal);
            (
                normal,
                radius - (dot(normal, position) - dot(normal, surface)),
                radius,
            )
        } else if separation > 1e-7 {
            let normal = scale(delta, 1.0 / separation);
            let radius = support_radius_packed(body, normal);
            (normal, radius - separation, radius)
        } else {
            let ds = [
                position[0] - min[0],
                max[0] - position[0],
                position[1] - min[1],
                max[1] - position[1],
                position[2] - min[2],
                max[2] - position[2],
            ];
            let normals = [
                v(-1.0, 0.0, 0.0),
                v(1.0, 0.0, 0.0),
                v(0.0, -1.0, 0.0),
                v(0.0, 1.0, 0.0),
                v(0.0, 0.0, -1.0),
                v(0.0, 0.0, 1.0),
            ];
            let mut chosen = 0;
            for i in 1..6 {
                if ds[i] < ds[chosen] {
                    chosen = i
                }
            }
            let normal = normals[chosen];
            let radius = support_radius_packed(body, normal);
            (normal, radius + ds[chosen].max(0.0), radius)
        };
        if penetration > best {
            best = penetration;
            best_normal = normal;
            best_radius = radius
        }
    });
    if best > 0.0 {
        resolve_static_contact(body, best_normal, best, best_radius, density)
    }
}

pub fn advance_rigid_bodies_with_contacts(
    bodies: &mut [RigidBodyState],
    exchange: &[FluidExchange],
    dt: f32,
    density: f32,
    gravity: Vec3,
    cell_volume: f32,
    snapshot_count: u32,
    static_provider: Option<&dyn StaticContactProvider>,
) -> Result<Vec<RigidBodyState>, SceneModelError> {
    let previous = bodies.to_vec();
    let snapshots = snapshot_count.max(1) as f32;
    for (index, state) in bodies.iter_mut().take(12).enumerate() {
        if state.held {
            continue;
        }
        let mut body = pack(state, density)?;
        if !(body[30] > 0.5 && body[30] < 1.5) {
            continue;
        }
        let e = exchange.get(index).copied().unwrap_or_default().lanes;
        let wet = e[6] as f32 / 65536.0 / snapshots;
        let volume = match state.dimensional_volume_m3 {
            Some(volume) => volume as f32,
            None => {
                primitive_volume(state.description.shape, state.description.dimensions_m)? as f32
            }
        };
        let displaced = (wet * cell_volume).clamp(0.0, volume);
        let impulse = v(e[0] as f32 * 1e-6, e[1] as f32 * 1e-6, e[2] as f32 * 1e-6);
        let angular = v(e[3] as f32 * 1e-6, e[4] as f32 * 1e-6, e[5] as f32 * 1e-6);
        let weighted = scale(
            v(e[7] as f32 * 1e-4, e[8] as f32 * 1e-4, e[9] as f32 * 1e-4),
            1.0 / snapshots,
        );
        let weight = e[11] as f32 / 65536.0 / snapshots;
        let mean = if weight > 1e-8 {
            scale(weighted, 1.0 / weight)
        } else {
            v(0.0, 0.0, 0.0)
        };
        let mass = if body[20] > 0.0 {
            density / body[20]
        } else {
            1e30
        };
        let immersed = (displaced / volume.max(1e-9)).clamp(0.0, 1.0);
        let velocity = v(body[12], body[13], body[14]);
        let relative = sub(velocity, mean);
        let drag = scale(
            relative,
            -0.5 * density
                * 0.9
                * std::f32::consts::PI
                * body[7]
                * body[7]
                * immersed
                * length(relative),
        );
        let g = v(gravity.x as f32, gravity.y as f32, gravity.z as f32);
        let buoyancy = if e[10] != 0 {
            v(0.0, 0.0, 0.0)
        } else {
            scale(g, -density * displaced)
        };
        let acceleration = scale(
            add(
                add(scale(g, mass), scale(impulse, 1.0 / dt.max(1e-8))),
                add(drag, buoyancy),
            ),
            1.0 / (mass + 0.5 * density * displaced).max(1e-8),
        );
        let next_velocity = add(velocity, scale(acceleration, dt));
        body[24] += angular[0];
        body[25] += angular[1];
        body[26] += angular[2];
        for a in 0..3 {
            body[12 + a] = next_velocity[a];
            body[a] += next_velocity[a] * dt
        }
        let omega = inverse_inertia(&body, v(body[24], body[25], body[26]), density);
        body[16..19].copy_from_slice(&omega);
        let q = [body[8], body[9], body[10], body[11]];
        let derivative = qmul([0.0, omega[0], omega[1], omega[2]], q);
        let mut uq = [0.0; 4];
        for a in 0..4 {
            uq[a] = q[a] + 0.5 * dt * derivative[a]
        }
        let norm = ((uq[0] * uq[0] + uq[1] * uq[1]) + uq[2] * uq[2] + uq[3] * uq[3]).sqrt();
        state.position_m = Vec3 {
            x: body[0] as f64,
            y: body[1] as f64,
            z: body[2] as f64,
        };
        state.linear_velocity_m_s = Vec3 {
            x: body[12] as f64,
            y: body[13] as f64,
            z: body[14] as f64,
        };
        state.angular_velocity_rad_s = Vec3 {
            x: omega[0] as f64,
            y: omega[1] as f64,
            z: omega[2] as f64,
        };
        state.angular_momentum_kg_m2_s = Vec3 {
            x: body[24] as f64,
            y: body[25] as f64,
            z: body[26] as f64,
        };
        state.orientation = Quaternion {
            w: (uq[0] / norm) as f64,
            x: (uq[1] / norm) as f64,
            y: (uq[2] / norm) as f64,
            z: (uq[3] / norm) as f64,
        };
    }
    let mut packed = bodies
        .iter()
        .take(12)
        .map(|body| pack(body, density))
        .collect::<Result<Vec<_>, _>>()?;
    for _ in 0..6 {
        if let Some(provider) = static_provider {
            for body in &mut packed {
                static_contact(body, provider, density);
            }
        }
        for a in 0..packed.len() {
            for b in a + 1..packed.len() {
                let (left, right) = packed.split_at_mut(b);
                solve_body_pair(&mut left[a], &mut right[0], density)
            }
        }
    }
    for (state, record) in bodies.iter_mut().zip(&packed) {
        apply_packed(state, record)
    }
    Ok(previous)
}

pub fn advance_rigid_bodies(
    bodies: &mut [RigidBodyState],
    exchange: &[FluidExchange],
    dt: f32,
    density: f32,
    gravity: Vec3,
    cell_volume: f32,
    snapshot_count: u32,
) -> Result<Vec<RigidBodyState>, SceneModelError> {
    advance_rigid_bodies_with_contacts(
        bodies,
        exchange,
        dt,
        density,
        gravity,
        cell_volume,
        snapshot_count,
        None,
    )
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RigidCouplingReceipt {
    pub body_id: String,
    #[serde(rename = "displacedVolume_m3")]
    pub displaced_volume_m3: f64,
    #[serde(rename = "meanFluidVelocity_m_s")]
    pub mean_fluid_velocity_m_s: Vec3,
    #[serde(rename = "force_N")]
    pub force_n: Vec3,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RigidCouplingResult {
    pub exchange: Vec<FluidExchange>,
    pub receipts: Vec<RigidCouplingReceipt>,
}

pub(crate) fn body_contains(body: &RigidBodyState, p: Vec3) -> Result<bool, SceneModelError> {
    let q = body.orientation;
    let x = p.x - body.position_m.x;
    let y = p.y - body.position_m.y;
    let z = p.z - body.position_m.z;
    let u = [-q.x, -q.y, -q.z];
    let uv = [
        u[1] * z - u[2] * y,
        u[2] * x - u[0] * z,
        u[0] * y - u[1] * x,
    ];
    let uuv = [
        u[1] * uv[2] - u[2] * uv[1],
        u[2] * uv[0] - u[0] * uv[2],
        u[0] * uv[1] - u[1] * uv[0],
    ];
    let p = [
        x + 2.0 * (q.w * uv[0] + uuv[0]),
        y + 2.0 * (q.w * uv[1] + uuv[1]),
        z + 2.0 * (q.w * uv[2] + uuv[2]),
    ];
    let d = body.description.dimensions_m;
    Ok(match body.description.shape {
        RigidShape::Sphere => p[0].hypot(p[1]).hypot(p[2]) <= d.x,
        RigidShape::Box => {
            p[0].abs() <= 0.5 * d.x && p[1].abs() <= 0.5 * d.y && p[2].abs() <= 0.5 * d.z
        }
        RigidShape::Cylinder => p[0] * p[0] + p[2] * p[2] <= d.x * d.x && p[1].abs() <= 0.5 * d.y,
        RigidShape::Capsule => {
            let cy = p[1].clamp(-0.5 * d.y, 0.5 * d.y);
            p[0].hypot(p[1] - cy).hypot(p[2]) <= d.x
        }
        RigidShape::Cup => {
            let t = d.z.clamp(1e-4, (0.95 * d.x).min(0.95 * d.y));
            let ox = p[0].hypot(p[2]) - d.x;
            let oy = p[1].abs() - 0.5 * d.y;
            let outer = ox.max(0.0).hypot(oy.max(0.0)) + ox.max(oy).min(0.0);
            let cavity = (p[0].hypot(p[2]) - (d.x - t)).max((-0.5 * d.y + t) - p[1]);
            outer.max(-cavity) <= 0.0
        }
    })
}
/// Signed primitive distance in body coordinates, shared by grid hosts.
pub(crate) fn body_signed_distance(body: &RigidBodyState, world: Vec3) -> f32 {
    let q = body.orientation;
    let d = sub(
        v(world.x as f32, world.y as f32, world.z as f32),
        v(
            body.position_m.x as f32,
            body.position_m.y as f32,
            body.position_m.z as f32,
        ),
    );
    let p = qrotate([q.w as f32, -q.x as f32, -q.y as f32, -q.z as f32], d);
    let size = body.description.dimensions_m;
    let box_sdf = |a: f32, b: f32, c: f32| {
        length(v(a.max(0.0), b.max(0.0), c.max(0.0))) + a.max(b).max(c).min(0.0)
    };
    let cylinder = |radius: f32, height: f32| {
        let x = p[0].hypot(p[2]) - radius;
        let y = p[1].abs() - 0.5 * height;
        x.max(0.0).hypot(y.max(0.0)) + x.max(y).min(0.0)
    };
    match body.description.shape {
        RigidShape::Sphere => length(p) - size.x as f32,
        RigidShape::Box => box_sdf(
            p[0].abs() - 0.5 * size.x as f32,
            p[1].abs() - 0.5 * size.y as f32,
            p[2].abs() - 0.5 * size.z as f32,
        ),
        RigidShape::Capsule => {
            length(v(
                p[0],
                p[1] - p[1].clamp(-0.5 * size.y as f32, 0.5 * size.y as f32),
                p[2],
            )) - size.x as f32
        }
        RigidShape::Cylinder => cylinder(size.x as f32, size.y as f32),
        RigidShape::Cup => {
            let t = (size.z as f32).clamp(1e-4, 0.95 * (size.x as f32).min(size.y as f32));
            let cavity =
                (p[0].hypot(p[2]) - (size.x as f32 - t)).max(-0.5 * size.y as f32 + t - p[1]);
            cylinder(size.x as f32, size.y as f32).max(-cavity)
        }
    }
}

pub(crate) fn round_ties_even(value: f64) -> i32 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 {
        floor as i32
    } else if fraction > 0.5 {
        (floor + 1.0) as i32
    } else if floor as i64 % 2 == 0 {
        floor as i32
    } else {
        (floor + 1.0) as i32
    }
}

/// Produce the fixed-point exchange consumed on the following rigid step.
pub fn rigid_coupling_loads(
    frame: SliceFrame,
    graph: &Graph,
    fields: &Fields,
    bodies: &[RigidBodyState],
    density_kg_m3: f64,
    gravity: Vec3,
) -> Result<RigidCouplingResult, SceneModelError> {
    let count = bodies.len().min(12);
    let mut exchange = vec![FluidExchange::default(); count];
    let h = frame.source_cell_size;
    for cell in &graph.cells {
        let (mut best, mut owner) = (0.0, -1i32);
        for (body_id, body) in bodies.iter().take(count).enumerate() {
            let mut covered = 0u32;
            let mut samples = 0u32;
            let y0 = (cell.center[1] as f64 - 0.5 * cell.widths[1] as f64).round() as i32;
            let y1 = (cell.center[1] as f64 + 0.5 * cell.widths[1] as f64).round() as i32;
            let x0 = (cell.center[0] as f64 - 0.5 * cell.widths[0] as f64).round() as i32;
            let x1 = (cell.center[0] as f64 + 0.5 * cell.widths[0] as f64).round() as i32;
            for y in y0..y1 {
                for x in x0..x1 {
                    for dx in [-0.4, 0.4] {
                        for dy in [-0.4, 0.4] {
                            let p = Vec3 {
                                x: frame.origin_x + (x as f64 + 0.5 + dx) * h,
                                y: frame.origin_y + (y as f64 + 0.5 + dy) * h,
                                z: frame.center_z,
                            };
                            covered += body_contains(body, p)? as u32;
                            samples += 1
                        }
                    }
                }
            }
            let coverage = if samples > 0 {
                covered as f64 / samples as f64
            } else {
                0.0
            };
            if coverage > best {
                best = coverage;
                owner = body_id as i32
            }
        }
        if owner < 0 || best <= 0.0 {
            continue;
        }
        let id = cell.id as usize;
        let open = fields.capacity[id] as f64;
        let wet = (fields.density[id] as f64 / open.max(0.125)).clamp(0.0, 1.0);
        if wet <= 0.0 {
            continue;
        }
        let weight = wet * best * cell.measure as f64;
        let e = &mut exchange[owner as usize].lanes;
        let q = round_ties_even(weight * 65536.0);
        e[6] += q;
        e[11] += q;
        e[7] += round_ties_even(weight * fields.cell_velocity[2 * id] as f64 * h * 10000.0);
        e[8] += round_ties_even(weight * fields.cell_velocity[2 * id + 1] as f64 * h * 10000.0)
    }
    let mut receipts = Vec::with_capacity(count);
    for (index, body) in bodies.iter().take(count).enumerate() {
        let e = exchange[index].lanes;
        let displaced = (e[6] as f64 / 65536.0 * h * h * h).clamp(
            0.0,
            primitive_volume(body.description.shape, body.description.dimensions_m)?,
        );
        let weight = e[11] as f64 / 65536.0;
        let mean = if weight > 1e-8 {
            Vec3 {
                x: e[7] as f64 * 1e-4 / weight,
                y: e[8] as f64 * 1e-4 / weight,
                z: 0.0,
            }
        } else {
            Vec3::default()
        };
        let relative = Vec3 {
            x: body.linear_velocity_m_s.x - mean.x,
            y: body.linear_velocity_m_s.y - mean.y,
            z: body.linear_velocity_m_s.z - mean.z,
        };
        let speed = relative.x.hypot(relative.y).hypot(relative.z);
        let volume = primitive_volume(body.description.shape, body.description.dimensions_m)?;
        let immersed = (displaced / volume.max(1e-9)).clamp(0.0, 1.0);
        let radius = bounding_radius(&body.description)? as f64;
        let drag_scale =
            -0.5 * density_kg_m3 * 0.9 * std::f64::consts::PI * radius * radius * immersed * speed;
        let drag = Vec3 {
            x: drag_scale * relative.x,
            y: drag_scale * relative.y,
            z: drag_scale * relative.z,
        };
        let buoyancy = Vec3 {
            x: -density_kg_m3 * displaced * gravity.x,
            y: -density_kg_m3 * displaced * gravity.y,
            z: -density_kg_m3 * displaced * gravity.z,
        };
        let denominator = (body.mass_kg + 0.5 * density_kg_m3 * displaced).max(1e-8);
        let desired = Vec3 {
            x: (body.mass_kg * gravity.x + drag.x + buoyancy.x) / denominator,
            y: (body.mass_kg * gravity.y + drag.y + buoyancy.y) / denominator,
            z: (body.mass_kg * gravity.z + drag.z + buoyancy.z) / denominator,
        };
        let force = Vec3 {
            x: body.mass_kg * (desired.x - gravity.x),
            y: body.mass_kg * (desired.y - gravity.y),
            z: body.mass_kg * (desired.z - gravity.z),
        };
        receipts.push(RigidCouplingReceipt {
            body_id: body.description.id.clone(),
            displaced_volume_m3: displaced,
            mean_fluid_velocity_m_s: mean,
            force_n: force,
        })
    }
    Ok(RigidCouplingResult { exchange, receipts })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(id: &str, x: f64, velocity: f64) -> RigidBodyDescription {
        RigidBodyDescription {
            id: id.into(),
            name: id.into(),
            shape: RigidShape::Sphere,
            dimensions_m: Vec3 {
                x: 0.5,
                y: 0.0,
                z: 0.0,
            },
            density_kg_m3: 500.0,
            position_m: Vec3 { x, y: 1.0, z: 0.0 },
            orientation: Quaternion::default(),
            linear_velocity_m_s: Vec3 {
                x: velocity,
                y: 0.0,
                z: 0.0,
            },
            angular_velocity_rad_s: Vec3::default(),
            restitution: 0.5,
            friction: 0.2,
            motion: RigidMotion::Dynamic,
        }
    }

    /// Values were emitted by stepSliceRigidBodies for this impact fixture.
    #[test]
    fn two_sphere_impact_matches_typescript_reference() {
        let mut bodies = vec![
            initialize_body(sphere("a", -0.4, 1.0)).unwrap(),
            initialize_body(sphere("b", 0.4, -1.0)).unwrap(),
        ];
        advance_rigid_bodies(&mut bodies, &[], 0.01, 1000.0, Vec3::default(), 1.0, 1).unwrap();
        assert_eq!(
            (bodies[0].position_m.x as f32).to_bits(),
            (-0.5f32).to_bits()
        );
        assert_eq!((bodies[1].position_m.x as f32).to_bits(), 0.5f32.to_bits());
        assert_eq!(
            (bodies[0].linear_velocity_m_s.x as f32).to_bits(),
            (-0.5f32).to_bits()
        );
        assert_eq!(
            (bodies[1].linear_velocity_m_s.x as f32).to_bits(),
            0.5f32.to_bits()
        );
    }

    #[test]
    fn authored_capsule_and_cup_match_typescript_shape_table() {
        let capsule = sphere("capsule", 0.0, 0.0);
        let mut capsule = RigidBodyDescription {
            shape: RigidShape::Capsule,
            dimensions_m: Vec3 {
                x: 0.2,
                y: 0.6,
                z: 0.0,
            },
            density_kg_m3: 700.0,
            ..capsule
        };
        let ci = inertia(&capsule).unwrap();
        assert_eq!(
            primitive_volume(capsule.shape, capsule.dimensions_m).unwrap(),
            0.10890854532444616
        );
        assert_eq!(
            [ci.x, ci.y, ci.z],
            [5.653191260379714, 1.4308907339550316, 5.653191260379714]
        );
        let cs = initialize_body(capsule.clone()).unwrap();
        assert!(body_contains(
            &cs,
            Vec3 {
                x: 0.0,
                y: 1.45,
                z: 0.0
            }
        )
        .unwrap());
        capsule.id = "cup".into();
        capsule.name = "cup".into();
        capsule.shape = RigidShape::Cup;
        capsule.dimensions_m = Vec3 {
            x: 0.5,
            y: 1.0,
            z: 0.1,
        };
        let ui = inertia(&capsule).unwrap();
        assert_eq!(
            primitive_volume(capsule.shape, capsule.dimensions_m).unwrap(),
            0.33300882128051795
        );
        assert_eq!(
            [ui.x, ui.y, ui.z],
            [45.34208317048588, 43.38853613872862, 45.34208317048588]
        );
        let cup = initialize_body(capsule).unwrap();
        assert!(body_contains(
            &cup,
            Vec3 {
                x: 0.45,
                y: 1.0,
                z: 0.0
            }
        )
        .unwrap());
        assert!(!body_contains(
            &cup,
            Vec3 {
                x: 0.0,
                y: 1.25,
                z: 0.0
            }
        )
        .unwrap());
    }

    struct UnitSolid;
    impl StaticContactProvider for UnitSolid {
        fn cell_size_m(&self) -> [f32; 3] {
            [1.0; 3]
        }
        fn visit_candidates(
            &self,
            _: [f32; 3],
            _: f32,
            visitor: &mut dyn FnMut(StaticContactVoxel),
        ) {
            visitor(StaticContactVoxel {
                coordinate: [0; 3],
                fraction: 1.0,
                minimum_m: [0.0; 3],
                maximum_m: [1.0; 3],
            });
        }
        fn signed_distance_cells(&self, _: [i32; 3]) -> f32 {
            0.0
        }
    }

    #[test]
    fn full_voxel_interior_contact_uses_typescript_axis_order() {
        let mut description = sphere("contact", 0.5, 0.0);
        description.position_m = Vec3 {
            x: 0.5,
            y: 0.5,
            z: 0.5,
        };
        let mut bodies = vec![initialize_body(description).unwrap()];
        advance_rigid_bodies_with_contacts(
            &mut bodies,
            &[],
            0.01,
            1000.0,
            Vec3::default(),
            1.0,
            1,
            Some(&UnitSolid),
        )
        .unwrap();
        assert_eq!(
            (bodies[0].position_m.x as f32).to_bits(),
            (-0.5000001f32).to_bits()
        );
        assert_eq!(bodies[0].position_m.y as f32, 0.5);
    }
}
