//! Serde model for production-authored physical scene fields.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
impl Vec3 {
    pub fn array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
impl Default for Quaternion {
    fn default() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RigidShape {
    Sphere,
    Box,
    Capsule,
    Cylinder,
    Cup,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RigidMotion {
    #[default]
    Dynamic,
    Static,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RigidBodyDescription {
    pub id: String,
    pub name: String,
    pub shape: RigidShape,
    #[serde(rename = "dimensions_m")]
    pub dimensions_m: Vec3,
    #[serde(rename = "density_kg_m3")]
    pub density_kg_m3: f64,
    #[serde(rename = "position_m")]
    pub position_m: Vec3,
    pub orientation: Quaternion,
    #[serde(rename = "linearVelocity_m_s")]
    pub linear_velocity_m_s: Vec3,
    #[serde(rename = "angularVelocity_rad_s")]
    pub angular_velocity_rad_s: Vec3,
    pub restitution: f64,
    pub friction: f64,
    #[serde(default)]
    pub motion: RigidMotion,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FluidInflow {
    #[serde(rename = "center_m")]
    pub center_m: Vec3,
    #[serde(rename = "radius_m")]
    pub radius_m: f64,
    #[serde(rename = "length_m")]
    pub length_m: f64,
    #[serde(rename = "velocity_m_s")]
    pub velocity_m_s: Vec3,
    #[serde(rename = "start_s")]
    pub start_s: f64,
    #[serde(rename = "end_s")]
    pub end_s: f64,
    #[serde(rename = "ramp_s")]
    pub ramp_s: f64,
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Container {
    #[serde(rename = "width_m")]
    pub width_m: f64,
    #[serde(rename = "height_m")]
    pub height_m: f64,
    #[serde(rename = "depth_m")]
    pub depth_m: f64,
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FluidPhysics {
    #[serde(rename = "density_kg_m3")]
    pub density_kg_m3: f64,
    #[serde(rename = "dynamicViscosity_Pa_s")]
    pub dynamic_viscosity_pa_s: f64,
    #[serde(rename = "surfaceTension_N_m")]
    pub surface_tension_n_m: f64,
    #[serde(rename = "gravity_m_s2")]
    pub gravity_m_s2: Vec3,
    #[serde(default)]
    pub inflow: Option<FluidInflow>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhysicalScene {
    pub schema_version: String,
    pub scene_id: String,
    pub container: Container,
    pub fluid: FluidPhysics,
    #[serde(default)]
    pub rigid_bodies: Vec<RigidBodyDescription>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SliceFrame {
    pub origin_x: f64,
    pub origin_y: f64,
    pub center_z: f64,
    pub source_cell_size: f64,
    pub center_cell_z: i32,
    pub source_dimensions: [u32; 3],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SceneModelError(pub String);
impl std::fmt::Display for SceneModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for SceneModelError {}

pub fn validate_supported_shape(shape: RigidShape) -> Result<(), SceneModelError> {
    let _ = shape;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_literal_production_si_keys() {
        let json = r#"{
          "schemaVersion":"2.0.0","sceneId":"fixture",
          "container":{"width_m":4,"height_m":3,"depth_m":2,"top":"open"},
          "fluid":{"density_kg_m3":998.2,"dynamicViscosity_Pa_s":0.001,
            "surfaceTension_N_m":0.072,"gravity_m_s2":{"x":0,"y":-9.81,"z":0},
            "initialCondition":"tank-fill",
            "inflow":{"center_m":{"x":-1,"y":2,"z":0},"radius_m":0.1,
              "length_m":0.2,"velocity_m_s":{"x":2,"y":0,"z":0},
              "start_s":0,"end_s":4,"ramp_s":0.5}},
          "rigidBodies":[{"id":"ball","name":"Ball","shape":"sphere",
            "dimensions_m":{"x":0.25,"y":0,"z":0},"density_kg_m3":750,
            "position_m":{"x":0,"y":1,"z":0},"orientation":{"w":1,"x":0,"y":0,"z":0},
            "linearVelocity_m_s":{"x":1,"y":0,"z":0},
            "angularVelocity_rad_s":{"x":0,"y":0,"z":2},
            "restitution":0.2,"friction":0.4}]
        }"#;
        let scene: PhysicalScene = serde_json::from_str(json).unwrap();
        assert_eq!(scene.fluid.gravity_m_s2.y, -9.81);
        assert_eq!(scene.fluid.inflow.unwrap().velocity_m_s.x, 2.0);
        assert_eq!(scene.rigid_bodies[0].position_m.y, 1.0);
        assert_eq!(scene.rigid_bodies[0].motion, RigidMotion::Dynamic);
    }
}
