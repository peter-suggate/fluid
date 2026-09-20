//! Serialized whole-scene boundary shared by native and Wasm parity runners.
//! This is a diagnostic runner, separate from the interactive owned-world ABI.
use super::{grid::Grid, world::World, UniformGeometricOptions};
use crate::types::ValidationError;
use serde::Deserialize;
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Request {
    #[serde(default)]
    scene: Option<crate::initial_scene::SceneDocument>,
    #[serde(default)]
    options: UniformGeometricOptions,
    #[serde(default = "dims")]
    dimensions: [usize; 2],
    #[serde(default = "spacing")]
    cell_size: [f32; 2],
    #[serde(default)]
    phi: Vec<f32>,
    #[serde(default)]
    volume: Vec<f32>,
    #[serde(default)]
    capacity: Vec<f32>,
    #[serde(default)]
    velocity: Vec<[f32; 2]>,
    #[serde(default)]
    low_x: Vec<f32>,
    #[serde(default)]
    low_y: Vec<f32>,
    #[serde(default)]
    released: Vec<u8>,
    #[serde(default)]
    gravity: [f32; 2],
    #[serde(default)]
    open_top: bool,
    #[serde(default = "rho")]
    density: f32,
    #[serde(default = "dt")]
    dt: f32,
    #[serde(default = "frames")]
    frames: u32,
    #[serde(default)]
    audit_surface: bool,
    #[serde(default)]
    redistance_only: bool,
}
fn dims() -> [usize; 2] {
    [16, 16]
}
fn spacing() -> [f32; 2] {
    [0.05; 2]
}
fn rho() -> f32 {
    1000.0
}
fn dt() -> f32 {
    1.0 / 30.0
}
fn frames() -> u32 {
    30
}
pub fn run(r: Request) -> Result<serde_json::Value, ValidationError> {
    let mut world = if let Some(scene) = r.scene {
        World::from_document(&scene, r.options)?
    } else {
        let mut g = Grid::new(r.dimensions, r.cell_size, r.open_top)?;
        if !r.phi.is_empty() {
            g.phi = r.phi;
        }
        if !r.volume.is_empty() {
            g.volume = r.volume;
        }
        if !r.capacity.is_empty() {
            g.capacity = r.capacity;
        }
        if !r.velocity.is_empty() {
            g.velocity = r.velocity;
        }
        if !r.low_x.is_empty() {
            g.low_x = r.low_x;
        }
        if !r.low_y.is_empty() {
            g.low_y = r.low_y;
        }
        if !r.released.is_empty() {
            g.released = r.released;
        }
        World::from_grid(g, r.options, r.gravity, r.density, 0.0, 0.0)?
    };
    if r.redistance_only {
        super::surface::redistance(&mut world.grid);
        return Ok(serde_json::json!({"phi":world.grid.phi}));
    }
    if r.audit_surface {
        world.advected_phi_audit = Some(Vec::new());
    }
    let initial: f64 = world.grid.volume.iter().map(|&v| v as f64).sum();
    let mut receipts = Vec::new();
    for _ in 0..r.frames {
        world.advance(r.dt)?;
        receipts.push(world.receipt.clone());
    }
    let mut result = serde_json::json!({"initialVolume":initial,"receipts":receipts,"volume":world.grid.volume,"phi":world.grid.phi,"velocity":world.grid.velocity,"lowX":world.grid.low_x,"lowY":world.grid.low_y,"pressure":world.pressure.levels[0].p,"released":world.grid.released,"tileClasses":world.tile_classes});
    if let Some(phi) = world.advected_phi_audit {
        result["advectedPhi"] = serde_json::json!(phi);
    }
    Ok(result)
}
