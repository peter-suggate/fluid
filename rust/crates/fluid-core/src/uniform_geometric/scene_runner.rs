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
    #[serde(default)]
    audit_stages: bool,
    #[serde(default)]
    audit_energy: bool,
    #[serde(default)]
    disable_overfill_correction: bool,
    #[serde(default)]
    energy_experiment: super::energy_experiment::Config,
    #[serde(default)]
    energy_snapshot_frames: Vec<u32>,
    #[serde(default)]
    energy_snapshot_stages: Vec<String>,
    #[serde(default)]
    liquid_injections: Vec<ScheduledDrop>,
    #[serde(default)]
    sharpening_rounds: Option<usize>,
    #[serde(default)]
    audit_replay_frames: Vec<u32>,
    #[serde(default)]
    audit_trace_frames: Vec<u32>,
    #[serde(default)]
    swept_extension: Option<super::swept_extension::Config>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ScheduledDrop {
    frame: u32,
    drop: super::grid::LiquidDrop,
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
    let gravity = r.scene.as_ref().map_or(r.gravity, |s| {
        [s.fluid.gravity_m_s2.x as f32, s.fluid.gravity_m_s2.y as f32]
    });
    let density = r
        .scene
        .as_ref()
        .map_or(r.density, |s| s.fluid.density_kg_m3 as f32);
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
    r.energy_experiment.validate()?;
    if r.disable_overfill_correction && r.energy_experiment.mode != "off" {
        return Err(ValidationError(
            "compensation must preserve the overfill source".into(),
        ));
    }
    for injection in &r.liquid_injections {
        if injection.frame == 0
            || injection.frame > r.frames
            || !injection.drop.radius_m.is_finite()
            || injection.drop.radius_m <= 0.0
            || injection.drop.centre_m.iter().any(|v| !v.is_finite())
            || r.liquid_injections
                .iter()
                .filter(|x| x.frame == injection.frame)
                .count()
                > 64
        {
            return Err(ValidationError(
                "invalid diagnostic liquid injection".into(),
            ));
        }
    }
    world.energy_experiment = r.energy_experiment;
    world.diagnostic_disable_overfill_correction = r.disable_overfill_correction;
    if let Some(config) = r.swept_extension {
        config.validate()?;
        world.swept_extension = config;
    }
    if r.redistance_only {
        super::surface::redistance(&mut world.grid);
        return Ok(serde_json::json!({"phi":world.grid.phi}));
    }
    if r.audit_surface {
        world.advected_phi_audit = Some(Vec::new());
    }
    let initial: f64 = world.grid.volume.iter().map(|&v| v as f64).sum();
    let mut receipts = Vec::new();
    let mut stages = Vec::new();
    let mut energy = Vec::new();
    let mut energy_snapshots = Vec::new();
    let mut compensation = Vec::new();
    let mut replays = Vec::new();
    let mut trace_replays = Vec::new();
    let rounds = r.sharpening_rounds.unwrap_or(8);
    if rounds > 512 {
        return Err(ValidationError(
            "diagnostic sharpening rounds must be <= 512".into(),
        ));
    }
    for frame in 1..=r.frames {
        for injection in r.liquid_injections.iter().filter(|x| x.frame == frame) {
            world.grid.drops.push(injection.drop.clone());
        }
        if r.audit_stages
            || r.audit_energy
            || !r.energy_snapshot_frames.is_empty()
            || r.sharpening_rounds.is_some()
            || !r.audit_replay_frames.is_empty()
            || !r.audit_trace_frames.is_empty()
        {
            let options = world.options.clone();
            let compensation_config = world.energy_experiment.clone();
            world.advance_observed(r.dt, rounds, |stage, grid| {
                if stage == "start" && r.audit_trace_frames.contains(&frame) {
                    trace_replays.push(serde_json::json!({"frame":frame,
                        "initialArea":super::diagnostics::measure(grid,r.dt,32).contour_area,
                        "traces":super::diagnostics::trace_replays(grid,&options,r.dt)}));
                }
                if r.energy_snapshot_frames.contains(&frame) && (r.energy_snapshot_stages.is_empty() || r.energy_snapshot_stages.iter().any(|s| s == stage)) {
                    if stage == "sharpened" && compensation_config.mode.starts_with("balance-") {
                        energy_snapshots.push(serde_json::json!({"frame":frame,"stage":"balanceSources","fields":super::energy_experiment::balance_fields(grid,&options.volume_pressure_rows,r.dt,&compensation_config)}));
                    }
                    energy_snapshots.push(serde_json::json!({"frame":frame,"stage":stage,"volume":grid.volume,"phi":grid.phi,"velocity":grid.velocity,"lowX":grid.low_x,"lowY":grid.low_y,"released":grid.released}));
                }
                if r.audit_energy {
                    energy.push(serde_json::json!({"frame":frame,"stage":stage,"metrics":energy_metrics(grid, density, gravity)}));
                }
                if r.audit_stages {
                    stages.push(serde_json::json!({"frame":frame,"stage":stage,
                        "metrics":super::diagnostics::measure(grid,r.dt,32)}));
                }
                if stage == "transported" && r.audit_replay_frames.contains(&frame) {
                    let mut replay = grid.clone();
                    let mut completed = 0;
                    for rounds in [0, 8, 32, 128, 512] {
                        super::surface::sharpen_rounds(&mut replay, &options, rounds - completed);
                        completed = rounds;
                        replays.push(serde_json::json!({"frame":frame,"rounds":rounds,
                            "metrics":super::diagnostics::measure(&replay,r.dt,128)}));
                    }
                }
            })?;
        } else {
            world.advance(r.dt)?;
        }
        receipts.push(world.receipt.clone());
        if world.energy_experiment.mode != "off" && world.energy_experiment.audit_receipt {
            compensation.push(world.energy_receipt.clone());
        }
    }
    let mut result = serde_json::json!({"initialVolume":initial,"receipts":receipts,"volume":world.grid.volume,"phi":world.grid.phi,"velocity":world.grid.velocity,"lowX":world.grid.low_x,"lowY":world.grid.low_y,"pressure":world.pressure.levels[0].p,"released":world.grid.released,"tileClasses":world.tile_classes});
    if let Some(phi) = world.advected_phi_audit {
        result["advectedPhi"] = serde_json::json!(phi);
    }
    if !compensation.is_empty() {
        result["energyCompensation"] = serde_json::json!(compensation);
    }
    if !energy_snapshots.is_empty() {
        result["energySnapshots"] = serde_json::json!(energy_snapshots);
    }
    if r.audit_energy {
        result["energy"] = serde_json::json!(energy);
    }
    if r.audit_stages {
        result["stages"] = serde_json::json!(stages);
        result["finalHighResolutionMetrics"] =
            serde_json::json!(super::diagnostics::measure(&world.grid, r.dt, 128));
    }
    if !replays.is_empty() {
        result["sharpeningReplays"] = serde_json::json!(replays);
    }
    if !trace_replays.is_empty() {
        result["traceReplays"] = serde_json::json!(trace_replays);
    }
    Ok(result)
}

// Joules per metre of depth. Both cell-centred and face-square quadratures
// are reported so opposing MAC faces cannot hide energy by cancellation.
fn energy_metrics(g: &Grid, rho: f32, gravity: [f32; 2]) -> serde_json::Value {
    let mut kinetic = 0.0;
    let mut mac = 0.0;
    let mut potential = 0.0;
    let mut excess = 0.0;
    let mut max_excess = 0.0_f64;
    let mut phi_kinetic = 0.0;
    let mut phi_area = 0.0;
    let mut phi_potential = 0.0;
    for i in 0..g.volume.len() {
        let p = g.point(i);
        let m = rho as f64 * g.h[0] as f64 * g.h[1] as f64;
        let mut speed2 = 0.0;
        let mut faces2 = 0.0;
        for a in 0..2 {
            let mut q = p;
            q[a] -= 1;
            let lo = g.face(q, a) as f64;
            let hi = g.face(p, a) as f64;
            speed2 += (0.5 * (lo + hi)).powi(2);
            faces2 += 0.5 * (lo * lo + hi * hi);
            potential -=
                m * g.volume[i] as f64 * gravity[a] as f64 * (p[a] as f64 + 0.5) * g.h[a] as f64;
        }
        kinetic += 0.5 * m * g.volume[i] as f64 * speed2;
        mac += 0.5 * m * g.volume[i] as f64 * faces2;
        let fill = super::swept_extension::contour_fill(g, i, 0.0) as f64;
        phi_area += fill;
        phi_potential -= m
            * fill
            * (gravity[0] as f64 * (p[0] as f64 + 0.5) * g.h[0] as f64
                + gravity[1] as f64 * (p[1] as f64 + 0.5) * g.h[1] as f64);
        phi_kinetic += 0.5 * m * fill * faces2;
        let e = (g.volume[i] - g.capacity[i]).max(0.0) as f64;
        excess += e;
        max_excess = max_excess.max(e);
    }
    serde_json::json!({"kinetic":kinetic,"macKinetic":mac,"potential":potential,"phiKinetic":phi_kinetic,"phiArea":phi_area,"phiPotential":phi_potential,"excess":excess,"maxExcess":max_excess})
}
