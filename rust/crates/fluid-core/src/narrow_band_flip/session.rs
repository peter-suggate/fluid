//! Revisioned interactive owner of the advance-lab's narrow-band FLIP world.
use super::{world::World, NarrowBandFlipOptions};
use crate::{
    publication::{encode_publication, Plane, PlaneId},
    types::ValidationError,
    uniform_geometric::{
        grid::{Grid, LiquidDrop},
        physical::Physical,
    },
    world::Revision,
};
use serde::Deserialize;

pub const METHOD: &str = "uniform-narrow-band-flip";

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Seed {
    pub dimensions: [usize; 2],
    pub cell_size: [f32; 2],
    pub capacity: Vec<f32>,
    pub phi: Vec<f32>,
    pub gravity: [f32; 2],
    pub density: f32,
    pub viscosity: f32,
    pub surface_tension: f32,
    pub open_top: bool,
}
pub struct Session {
    pub world: World,
    revision: Revision,
    initial_volume: f64,
    publication: Vec<u8>,
    samples: Vec<f32>,
    halted: bool,
    physical: Physical,
    injected_volume: f64,
}
impl Session {
    pub fn new(
        seed: Seed,
        scene: crate::initial_scene::SceneDocument,
        options: NarrowBandFlipOptions,
        run_epoch: u32,
        command_sequence: u32,
    ) -> Result<Self, ValidationError> {
        let mut grid = Grid::new(seed.dimensions, seed.cell_size, seed.open_top)?;
        grid.capacity = seed.capacity;
        grid.phi = seed.phi;
        let mut world = World::new(
            grid,
            options,
            seed.gravity,
            seed.density,
            seed.viscosity,
            seed.surface_tension,
        )?;
        world.fluid.inflow = scene.fluid.inflow;
        let physical = Physical::new(&scene, &world.fluid.grid)?;
        physical.geometry(&mut world.fluid.grid);
        world.bootstrap();
        Ok(Self {
            initial_volume: world.receipt.volume,
            world,
            revision: Revision {
                schema_version: 1,
                dimension: 2,
                run_epoch,
                command_sequence,
                frame: 0,
                time: 0.0,
                injections: 0,
                topology_generation: 1,
                field_revision: 1,
                surface_revision: 1,
                memory_epoch: 1,
            },
            publication: Vec::new(),
            samples: Vec::new(),
            halted: false,
            physical,
            injected_volume: 0.0,
        })
    }
    fn ordered(&self, sequence: u32, epoch: u32) -> Result<(), ValidationError> {
        if epoch != self.revision.run_epoch || sequence <= self.revision.command_sequence {
            return Err(ValidationError("stale narrow-band FLIP command".into()));
        }
        Ok(())
    }
    pub fn advance(&mut self, sequence: u32, dt: f64) -> Result<(), ValidationError> {
        self.ordered(sequence, self.revision.run_epoch)?;
        if self.halted {
            return Err(ValidationError(
                "narrow-band FLIP run halted; reset before advancing".into(),
            ));
        }
        if !dt.is_finite() || dt <= 0.0 || dt > 1.0 {
            return Err(ValidationError("invalid narrow-band FLIP timestep".into()));
        }
        self.physical.geometry(&mut self.world.fluid.grid);
        if let Err(error) = self.world.advance(dt as f32) {
            self.halted = true;
            return Err(error);
        }
        self.injected_volume += self.world.receipt.injected_volume;
        if self.world.options.rigid_coupling == "on" {
            if let Err(error) = self.physical.advance(&self.world.fluid.grid, dt as f32) {
                self.halted = true;
                return Err(error);
            }
        }
        self.revision.command_sequence = sequence;
        self.revision.frame = self.world.receipt.frame;
        self.revision.time = self.world.receipt.time;
        self.revision.field_revision += 1;
        self.revision.surface_revision += 1;
        Ok(())
    }
    pub fn apply_command(
        &mut self,
        sequence: u32,
        epoch: u32,
        command: serde_json::Value,
    ) -> Result<(), ValidationError> {
        self.ordered(sequence, epoch)?;
        let grid = &mut self.world.fluid.grid;
        match command.get("type").and_then(|v| v.as_str()) {
            Some("snapshot") => {}
            Some("inject-liquid") => {
                let drop: LiquidDrop = serde_json::from_value(
                    command
                        .get("drop")
                        .cloned()
                        .ok_or_else(|| ValidationError("Missing liquid drop".into()))?,
                )
                .map_err(|e| ValidationError(e.to_string()))?;
                if !drop.radius_m.is_finite()
                    || drop.radius_m <= 0.0
                    || drop.centre_m.iter().any(|v| !v.is_finite())
                    || grid.drops.len() >= 64
                {
                    return Err(ValidationError(
                        "Invalid liquid drop or pending drop budget exceeded".into(),
                    ));
                }
                grid.drops.push(drop);
                self.revision.injections += 1;
            }
            Some("remove-rigid-body") => {
                let id = command
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ValidationError("Missing body id".into()))?;
                self.physical.bodies.retain(|b| b.description.id != id);
                self.physical.geometry(grid);
                self.revision.field_revision += 1;
            }
            Some("set-rigid-pose") => {
                let id = command
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ValidationError("Missing body id".into()))?;
                let position: crate::scene_model::Vec3 = serde_json::from_value(
                    command
                        .get("position_m")
                        .cloned()
                        .ok_or_else(|| ValidationError("Missing body position".into()))?,
                )
                .map_err(|e| ValidationError(e.to_string()))?;
                let velocity: crate::scene_model::Vec3 = serde_json::from_value(
                    command
                        .get("velocity_m_s")
                        .cloned()
                        .ok_or_else(|| ValidationError("Missing body velocity".into()))?,
                )
                .map_err(|e| ValidationError(e.to_string()))?;
                if position
                    .array()
                    .iter()
                    .chain(velocity.array().iter())
                    .any(|v| !v.is_finite())
                {
                    return Err(ValidationError("Invalid rigid pose".into()));
                }
                let body = self
                    .physical
                    .bodies
                    .iter_mut()
                    .find(|b| b.description.id == id)
                    .ok_or_else(|| ValidationError("Unknown rigid body".into()))?;
                body.position_m = position;
                body.position_m.z = 0.0;
                body.linear_velocity_m_s = velocity;
                body.linear_velocity_m_s.z = 0.0;
                body.held = command
                    .get("held")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                self.physical.geometry(grid);
                self.revision.field_revision += 1;
            }
            Some("add-rigid-body") => {
                let body = serde_json::from_value(
                    command
                        .get("body")
                        .cloned()
                        .ok_or_else(|| ValidationError("Missing rigid body".into()))?,
                )
                .map_err(|e| ValidationError(e.to_string()))?;
                self.physical.add(body)?;
                if command.get("held").and_then(|v| v.as_bool()) == Some(true) {
                    if let Some(b) = self.physical.bodies.last_mut() {
                        b.held = true;
                    }
                }
                self.physical.geometry(grid);
                self.revision.field_revision += 1;
            }
            _ => {
                return Err(ValidationError(
                    "Unsupported narrow-band FLIP editing command".into(),
                ))
            }
        }
        self.revision.command_sequence = sequence;
        Ok(())
    }
    pub fn receipt(&self) -> serde_json::Value {
        let mut value = serde_json::to_value(&self.revision).expect("revision is serializable");
        value["method"] = serde_json::json!(METHOD);
        value["narrowBandFlip"] = serde_json::to_value(&self.world.receipt)
            .expect("narrow-band FLIP receipt is serializable");
        value["initialVolume"] = serde_json::json!(self.initial_volume);
        value["injectedVolume"] = serde_json::json!(self.injected_volume);
        value["rigidBodies"] = serde_json::json!(self.physical.bodies);
        value["pendingDrops"] = serde_json::json!(self.world.fluid.grid.drops);
        value["halted"] = serde_json::json!(self.halted);
        value
    }
    pub fn snapshot(&mut self, _view_mask: u32) -> Result<&[u8], ValidationError> {
        let grid = &self.world.fluid.grid;
        let velocity: Vec<f32> = grid.velocity.iter().flatten().copied().collect();
        self.world.particles.publish(&mut self.samples);
        let metadata = serde_json::to_vec(&serde_json::json!({
            "revision": self.revision,
            "receipt": self.receipt(),
            "method": METHOD,
            "dimensions": grid.dims,
            "cellSize": grid.h,
            "options": self.world.options,
        }))
        .map_err(|e| ValidationError(e.to_string()))?;
        encode_publication(
            &metadata,
            &[
                Plane::F32(PlaneId::Density, &grid.volume),
                Plane::F32(PlaneId::Capacity, &grid.capacity),
                Plane::F32(PlaneId::RdfVertices, &grid.phi),
                Plane::F32(PlaneId::CellVelocity, &velocity),
                Plane::F32(PlaneId::Pressure, &self.world.band.pressure),
                Plane::F32(PlaneId::UniformLowX, &grid.low_x),
                Plane::F32(PlaneId::UniformLowY, &grid.low_y),
                Plane::U8(PlaneId::UniformReleased, &grid.released),
                Plane::U8(PlaneId::UniformTiles, &self.world.tiles),
                Plane::F32(PlaneId::UniformParticles, &self.samples),
            ],
            &mut self.publication,
        )
        .map_err(|e| ValidationError(e.into()))?;
        Ok(&self.publication)
    }
}
