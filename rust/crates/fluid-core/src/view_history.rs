//! Read-only historical lens planes in the finest canvas lattice. Keeping
//! them in this fixed address space preserves observations across regridding.
use crate::resolution::ResolutionPolicyReceipt;
use crate::scene::SceneState;
#[derive(Default)]
pub struct ViewHistory {
    pub density_before: Vec<f32>,
    pub velocity_x_before: Vec<f32>,
    pub velocity_y_before: Vec<f32>,
    pub brick_resolution_before: Vec<u8>,
    pub brick_activity: Vec<f32>,
}
impl ViewHistory {
    pub fn new(state: &SceneState<2>) -> Self {
        let [nx, ny, _] = state.description.dimensions.map(|v| v as usize);
        let mut view = Self {
            density_before: vec![0.0; nx * ny],
            velocity_x_before: vec![0.0; (nx + 1) * ny],
            velocity_y_before: vec![0.0; nx * (ny + 1)],
            brick_resolution_before: vec![0; nx.div_ceil(8) * ny.div_ceil(8)],
            brick_activity: vec![0.0; nx.div_ceil(8) * ny.div_ceil(8)],
        };
        view.capture_density(state);
        view
    }
    pub fn capture_density(&mut self, state: &SceneState<2>) {
        let [nx, ny, _] = state.description.dimensions.map(|v| v as usize);
        self.density_before.fill(0.0);
        for cy in 0..ny {
            for x in 0..nx {
                if let Some(owner) = crate::numerics::owner_at(
                    &state.topology.graph,
                    [x as f32 + 0.5, (ny - cy) as f32 - 0.5, 0.0],
                ) {
                    self.density_before[cy * nx + x] = state.fields.density[owner];
                }
            }
        }
    }
    pub fn capture_faces(&mut self, state: &SceneState<2>, h: f64) {
        let [nx, ny, _] = state.description.dimensions.map(|v| v as i32);
        self.velocity_x_before.fill(0.0);
        self.velocity_y_before.fill(0.0);
        for row in &state.topology.graph.rows {
            let value = state.fields.face_velocity[row.id as usize] as f64 * h;
            if row.axis == 0 {
                let x = (row.center[0] as f64 + 0.5).floor() as i32;
                let cy = ny - 1 - row.center[1].floor() as i32;
                if x >= 0 && x <= nx && cy >= 0 && cy < ny {
                    self.velocity_x_before[(cy * (nx + 1) + x) as usize] = value as f32;
                }
            } else {
                let x = row.center[0].floor() as i32;
                let cy = ny - (row.center[1] as f64 + 0.5).floor() as i32;
                if x >= 0 && x < nx && cy >= 0 && cy <= ny {
                    self.velocity_y_before[(cy * nx + x) as usize] = -value as f32;
                }
            }
        }
    }
    pub fn capture_bricks(&mut self, state: &SceneState<2>) {
        let bx = state.description.dimensions[0].div_ceil(8) as i32;
        let by = state.description.dimensions[1].div_ceil(8) as i32;
        self.brick_resolution_before.fill(0);
        for b in &state.topology.bricks {
            let x = b.seed.coordinate[0];
            let cy = by - 1 - b.seed.coordinate[1];
            if x >= 0 && x < bx && cy >= 0 && cy < by {
                self.brick_resolution_before[(cy * bx + x) as usize] = if b.seed.active {
                    [1, 2, 4, 8]
                        .iter()
                        .position(|&r| r == b.seed.resolution)
                        .unwrap_or(0) as u8
                } else {
                    0
                };
            }
        }
    }
    pub fn capture_activity(&mut self, state: &SceneState<2>, receipt: &ResolutionPolicyReceipt) {
        let bx = state.description.dimensions[0].div_ceil(8) as i32;
        let by = state.description.dimensions[1].div_ceil(8) as i32;
        for record in &receipt.bricks {
            if let Some(b) = state
                .topology
                .bricks
                .iter()
                .find(|b| b.seed.key == record.brick_key)
            {
                let x = b.seed.coordinate[0];
                let cy = by - 1 - b.seed.coordinate[1];
                if x >= 0 && x < bx && cy >= 0 && cy < by {
                    self.brick_activity[(cy * bx + x) as usize] = record.score_byte as f32 / 255.0;
                }
            }
        }
    }
}
