//! Tracer physics shares the fluid's accepted transport characteristic.
use crate::{owner_at, trace_effective_transport_arrival, Fields, Graph, ValidationError};
use serde::{Deserialize, Serialize};

pub const TRACER_BUDGET: usize = 1_048_576;
pub const TRACER_DENSITY: f64 = 4.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TracerLattice {
    pub dimensions: [usize; 3],
    pub count: usize,
    pub origin_fine: [f64; 3],
    pub spacing_fine: f64,
}
impl TracerLattice {
    pub fn new(dimensions: [u32; 3], budget: usize) -> Self {
        let empty = Self {
            dimensions: [0; 3],
            count: 0,
            origin_fine: [0.0; 3],
            spacing_fine: 1.0,
        };
        if budget == 0 {
            return empty;
        }
        let domain = dimensions.map(|v| v.max(1) as f64);
        let mut spacing = (1.0 / TRACER_DENSITY)
            .cbrt()
            .max((domain[0] * domain[1] * domain[2] / budget as f64).cbrt());
        let extent = |s: f64| domain.map(|v| (v / s).floor().max(1.0) as usize);
        let mut dims = extent(spacing);
        let count = |d: [usize; 3]| {
            d[0].checked_mul(d[1])
                .and_then(|v| v.checked_mul(d[2]))
                .unwrap_or(usize::MAX)
        };
        for _ in 0..64 {
            if count(dims) <= budget {
                break;
            }
            spacing *= 1.05;
            dims = extent(spacing)
        }
        let count = count(dims);
        if count > budget {
            return empty;
        }
        Self {
            dimensions: dims,
            count,
            origin_fine: std::array::from_fn(|a| 0.5 * (domain[a] - dims[a] as f64 * spacing)),
            spacing_fine: spacing,
        }
    }
    fn position(&self, index: usize) -> [f32; 3] {
        let mut q = index;
        std::array::from_fn(|a| {
            let i = q % self.dimensions[a].max(1);
            q /= self.dimensions[a].max(1);
            (self.origin_fine[a] + (i as f64 + 0.5) * self.spacing_fine) as f32
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tracers {
    pub lattice: TracerLattice,
    /// Literal vec4 x/y/z/live state. Retirement preserves stable marker IDs.
    pub state: Vec<f32>,
    pub enabled: bool,
    pub seed_pending: bool,
    pub generation: u32,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TracerReceipt {
    pub generation: u32,
    pub seeded: bool,
    pub count: usize,
    pub live_count: usize,
    pub retired_count: usize,
}
impl Tracers {
    pub fn new(dimensions: [u32; 3], budget: usize) -> Self {
        let lattice = TracerLattice::new(dimensions, budget);
        Self {
            state: vec![0.0; 4 * lattice.count],
            lattice,
            enabled: false,
            seed_pending: false,
            generation: 0,
        }
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        if self.enabled != enabled {
            self.enabled = enabled;
            self.seed_pending = enabled
        }
    }
    pub fn reseed(&mut self) {
        if self.enabled {
            self.seed_pending = true
        }
    }
    pub fn advance(
        &mut self,
        graph: &Graph,
        fields: &Fields,
        dt: f32,
    ) -> Result<TracerReceipt, ValidationError> {
        fields.validate_for(graph)?;
        let mut receipt = TracerReceipt {
            generation: self.generation,
            count: self.lattice.count,
            ..TracerReceipt::default()
        };
        if !self.enabled || self.lattice.count == 0 {
            return Ok(receipt);
        }
        if !(dt.is_finite() && dt > 0.0) {
            return Err(ValidationError(
                "tracer dt must be positive and finite".into(),
            ));
        }
        receipt.seeded = self.seed_pending;
        if self.seed_pending {
            for index in 0..self.lattice.count {
                let mut p = self.lattice.position(index);
                if graph.dimension == 2 {
                    p[2] = 0.5
                }
                let owner = owner_at(graph, p.map(|v| v.floor() + 0.5));
                let at = 4 * index;
                self.state[at..at + 3].copy_from_slice(&p);
                self.state[at + 3] = if owner.is_some_and(|i| fields.density[i] > 0.5) {
                    1.0
                } else {
                    0.0
                };
            }
        }
        // Fresh seeds advance in the same pass, matching resident semantics.
        for index in 0..self.lattice.count {
            let at = 4 * index;
            if self.state[at + 3] < 0.5 {
                continue;
            }
            let p = trace_effective_transport_arrival(
                graph,
                fields,
                [
                    self.state[at],
                    self.state[at + 1],
                    if graph.dimension == 2 {
                        0.5
                    } else {
                        self.state[at + 2]
                    },
                ],
                dt,
            );
            self.state[at..at + graph.dimension as usize]
                .copy_from_slice(&p[..graph.dimension as usize]);
            let owner_position = [
                p[0].floor() + 0.5,
                p[1].floor() + 0.5,
                if graph.dimension == 2 {
                    0.5
                } else {
                    p[2].floor() + 0.5
                },
            ];
            let cell = owner_at(graph, owner_position);
            if cell.is_none_or(|i| fields.density[i] < 1e-5) {
                self.state[at + 3] = 0.0;
                receipt.retired_count += 1
            } else {
                receipt.live_count += 1
            }
        }
        self.seed_pending = false;
        self.generation += 1;
        receipt.generation = self.generation;
        Ok(receipt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::BoundaryMode;
    use crate::topology::{compile_topology, BrickSeed, TopologySeed};
    #[test]
    fn lattice_is_isotropic_centered_and_budgeted() {
        let a = TracerLattice::new([96, 40, 1], 4096);
        assert!(a.count <= 4096);
        for axis in 0..3 {
            let domain = [96.0, 40.0, 1.0][axis];
            assert!(
                (2.0 * a.origin_fine[axis] + a.dimensions[axis] as f64 * a.spacing_fine - domain)
                    .abs()
                    < 1e-12
            )
        }
        assert_eq!(TracerLattice::new([96, 40, 1], 0).count, 0);
    }
    #[test]
    fn reenable_requests_reseed_but_repeated_enable_does_not() {
        let mut t = Tracers::new([8, 8, 1], 100);
        t.set_enabled(true);
        assert!(t.seed_pending);
        t.seed_pending = false;
        t.set_enabled(true);
        assert!(!t.seed_pending);
        t.set_enabled(false);
        t.reseed();
        assert!(!t.seed_pending);
        t.set_enabled(true);
        assert!(t.seed_pending);
    }

    #[test]
    fn three_dimensional_markers_use_the_fluid_characteristic() {
        let graph = compile_topology::<3>(TopologySeed {
            dimensions: [8; 3],
            generation: 1,
            sparse_air_phi: 0.5,
            boundaries: [BoundaryMode::Closed; 6],
            bricks: vec![BrickSeed {
                id: 0,
                key: 0,
                coordinate: [0; 3],
                span_bricks: 1,
                resolution: 8,
                active: true,
                density: vec![],
                gamma: vec![],
                refinement_region_scale: None,
            }],
        })
        .unwrap()
        .graph;
        let n = graph.cells.len();
        let mut fields = Fields {
            density: vec![1.0; n],
            gamma: vec![1.0; n],
            capacity: vec![1.0; n],
            pressure: vec![0.0; n],
            pressure_rhs: vec![0.0; n],
            pressure_diagonal: vec![0.0; n],
            pressure_member: vec![0; n],
            extension_depth: vec![0; n],
            interface_offset: vec![0.0; n],
            interface_normal: vec![0.0; 3 * n],
            cell_velocity: vec![0.0; 3 * n],
            face_velocity: vec![0.0; graph.rows.len()],
            ..Fields::default()
        };
        for velocity in fields.cell_velocity.chunks_exact_mut(3) {
            velocity.copy_from_slice(&[0.0, 0.0, 0.25]);
        }
        let mut tracers = Tracers::new([8; 3], 64);
        tracers.set_enabled(true);
        let initial_z = tracers.lattice.position(21)[2];
        let receipt = tracers.advance(&graph, &fields, 0.4).unwrap();
        assert!(receipt.seeded);
        assert_eq!(receipt.live_count, tracers.lattice.count);
        assert_eq!(tracers.state[4 * 21 + 2], initial_z + 0.1);
    }
}
