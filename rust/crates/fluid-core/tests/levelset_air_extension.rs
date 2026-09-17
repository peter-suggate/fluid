use fluid_core::geometry::BoundaryMode;
use fluid_core::levelset_air_extension::project_air_extension;
use fluid_core::staggered_velocity::extend_faces;
use fluid_core::topology::{compile_topology, BrickSeed, TopologySeed};
use fluid_core::{Fields, Graph, RowKind};

fn grid(left: u8, right: u8, closed: bool) -> Graph {
    compile_topology::<2>(TopologySeed {
        dimensions: [16, 16, 1],
        generation: 1,
        sparse_air_phi: 0.5,
        boundaries: [if closed {
            BoundaryMode::Closed
        } else {
            BoundaryMode::Open
        }; 6],
        bricks: (0..4)
            .map(|key| BrickSeed {
                id: key,
                key,
                coordinate: [(key % 2) as i32, (key / 2) as i32, 0],
                span_bricks: 1,
                resolution: 8 / if key % 2 == 0 { left } else { right },
                active: true,
                density: vec![],
                gamma: vec![],
                refinement_region_scale: None,
            })
            .collect(),
    })
    .unwrap()
    .graph
}
fn field(g: &Graph, sample: impl Fn(usize, [f32; 2]) -> f32) -> Fields {
    Fields {
        capacity: vec![1.0; g.cells.len()],
        face_velocity: g
            .rows
            .iter()
            .map(|r| sample(r.axis as usize, [r.center[0], r.center[1]]))
            .collect(),
        ..Default::default()
    }
}
#[test]
fn extension_closes_air_divergence_on_every_rung_without_changing_liquid_faces() {
    for (left, right) in [(1, 1), (2, 2), (4, 4), (1, 2), (2, 1)] {
        let g = grid(left, right, false);
        let phi: Vec<_> = g.cells.iter().map(|c| c.center[1] - 4.25).collect();
        let mut f = field(&g, |a, p| {
            if a == 0 {
                p[0] * p[1]
            } else {
                -0.5 * p[1] * p[1]
            }
        });
        extend_faces(&g, &mut f, &phi, 8).unwrap();
        let before = f.face_velocity.clone();
        let r = project_air_extension(&g, &mut f, &phi).unwrap();
        assert!(r.maximum_initial_divergence > 0.1, "{left}/{right}: {r:?}");
        assert!(
            r.maximum_final_divergence_error < 2e-5,
            "{left}/{right}: {r:?}"
        );
        assert_eq!(r.maximum_compatible_divergence, 0.0);
        for row in &g.rows {
            if row.terms.iter().any(|t| phi[t.cell_id as usize] <= 0.0)
                || row.kind == RowKind::ClosedWorld
            {
                assert_eq!(
                    before[row.id as usize], f.face_velocity[row.id as usize],
                    "solved face changed"
                );
            }
        }
        let once = f.face_velocity.clone();
        project_air_extension(&g, &mut f, &phi).unwrap();
        assert!(once
            .iter()
            .zip(&f.face_velocity)
            .all(|(a, b)| (a - b).abs() < 2e-5));
    }
}
#[test]
fn stationary_and_liquid_only_fields_are_unchanged() {
    let g = grid(1, 2, true);
    let mut f = field(&g, |_, _| 0.0);
    let before = f.face_velocity.clone();
    project_air_extension(
        &g,
        &mut f,
        &g.cells
            .iter()
            .map(|c| c.center[1] - 4.0)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(before, f.face_velocity);
    let mut f = field(&g, |a, p| p[a]);
    let before = f.face_velocity.clone();
    project_air_extension(&g, &mut f, &vec![-1.0; g.cells.len()]).unwrap();
    assert_eq!(before, f.face_velocity);
}
#[test]
fn closed_air_component_retains_imposed_net_flux_and_prescribed_wall_faces() {
    let g = grid(1, 2, true);
    let mut f = field(&g, |a, p| if a == 0 && p[0] == 0.0 { 1.0 } else { 0.0 });
    let before = f.face_velocity.clone();
    let r = project_air_extension(&g, &mut f, &vec![0.5; g.cells.len()]).unwrap();
    assert!(
        (r.maximum_compatible_divergence - 1.0 / 16.0).abs() < 1e-12,
        "{r:?}"
    );
    assert!(r.maximum_final_divergence_error < 2e-6, "{r:?}");
    for row in &g.rows {
        if row.kind == RowKind::ClosedWorld {
            assert_eq!(before[row.id as usize], f.face_velocity[row.id as usize]);
        }
    }
}
#[test]
fn mixed_air_projection_reflects_with_the_scene() {
    let a = grid(1, 2, true);
    let b = grid(2, 1, true);
    let q = |axis: usize, p: [f32; 2]| {
        if axis == 0 {
            (p[0] * 0.19).sin() * (p[1] * 0.31).cos()
        } else {
            (p[0] * 0.11).cos() * (p[1] * 0.27).sin()
        }
    };
    let mut fa = field(&a, q);
    let mut fb = field(&b, |axis, p| {
        (if axis == 0 { -1.0 } else { 1.0 }) * q(axis, [16.0 - p[0], p[1]])
    });
    for (g, f) in [(&a, &mut fa), (&b, &mut fb)] {
        let phi: Vec<_> = g.cells.iter().map(|c| c.center[1] - 4.25).collect();
        extend_faces(g, f, &phi, 8).unwrap();
        project_air_extension(g, f, &phi).unwrap();
    }
    for r in &a.rows {
        let reflected = b
            .rows
            .iter()
            .find(|s| {
                s.axis == r.axis
                    && s.center[0] == 16.0 - r.center[0]
                    && s.center[1] == r.center[1]
                    && s.measure == r.measure
            })
            .unwrap();
        let sign = if r.axis == 0 { -1.0 } else { 1.0 };
        assert!(
            (fa.face_velocity[r.id as usize] - sign * fb.face_velocity[reflected.id as usize])
                .abs()
                < 2e-6
        );
    }
}

#[test]
fn partial_apertures_project_fluid_flux_without_changing_solid_faces() {
    let mut g = grid(1, 2, false);
    for row in &mut g.rows {
        if row.center[1] > 4.0 && row.axis == 0 {
            row.open_fraction = 0.25;
            row.solid_velocity = 0.5;
        }
    }
    let phi: Vec<_> = g.cells.iter().map(|c| c.center[1] - 4.25).collect();
    let mut f = field(&g, |a, p| if a == 0 { p[0] * 0.1 } else { p[1] * 0.2 });
    let before = f.face_velocity.clone();
    let receipt = project_air_extension(&g, &mut f, &phi).unwrap();
    assert!(receipt.maximum_final_divergence_error < 2e-6, "{receipt:?}");
    for row in &g.rows {
        if row.terms.iter().any(|t| phi[t.cell_id as usize] <= 0.0) {
            assert_eq!(before[row.id as usize], f.face_velocity[row.id as usize]);
        }
    }
}
