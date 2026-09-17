use fluid_core::{
    geometry::BoundaryMode,
    staggered_velocity::{extend_faces, StaggeredVelocity2d},
    topology::{compile_topology, BrickSeed, TopologySeed},
    Fields, Graph, RowKind,
};

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
                resolution: if key % 2 == 0 { left } else { right },
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
        face_velocity: g
            .rows
            .iter()
            .map(|r| sample(r.axis as usize, [r.center[0], r.center[1]]))
            .collect(),
        cell_velocity: vec![12345.0; g.cells.len() * 2],
        capacity: vec![1.0; g.cells.len()],
        ..Default::default()
    }
}

#[test]
fn regular_mac_interpolation_is_bilinear_and_ignores_cell_velocity() {
    let g = grid(8, 8, false);
    let q = |axis: usize, p: [f32; 2]| {
        if axis == 0 {
            2.0 + 0.25 * p[0] * p[1]
        } else {
            -1.0 - 0.5 * p[0] * p[1]
        }
    };
    let v = StaggeredVelocity2d::new(&g, &field(&g, q)).unwrap();
    for y in 2..27 {
        for x in 2..27 {
            let p = [x as f32 * 0.5, y as f32 * 0.5];
            let actual = v.sample(p);
            for a in 0..2 {
                assert!((actual[a] - q(a, p)).abs() < 2e-5, "{p:?}: {actual:?}");
            }
        }
    }
}

#[test]
fn mixed_grid_reproduces_affine_components_including_both_sides_of_t_junctions() {
    for (left, right) in [(4, 8), (8, 4), (2, 4), (4, 2)] {
        let g = grid(left, right, false);
        let q = |a: usize, p: [f32; 2]| {
            if a == 0 {
                1.0 + 0.3 * p[0] - 0.7 * p[1]
            } else {
                -2.0 + 0.2 * p[0] + 0.4 * p[1]
            }
        };
        let v = StaggeredVelocity2d::new(&g, &field(&g, q)).unwrap();
        for y in 16..49 {
            for x in 16..49 {
                let p = [x as f32 * 0.25, y as f32 * 0.25];
                let actual = v.sample(p);
                for a in 0..2 {
                    assert!(
                        (actual[a] - q(a, p)).abs() < 3e-6,
                        "rungs {left}/{right}, {p:?}: {actual:?}, expected {}",
                        q(a, p)
                    );
                }
            }
        }
    }
}

#[test]
fn free_slip_boundary_preserves_tangent_and_enforces_normal_without_half_cell_clamp() {
    let g = grid(4, 8, true);
    let f = field(&g, |a, p| if a == 0 { 3.0 } else { 0.5 * p[1] });
    let v = StaggeredVelocity2d::new(&g, &f).unwrap();
    for x in 12..53 {
        let p = [x as f32 * 0.25, 0.0];
        let value = v.sample(p);
        assert!(
            (value[0] - 3.0).abs() < 1e-6 && value[1].abs() < 1e-6,
            "{p:?}: {value:?}"
        );
        let p = [p[0], 0.1];
        let value = v.sample(p);
        assert!(
            (value[1] - 0.05).abs() < 1e-6,
            "wall-normal affine field was flattened: {value:?}"
        );
    }
}

#[test]
fn extension_preserves_liquid_faces_and_does_not_use_volume_or_cell_velocity() {
    let g = grid(4, 8, true);
    let mut f = field(&g, |a, p| {
        if a == 0 && p[0] >= 4.0 && p[0] <= 12.0 {
            7.0
        } else {
            0.0
        }
    });
    f.density = vec![100.0; g.cells.len()];
    let phi: Vec<_> = g
        .cells
        .iter()
        .map(|c| {
            if (4.0..12.0).contains(&c.center[0]) {
                -1.0
            } else {
                1.0
            }
        })
        .collect();
    let before = f.face_velocity.clone();
    extend_faces(&g, &mut f, &phi, 8).unwrap();
    for r in &g.rows {
        let seed = r.terms.iter().any(|t| phi[t.cell_id as usize] <= 0.0);
        if seed || r.kind == RowKind::ClosedWorld {
            assert_eq!(f.face_velocity[r.id as usize], before[r.id as usize]);
        }
        if r.axis == 0 && r.kind != RowKind::ClosedWorld {
            assert!(
                (f.face_velocity[r.id as usize] - 7.0).abs() < 1e-6,
                "face {:?}: {}",
                r.center,
                f.face_velocity[r.id as usize]
            );
        }
    }
}

#[test]
fn invalid_face_fields_are_rejected_instead_of_falling_back_to_cells() {
    let g = grid(8, 8, false);
    let mut f = field(&g, |_, _| 0.0);
    f.face_velocity[0] = f32::NAN;
    assert!(StaggeredVelocity2d::new(&g, &f).is_err());
    f.face_velocity.clear();
    assert!(StaggeredVelocity2d::new(&g, &f).is_err());
}

#[test]
fn reflected_mixed_layout_samples_the_reflected_face_field() {
    let left = grid(4, 8, true);
    let right = grid(8, 4, true);
    let q = |a: usize, p: [f32; 2]| {
        if a == 0 {
            (p[0] * 0.19).sin() * (p[1] * 0.31).cos()
        } else {
            (p[0] * 0.11).cos() * (p[1] * 0.27).sin()
        }
    };
    let a = StaggeredVelocity2d::new(&left, &field(&left, q)).unwrap();
    let b = StaggeredVelocity2d::new(
        &right,
        &field(&right, |axis, p| {
            (if axis == 0 { -1.0 } else { 1.0 }) * q(axis, [16.0 - p[0], p[1]])
        }),
    )
    .unwrap();
    for y in 0..65 {
        for x in 0..65 {
            let p = [x as f32 * 0.25, y as f32 * 0.25];
            let u = a.sample(p);
            let v = b.sample([16.0 - p[0], p[1]]);
            assert!(
                (u[0] + v[0]).abs() < 2e-6 && (u[1] - v[1]).abs() < 2e-6,
                "{p:?}: {u:?} vs {v:?}"
            );
        }
    }
}

#[test]
fn next_frame_advects_from_the_retained_precoarsening_face_field() {
    use fluid_core::world::{TransportExperiment, World, WorldOptions};
    let seed: serde_json::Value = serde_json::from_str(include_str!(
        "../../../core/testdata/quarter-pool-transition-seed.json"
    ))
    .unwrap();
    let mut w = World::from_document(
        serde_json::from_value(seed["scene"].clone()).unwrap(),
        serde_json::from_value(serde_json::json!({"dtS":1.0/30.0,"timeStep":"paper"})).unwrap(),
        WorldOptions {
            pressure_iterations: 256,
            pressure_relative_tolerance: 1e-6,
            transport_experiment: TransportExperiment::LevelSetVolume,
            ..Default::default()
        },
    )
    .unwrap();
    let mut retained = None;
    let mut checks = 0;
    for frame in 1..=8 {
        let mut before = None;
        let mut coarsened = None;
        w.advance_with_observer(frame, 1.0 / 30.0, |stage, g, f| {
            if stage == "face-preparation" {
                if let Some(source) = &retained {
                    let source: &StaggeredVelocity2d = source;
                    for r in &g.rows {
                        if r.kind == RowKind::ClosedWorld
                            || r.open_fraction != 1.0
                            || (0..2)
                                .any(|a| r.center[a] < 2.0 || r.center[a] > g.dimensions[a] - 2.0)
                        {
                            continue;
                        }
                        let expected = source
                            .sample(source.trace([r.center[0], r.center[1]], 1.0 / 30.0))
                            [r.axis as usize];
                        assert!(
                            (f.face_velocity[r.id as usize] - expected).abs() < 2e-6,
                            "frame {frame}, {:?}: {} != retained source {expected}",
                            r.center,
                            f.face_velocity[r.id as usize]
                        );
                        checks += 1;
                    }
                }
            }
            if stage == "resolution-transfer-before" {
                before = Some((g.clone(), StaggeredVelocity2d::new(g, f).unwrap()));
            }
            if stage == "resolution-transfer-after" {
                if let Some((old, source)) = before.take() {
                    if g.cells.iter().any(|c| {
                        old.cells
                            .iter()
                            .any(|o| o.brick_key == c.brick_key && o.widths[0] < c.widths[0])
                    }) {
                        coarsened = Some(source);
                    }
                }
            }
        })
        .unwrap();
        retained = coarsened;
        if checks > 20 {
            break;
        }
    }
    assert!(
        checks > 20,
        "fixture did not exercise advection after accepted coarsening"
    );
}

#[test]
fn split_dam_front_improves_before_wall_contact_and_tracks_mirrored_regions() {
    use fluid_core::world::{Command, TransportExperiment, World, WorldOptions};
    let seed: serde_json::Value = serde_json::from_str(include_str!(
        "../../../core/testdata/split-resolution-ladder-seed.json"
    ))
    .unwrap();
    let run = |left: u8, right: u8| {
        let mut world = World::from_document(
            serde_json::from_value(seed["scene"].clone()).unwrap(),
            serde_json::from_value(serde_json::json!({"dtS":1.0/30.0,"timeStep":"paper"})).unwrap(),
            WorldOptions {
                pressure_iterations: 256,
                pressure_relative_tolerance: 1e-6,
                transport_experiment: TransportExperiment::LevelSetVolume,
                ..Default::default()
            },
        )
        .unwrap();
        world.apply_command(1,1,serde_json::from_value::<Command>(serde_json::json!({"type":"set-refinement-regions","regions":[
            {"minimumFine":[0,0],"maximumFine":[16,16],"minimumCellWidth":left,"maximumCellWidth":left},
            {"minimumFine":[16,0],"maximumFine":[32,16],"minimumCellWidth":right,"maximumCellWidth":right}
        ]})).unwrap()).unwrap();
        let mut result = Vec::new();
        for frame in 1..=6 {
            world.advance(frame + 1, 1.0 / 30.0).unwrap();
            let mass: f64 = world
                .state
                .topology
                .graph
                .cells
                .iter()
                .map(|c| c.measure as f64 * world.state.fields.density[c.id as usize] as f64)
                .sum();
            assert!((mass - 128.0).abs() < 1e-8, "frame {frame}: mass {mass}");
            let bytes = world.snapshot(2).unwrap();
            let word =
                |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
            let header = (0..word(16))
                .map(|i| 32 + 16 * i)
                .find(|&i| word(i) == 32)
                .unwrap();
            let offset = word(header + 8);
            let count = word(header + 12);
            let mut bounds = [f32::INFINITY, f32::NEG_INFINITY];
            for i in (0..count).step_by(2) {
                let x = f32::from_le_bytes(
                    bytes[offset + 4 * i..offset + 4 * i + 4]
                        .try_into()
                        .unwrap(),
                );
                bounds = [bounds[0].min(x), bounds[1].max(x)];
            }
            result.push(bounds);
        }
        result
    };
    let left = run(2, 1);
    let right = run(1, 2);
    assert!(
        8.0 - left[0][0] > 0.32,
        "coarse front lost its face velocity: {:?}",
        left[0]
    );
    // Frame 6 now enters the wall-contact continuation band on the fine side.
    // Measure the interior front at frame 5, before that boundary operation;
    // retain frame 6 below as a reflected-contact check.
    let last = left[4];
    let gap = last[0] + last[1] - 32.0;
    assert!(last[0] > 0.0 && last[1] < 32.0);
    assert!(
        (0.0..1.1).contains(&gap),
        "coarse/fine pre-impact gap: {gap}"
    );
    for (a, b) in left.iter().zip(&right) {
        assert!(
            (a[0] + b[1] - 32.0).abs() < 1e-3 && (a[1] + b[0] - 32.0).abs() < 1e-3,
            "held-region reflection changed the evolution: {a:?}, {b:?}"
        );
    }
}
