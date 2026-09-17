use fluid_core::world::{Command, TransportExperiment, World, WorldOptions};

#[test]
fn split_dam_contacts_dry_walls_without_persistent_fine_side_overload() {
    let seed: serde_json::Value = serde_json::from_str(include_str!(
        "../../../core/testdata/split-resolution-ladder-seed.json"
    ))
    .unwrap();
    for (left, right) in [(1, 2), (2, 1)] {
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
        world.apply_command(1, 1, serde_json::from_value::<Command>(serde_json::json!({
            "type":"set-refinement-regions","regions":[
                {"minimumFine":[0,0],"maximumFine":[16,16],"minimumCellWidth":left,"maximumCellWidth":left},
                {"minimumFine":[16,0],"maximumFine":[32,16],"minimumCellWidth":right,"maximumCellWidth":right}
            ]
        })).unwrap()).unwrap();
        for frame in 1..=120 {
            world.advance(frame + 1, 1.0 / 30.0).unwrap();
            assert!(
                world.state.fields.fault.is_none(),
                "frame {frame}: numerical fault"
            );
            let snapshot = if frame == 120 {
                Some(world.snapshot(2).unwrap().to_vec())
            } else {
                None
            };
            let graph = &world.state.topology.graph;
            let fields = &world.state.fields;
            let volume: f64 = graph
                .cells
                .iter()
                .map(|c| c.measure as f64 * fields.density[c.id as usize] as f64)
                .sum();
            assert!(
                (volume - 128.0).abs() < 1e-5,
                "frame {frame}, split {left}/{right}, volume {volume}"
            );
            if frame == 7 || frame == 8 {
                let corner = graph
                    .cells
                    .iter()
                    .find(|c| {
                        c.center[1] == 0.5 && c.center[0] == if left == 1 { 0.5 } else { 31.5 }
                    })
                    .unwrap()
                    .id as usize;
                assert!(
                    world.level_set_phi[corner] < 0.0,
                    "frame {frame}: false wall-air gap delayed contact on split {left}/{right}"
                );
                if frame == 8 {
                    assert_eq!(
                        fields.pressure_member[corner], 1,
                        "contact must enter the next pressure solve"
                    );
                    assert!(
                        fields.density[corner] < 1.1,
                        "fine cell overloaded before wall pressure responded"
                    );
                }
            }
            if frame == 120 {
                // The level set must not be compressed by divergent air
                // extension before sharpening gets a chance to return V.
                let bytes = snapshot.as_ref().unwrap();
                let word =
                    |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
                let offset = (0..word(16))
                    .map(|i| 32 + 16 * i)
                    .find(|&i| word(i) == 31)
                    .unwrap();
                let start = word(offset + 8);
                let vertices = (0..word(offset + 12))
                    .map(|i| {
                        f32::from_le_bytes(
                            bytes[start + 4 * i..start + 4 * i + 4].try_into().unwrap(),
                        )
                    })
                    .collect();
                let surface =
                    fluid_core::levelset_surface::publish([32, 16], vertices, 128.0).unwrap();
                let fill = fluid_core::levelset_surface::implied_fill_fine_cells(&surface).unwrap();
                let mut positive = [0.0_f64; 2];
                for cell in &graph.cells {
                    let mut target = 0.0;
                    for y in cell.minimum[1] as usize..cell.maximum[1] as usize {
                        for x in cell.minimum[0] as usize..cell.maximum[0] as usize {
                            target += fill[x + 32 * y] as f64;
                        }
                    }
                    let actual = fields.density[cell.id as usize] as f64 * cell.measure as f64;
                    positive[usize::from(cell.center[0] >= 16.0)] += (actual - target).max(0.0);
                }
                assert!(
                    positive.iter().all(|&v| v < 4.0),
                    "split {left}/{right}: air extension created diffuse residue {positive:?}"
                );
                assert!(
                    (positive[0] - positive[1]).abs() < 0.75,
                    "split {left}/{right}: coarse/fine residue diverged {positive:?}"
                );
            }
            if [20, 60, 120].contains(&frame) {
                for right_wall in [false, true] {
                    let excess: f64 = graph
                        .cells
                        .iter()
                        .filter(|c| {
                            if right_wall {
                                c.maximum[0] > 30.0
                            } else {
                                c.minimum[0] < 2.0
                            }
                        })
                        .map(|c| {
                            ((fields.density[c.id as usize] - fields.capacity[c.id as usize])
                                as f64)
                                .max(0.0)
                                * c.measure as f64
                        })
                        .sum();
                    assert!(
                        excess < 0.02,
                        "frame {frame}, split {left}/{right}: wall overload {excess}"
                    );
                }
            }
        }
    }
}
