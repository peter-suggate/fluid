//! Prospective contract, authored BEFORE running the algorithm against it.
//! See docs/COARSENING_2D_CONTRACT.md. Do not fit these budgets to solver output.
use fluid_core::{
    initial_scene::SceneDocument,
    levelset_surface::{implied_fill_fine_cells, publish},
    production_scene::ProductionSceneOptions,
    publication::PlaneId,
    resolution::ResolutionRegion,
    world::{TransportExperiment, World, WorldOptions},
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;

// Also serialize when invoked by plain cargo test: timing arms must not overlap.
static SUITE_LOCK: Mutex<()> = Mutex::new(());
const DT: f64 = 1.0 / 30.0;
const MASS_REL: f64 = 1e-5;
const AREA_REL: f64 = 0.05;
const SHAPE_REL: f64 = 0.10;
const COMPONENT_MIN_AREA: usize = 4;
const QUIET_STEPS: u8 = 6;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Lane { Dam, Circle, Hydro, Pool }
impl Lane {
    fn input(self) -> &'static str {
        match self {
            Self::Dam => include_str!("../../../core/testdata/coarsening-2d/sparse-cm12-long-dam-break.json"),
            Self::Circle => include_str!("../../../core/testdata/coarsening-2d/cm12-figure-2.json"),
            Self::Hydro => include_str!("../../../core/testdata/coarsening-2d/hydrostatic-power-large-offset.json"),
            Self::Pool => include_str!("../../../core/testdata/coarsening-2d/coarse-first-pool-impact-half-slab.json"),
        }
    }
    fn steps(self) -> usize { match self { Self::Dam | Self::Pool => 120, Self::Circle => 90, Self::Hydro => 90 } }
    fn cell_budget(self) -> f64 { match self { Self::Dam => 0.55, Self::Circle => 0.60, Self::Hydro => 0.25, Self::Pool => 0.65 } }
    fn vertex_budget(self) -> f64 { match self { Self::Hydro => 0.35, _ => 0.70 } }
    fn time_budget(self) -> f64 { match self { Self::Hydro => 0.80, _ => 0.90 } }
}

// Accumulate independent failures so one lane run gives useful design feedback.
// A NaN always fails. Never drop an event/ROI merely because it was not observed.
#[derive(Default)]
struct Checks { failures: BTreeMap<String, String> }
impl Checks {
    fn require(&mut self, rule: &str, ok: bool, detail: impl std::fmt::Display) {
        if !ok { self.failures.entry(rule.into()).or_insert_with(|| detail.to_string()); }
    }
    fn upper(&mut self, rule: &str, value: f64, limit: f64, frame: usize) {
        self.require(rule, value.is_finite() && value <= limit,
            format!("frame {frame}: {value:.6} > {limit:.6}"));
    }
    fn lower(&mut self, rule: &str, value: f64, limit: f64, frame: usize) {
        self.require(rule, value.is_finite() && value >= limit,
            format!("frame {frame}: {value:.6} < {limit:.6}"));
    }
}

struct Shot {
    nx: usize, ny: usize,
    fill: Vec<f64>, width: Vec<f64>, velocity: Vec<[f64; 2]>,
    contour: Vec<[f64; 2]>,
    volume: f64, area: f64, centroid: [f64; 2], kinetic: f64,
    cells: usize, vertices: usize, pressure_rows: usize, advance_ns: u64,
    max_speed: f64,
}

fn make_world(lane: Lane, fine: bool) -> World {
    let document: SceneDocument = serde_json::from_str(lane.input()).unwrap();
    let h = document.voxel_domain.finest_cell_size_m;
    let mut production: ProductionSceneOptions = serde_json::from_value(json!({"dtS":DT,"timeStep":"paper"})).unwrap();
    if fine {
        production.atlas.fixed_resolution = Some(8);
        // Keep sparse support live. Freezing topology would prevent a valid fine
        // reference from reaching new regions. Every newly admitted cell is fine.
        production.resolution.refinement_regions = vec![ResolutionRegion {
            minimum_fine: [0.0, 0.0],
            maximum_fine: [document.container.width_m / h, document.container.height_m / h],
            minimum_cell_width: 1, maximum_cell_width: Some(1),
        }];
    }
    World::from_document(document, production, WorldOptions {
        pressure_iterations: 1000, pressure_relative_tolerance: 1e-8,
        tracer_budget: 0, adaptive_sdf: true,
        transport_experiment: TransportExperiment::LevelSetVolume,
        ..Default::default()
    }).unwrap()
}

fn capture(world: &mut World, checks: &mut Checks, frame: usize, fine: bool) -> Shot {
    let nx = world.state.topology.graph.dimensions[0] as usize;
    let ny = world.state.topology.graph.dimensions[1] as usize;
    let bytes = world.snapshot(2).unwrap();
    let word = |at: usize| u32::from_le_bytes(bytes[at..at+4].try_into().unwrap()) as usize;
    let metadata: Value = serde_json::from_slice(&bytes[word(20)..word(20)+word(24)]).unwrap();
    let plane = |id: PlaneId| -> Vec<f32> {
        let at = (0..word(16)).map(|n| 32 + 16*n).find(|&at| word(at) == id as usize).unwrap();
        bytes[word(at+8)..word(at+8)+4*word(at+12)].chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
    };
    let surface = publish([nx as u32, ny as u32], plane(PlaneId::RdfVertices), 0.0).unwrap();
    let fill: Vec<f64> = implied_fill_fine_cells(&surface).unwrap().into_iter().map(f64::from).collect();
    let segments = plane(PlaneId::RdfSegments);
    let contour: Vec<_> = segments.chunks_exact(4).flat_map(|s|
        [[s[0] as f64, s[1] as f64], [(s[0]+s[2]) as f64/2.0, (s[1]+s[3]) as f64/2.0]]) .collect();
    checks.require("adaptive phi is the simulated authority", metadata["sdf"]["adaptive"] == true, frame);
    let vertices = metadata["sdf"]["vertexCount"].as_u64().unwrap() as usize;
    let g = &world.state.topology.graph;
    let mut fields = world.state.fields.clone();
    fluid_core::collocate_velocity(g, &mut fields);
    checks.require("finite physical fields", fields.face_velocity.iter().chain(&fields.density)
        .chain(&world.level_set_phi).all(|v| v.is_finite()), frame);
    checks.require("nonnegative volume", fields.density.iter().all(|&v| v >= -1e-7), frame);
    checks.require("nonempty contour", !contour.is_empty(), frame);
    if frame > 0 {
        checks.require("pressure converged", world.pressure.converged, format!("frame {frame}, fine={fine}: {:?}",world.pressure));
        checks.require("topology fault free", world.resolution_receipt.as_ref().is_some_and(|r| r.fault_bits == 0), frame);
    }
    let mut width = vec![0.0; nx*ny]; // 0 denotes omitted dry support, never a coarse wet cell.
    let mut velocity = vec![[0.0; 2]; nx*ny];
    let mut volume = 0.0;
    let mut kinetic = 0.0;
    for c in &g.cells {
        let i = c.id as usize;
        let u = [fields.cell_velocity[2*i] as f64, fields.cell_velocity[2*i+1] as f64];
        let m = fields.density[i] as f64 * c.measure as f64;
        volume += m;
        kinetic += 0.5*m*(u[0]*u[0]+u[1]*u[1]);
        if fine { checks.require("reference cells remain finest", c.widths[0] == 1.0 && c.widths[1] == 1.0, frame); }
        for y in c.minimum[1] as usize..(c.maximum[1] as usize).min(ny) {
            for x in c.minimum[0] as usize..(c.maximum[0] as usize).min(nx) {
                width[x+nx*y] = c.widths[0].max(c.widths[1]) as f64;
                velocity[x+nx*y] = u;
            }
        }
    }
    let area = fill.iter().sum::<f64>();
    checks.lower("positive phi area", area, 1.0, frame);
    let mut centroid = [0.0; 2];
    for (i, &f) in fill.iter().enumerate() {
        centroid[0] += f*((i%nx) as f64+0.5)/area;
        centroid[1] += f*((i/nx) as f64+0.5)/area;
        if f > 0.01 { checks.require("represented liquid has solver support", width[i] > 0.0, format!("frame {frame}, sample {i}")); }
    }
    let max_speed = g.rows.iter().filter(|r| r.terms.iter().any(|t| fields.pressure_member[t.cell_id as usize] != 0))
        .map(|r| fields.face_velocity[r.id as usize].abs() as f64).fold(0.0, f64::max);
    Shot { nx, ny, fill, width, velocity, contour, volume, area, centroid, kinetic,
        cells:g.cells.len(), vertices,
        pressure_rows:fields.pressure_member.iter().filter(|&&v| v != 0).count(),
        advance_ns:world.stage_timings.total_advance, max_speed }
}

impl Shot {
    fn wet(&self, i: usize) -> bool { self.fill[i] >= 0.5 }
    fn at(&self, p: [f64; 2]) -> usize {
        (p[0].floor().max(0.0) as usize).min(self.nx-1)
            + self.nx*(p[1].floor().max(0.0) as usize).min(self.ny-1)
    }
    fn neighbors(&self, i: usize) -> impl Iterator<Item=usize> {
        let x=i%self.nx; let y=i/self.nx;
        [if x>0 {Some(i-1)} else {None}, if x+1<self.nx {Some(i+1)} else {None},
         if y>0 {Some(i-self.nx)} else {None}, if y+1<self.ny {Some(i+self.nx)} else {None}].into_iter().flatten()
    }
    fn components(&self) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.fill.len()]; let mut result = Vec::new();
        for start in 0..visited.len() {
            if visited[start] || !self.wet(start) { continue; }
            let mut component = vec![start]; visited[start] = true; let mut cursor=0;
            while cursor<component.len() {
                for n in self.neighbors(component[cursor]) {
                    if !visited[n] && self.wet(n) { visited[n]=true; component.push(n); }
                }
                cursor+=1;
            }
            if component.len()>=COMPONENT_MIN_AREA { result.push(component); }
        }
        result
    }
    fn boundary_distance(&self) -> Vec<usize> {
        let mut distance = vec![usize::MAX; self.fill.len()]; let mut queue=VecDeque::new();
        for i in 0..distance.len() {
            if self.neighbors(i).any(|n| self.wet(n)!=self.wet(i)) { distance[i]=0; queue.push_back(i); }
        }
        while let Some(i)=queue.pop_front() {
            for n in self.neighbors(i) {
                if distance[n]>distance[i]+1 { distance[n]=distance[i]+1; queue.push_back(n); }
            }
        }
        distance
    }
    fn extent(&self, axis: usize, maximum: bool) -> f64 {
        self.contour.iter().map(|p| p[axis]).fold(if maximum {f64::NEG_INFINITY} else {f64::INFINITY},
            |a,b| if maximum {a.max(b)} else {a.min(b)})
    }
    fn fraction(&self, mask: impl Fn(usize)->bool, accept: impl Fn(f64)->bool) -> f64 {
        let mut n=0; let mut yes=0;
        for i in 0..self.fill.len() { if mask(i) { n+=1; yes+=usize::from(accept(self.width[i])); } }
        if n==0 {f64::NAN} else {yes as f64/n as f64}
    }
    fn surface_coarse(&self, minimum: f64) -> f64 {
        self.contour.iter().filter(|&&p| self.width[self.at(p)]>=minimum).count() as f64/self.contour.len() as f64
    }
}

fn check_pair(a: &Shot, r: &Shot, initial: &Shot, checks: &mut Checks, frame: usize, symmetric: bool) {
    checks.upper("conserved volume", (a.volume-initial.volume).abs()/initial.volume, MASS_REL, frame);
    checks.upper("phi area vs fine", (a.area-r.area).abs()/r.area, AREA_REL, frame);
    // Both arms must retain represented liquid; matching a broken fine run is insufficient.
    for (name, s) in [("adaptive phi volume mismatch",a),("reference phi volume mismatch",r)] {
        checks.upper(name, (s.area-s.volume).abs()/s.volume, 0.08, frame);
    }
    let mismatch = a.fill.iter().zip(&r.fill).map(|(a,b)|(a-b).abs()).sum::<f64>()/r.area;
    checks.upper("symmetric difference vs fine", mismatch, SHAPE_REL, frame);
    checks.upper("centroid vs fine", (a.centroid[0]-r.centroid[0]).hypot(a.centroid[1]-r.centroid[1]), 2.0, frame);
    let mut distances=Vec::new();
    // Bidirectional raster contour distance. L1 distance is conservative relative
    // to Euclidean distance. Fractional area above separately covers subcell loss.
    for (s,t) in [(a,r),(r,a)] {
        let target=t.boundary_distance();
        for i in 0..s.fill.len() {
            if s.neighbors(i).any(|n| s.wet(n)!=s.wet(i)) { distances.push(target[i] as f64); }
        }
    }
    distances.sort_by(f64::total_cmp);
    checks.upper("contour p95 distance", distances[distances.len()*95/100], 2.0, frame);
    checks.upper("contour maximum distance", *distances.last().unwrap(), 4.0, frame);
    for component in r.components() {
        let covered=component.iter().filter(|&&i|a.wet(i)).count() as f64/component.len() as f64;
        checks.lower("resolved component survives", covered, 0.70, frame);
    }
    let energy_scale=r.kinetic.max(initial.volume*initial.ny as f64); // one fine-length gravity scale, quiet floor
    checks.upper("kinetic energy vs fine", (a.kinetic-r.kinetic).abs()/energy_scale, 0.15, frame);
    if symmetric {
        let error=a.fill.iter().enumerate().map(|(i,v)|
            (v-a.fill[(i/a.nx)*a.nx+(a.nx-1-i%a.nx)]).abs()).sum::<f64>()/a.area;
        checks.upper("left right phi symmetry", error, 0.01, frame);
        let resolution_error=(0..a.fill.len()).filter(|&i|r.wet(i))
            .filter(|&i|a.width[i]!=a.width[(i/a.nx)*a.nx+(a.nx-1-i%a.nx)]).count() as f64/r.fill.iter().filter(|&&v|v>=0.5).count() as f64;
        checks.upper("left right resolution symmetry", resolution_error, 0.05, frame);
    }
}

// Geometry-derived feature masks from the FINE reference, never the tested
// planner's reasons/scores. Axis runs detect flat sheets as well as narrow tips.
fn thin_mask(r: &Shot) -> Vec<bool> {
    let mut thin=vec![false;r.fill.len()];
    for axis in 0..2 {
        let (lines,length)=if axis==0 {(r.ny,r.nx)} else {(r.nx,r.ny)};
        for line in 0..lines {
            let index=|k| if axis==0 {line*r.nx+k} else {k*r.nx+line};
            let mut start=0;
            while start<length {
                let phase=r.wet(index(start)); let mut end=start+1;
                while end<length && r.wet(index(end))==phase {end+=1;}
                // Thin liquid, or an internal air gap bounded by liquid.
                if end-start<=4 && (phase || (start>0 && end<length)) {
                    for k in start..end {thin[index(k)]=true;}
                }
                start=end;
            }
        }
    }
    thin
}

fn strain(r: &Shot, i: usize) -> f64 {
    if !r.wet(i) {return 0.0;}
    r.neighbors(i).filter(|&n|r.wet(n)).map(|n|
        (r.velocity[n][0]-r.velocity[i][0]).hypot(r.velocity[n][1]-r.velocity[i][1])*DT).fold(0.0,f64::max)
}

fn contact(lane: Lane, s: &Shot) -> bool {
    match lane {
        Lane::Circle => s.extent(1,false)<=1.0,
        Lane::Pool => s.components().len()==1,
        Lane::Dam => s.extent(0,true)>=s.nx as f64-2.0,
        Lane::Hydro => false,
    }
}

fn run(lane: Lane) {
    let _guard=SUITE_LOCK.lock().unwrap_or_else(|p|p.into_inner());
    let mut checks=Checks::default();
    let mut adaptive=make_world(lane,false); let mut fine=make_world(lane,true);
    let initial=capture(&mut adaptive,&mut checks,0,false);
    let reference_initial=capture(&mut fine,&mut checks,0,true);
    checks.upper("initial volume matches fine", (initial.volume-reference_initial.volume).abs()/reference_initial.volume,MASS_REL,0);
    let mut adaptive_frames=Vec::new(); let mut reference_frames=Vec::new(); let mut report=Vec::new();
    let mut quiet=vec![0u8;initial.fill.len()]; let mut dry=quiet.clone();
    let mut quiet_evidence=0; let mut thin_evidence=0; let mut dry_evidence=0;
    let mut dynamic_evidence=0; let mut visited=vec![false;initial.fill.len()];
    for frame in 1..=lane.steps() {
        // Alternate execution order to reduce systematic cache/thermal bias.
        if frame%2==0 { fine.advance(frame as u32,DT).unwrap(); adaptive.advance(frame as u32,DT).unwrap(); }
        else { adaptive.advance(frame as u32,DT).unwrap(); fine.advance(frame as u32,DT).unwrap(); }
        let a=capture(&mut adaptive,&mut checks,frame,false);
        let r=capture(&mut fine,&mut checks,frame,true);
        checks.upper("reference conserved volume",(r.volume-reference_initial.volume).abs()/reference_initial.volume,MASS_REL,frame);
        check_pair(&a,&r,&initial,&mut checks,frame,lane!=Lane::Dam);
        let distance=r.boundary_distance(); let thin=thin_mask(&r);
        let deformation:Vec<_>=(0..r.fill.len()).map(|i|strain(&r,i)).collect();
        for i in 0..quiet.len() {
            quiet[i]=if r.wet(i) && distance[i]>=4 && deformation[i]<=0.05 {quiet[i].saturating_add(1)} else {0};
            visited[i]|=r.wet(i);
            dry[i]=if !r.wet(i) && distance[i]>=8 {dry[i].saturating_add(1)} else {0};
        }
        if frame>=QUIET_STEPS as usize {
            if quiet.iter().filter(|&&q|q>=QUIET_STEPS).count()>=16 {
                quiet_evidence+=1;
                checks.lower("quiet interior recoarsens",a.fraction(|i|quiet[i]>=QUIET_STEPS,|w|w>=4.0),0.60,frame);
            }
            if (0..dry.len()).filter(|&i|visited[i] && dry[i]>=QUIET_STEPS).count()>=16 {
                dry_evidence+=1;
                checks.lower("vacated support retires or coarsens",a.fraction(|i|visited[i] && dry[i]>=QUIET_STEPS,|w|w==0.0 || w>=4.0),0.90,frame);
            }
        }
        if thin.iter().filter(|&&b|b).count()>=4 {
            thin_evidence+=1;
            checks.lower("thin liquid and internal air gaps stay resolved",a.fraction(|i|thin[i],|w|w>0.0 && w<=2.0),0.90,frame);
        }
        if deformation.iter().filter(|&&v|v>=0.20).count()>=8 {
            dynamic_evidence+=1;
            checks.lower("strong deformation gets fine cells",a.fraction(|i|deformation[i]>=0.20,|w|w>0.0 && w<=2.0),0.80,frame);
        }
        if lane==Lane::Hydro {
            for (name,s) in [("adaptive flat waterline",&a),("reference flat waterline",&r)] {
                checks.upper(name,s.contour.iter().map(|p|(p[1]-15.25).abs()).fold(0.0,f64::max),0.02,frame);
                checks.upper("hydrostatic face speed",s.max_speed,1e-3,frame);
            }
            if frame>=6 {
                checks.lower("hydrostatic wet area coarse",a.fraction(|i|r.wet(i),|w|w>=4.0),0.75,frame);
                checks.lower("hydrostatic surface coarse",a.surface_coarse(4.0),0.75,frame);
                checks.upper("hydrostatic pressure rows",a.pressure_rows as f64/r.pressure_rows as f64,0.25,frame);
            }
        }
        if lane==Lane::Circle && (3..=16).contains(&frame) {
            let t=frame as f64*DT;
            let expected_y=90.0-0.5*200.0*t*t;
            checks.upper("ballistic centre",(a.centroid[1]-expected_y).abs(),200.0*t*DT+0.5,frame);
            checks.upper("free fall area",(a.area-reference_initial.area).abs()/reference_initial.area,0.03,frame);
            let radial=a.contour.iter().map(|p|((p[0]-64.0).hypot(p[1]-a.centroid[1])-14.0).abs()).fold(0.0,f64::max);
            checks.upper("free fall roundness",radial,0.75,frame);
            checks.lower("free fall surface coarsens",a.surface_coarse(2.0),0.50,frame);
            checks.lower("free fall interior coarsens",a.fraction(|i|r.wet(i) && distance[i]>=4,|w|w>=4.0),0.60,frame);
        }
        if lane==Lane::Pool && (3..=9).contains(&frame) {
            checks.lower("remote pool stays coarse before impact",a.fraction(|i| {
                let x=i%r.nx; let y=i/r.nx; (x<12 || x>=52) && (4..16).contains(&y)
            },|w|w>=4.0),0.75,frame);
            checks.require("ball and pool remain separate before approach",a.components().len()==2,frame);
        }
        if lane==Lane::Dam && (3..=9).contains(&frame) {
            checks.lower("reservoir bulk coarsens during release",a.fraction(|i| {
                let x=i%r.nx; let y=i/r.nx; (4..16).contains(&x) && (4..24).contains(&y) && r.wet(i)
            },|w|w>=4.0),0.60,frame);
        }
        report.push(json!({"frame":frame,"time":frame as f64*DT,"cells":a.cells,"fineCells":r.cells,
            "vertices":a.vertices,"fineVertices":r.vertices,"volume":a.volume,"phiArea":a.area,
            "finePhiArea":r.area,"advanceNs":a.advance_ns,"fineAdvanceNs":r.advance_ns,
            "surfaceCoarseFraction":a.surface_coarse(2.0),"contact":contact(lane,&a),"fineContact":contact(lane,&r)}));
        adaptive_frames.push(a); reference_frames.push(r);
    }
    checks.require("quiet interior exercised",quiet_evidence>=6,quiet_evidence);
    if lane!=Lane::Hydro {
        checks.require("thin features exercised",thin_evidence>=3,thin_evidence);
        checks.require("strong deformation exercised",dynamic_evidence>=3,dynamic_evidence);
        if lane==Lane::Circle || lane==Lane::Dam {checks.require("vacated wake exercised",dry_evidence>=3,dry_evidence);}
        check_events(lane,&adaptive_frames,&reference_frames,&mut checks);
    } else {
        for f in 15..adaptive_frames.len() {
            let a=&adaptive_frames[f]; let p=&adaptive_frames[f-1];
            let changed=a.width.iter().zip(&p.width).filter(|(a,b)|a!=b).count() as f64/a.width.len() as f64;
            checks.upper("settled topology does not chatter",changed,0.01,f+1);
        }
    }
    // Exclude a fixed five-frame initialization allowance, not slow outliers.
    let sum=|frames:&[Shot], metric:fn(&Shot)->f64|frames[5..].iter().map(metric).sum::<f64>();
    checks.upper("integrated solver cell budget",sum(&adaptive_frames,|s|s.cells as f64)/sum(&reference_frames,|s|s.cells as f64),lane.cell_budget(),0);
    checks.upper("integrated adaptive phi storage",sum(&adaptive_frames,|s|s.vertices as f64)/sum(&reference_frames,|s|s.vertices as f64),lane.vertex_budget(),0);
    // Always report wall time. Enforce it only on explicit, isolated release
    // benchmark runs; a debug build's timing is not an algorithmic speed claim.
    let time_ratio=sum(&adaptive_frames,|s|s.advance_ns as f64)/sum(&reference_frames,|s|s.advance_ns as f64);
    if std::env::var_os("COARSENING_2D_TIMING").is_some() {
        checks.require("timing uses release build",!cfg!(debug_assertions),"use --release");
        checks.upper("whole advance time budget",time_ratio,lane.time_budget(),0);
    }
    let result=json!({"contractVersion":1,"lane":format!("{lane:?}"),"timingEnforced":std::env::var_os("COARSENING_2D_TIMING").is_some(),
        "advanceTimeRatio":time_ratio,"failures":checks.failures,"frames":report});
    if let Some(directory)=std::env::var_os("COARSENING_2D_REPORT_DIR") {
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(std::path::PathBuf::from(directory).join(format!("{lane:?}.json")),serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    }
    assert!(checks.failures.is_empty(),"{lane:?} coarsening contract:\n{}",serde_json::to_string_pretty(&result["failures"]).unwrap());
}

fn check_events(lane: Lane, a: &[Shot], r: &[Shot], checks: &mut Checks) {
    let reference_contact=r.iter().position(|s|contact(lane,s));
    let adaptive_contact=a.iter().position(|s|contact(lane,s));
    checks.require("reference reaches required contact",reference_contact.is_some(),format!("{lane:?}"));
    checks.require("adaptive reaches required contact",adaptive_contact.is_some(),format!("{lane:?}"));
    if let (Some(rc),Some(ac))=(reference_contact,adaptive_contact) {
        checks.upper("contact timing vs fine",rc.abs_diff(ac) as f64,2.0,ac+1);
        if lane==Lane::Circle || lane==Lane::Pool {
            let expected=if lane==Lane::Circle {(2.0_f64*3.8/10.0).sqrt()} else {(2.0_f64*1.05/9.80665).sqrt()};
            for c in [rc,ac] {checks.upper("contact timing ballistic",((c+1) as f64*DT-expected).abs(),2.0*DT,c+1);}
            checks.require("approach window exists",rc>=2,rc);
            if rc>=2 {
                // Two publications BEFORE reference contact: dry receivers as
                // well as the approaching liquid must be ready in advance.
                let s=&a[rc-2]; let centre=s.nx/2;
                let floor=if lane==Lane::Circle {0} else {14};
                let half=if lane==Lane::Circle {6} else {4};
                checks.lower("predictive contact patch",s.fraction(|i| {
                    let x=i%s.nx; let y=i/s.nx;
                    x>=centre-half && x<centre+half && y>=floor && y<floor+4
                },|w|w>0.0 && w<=2.0),0.75,rc-1);
            }
            let rest=&a[rc..(rc+16).min(a.len())];
            let peak=rest.iter().map(|s|s.fraction(|i|s.wet(i),|w|w==1.0)).fold(0.0,f64::max);
            let falling=a[2..9].iter().map(|s|s.fraction(|i|s.wet(i),|w|w==1.0)).sum::<f64>()/7.0;
            checks.lower("impact concentrates additional finest cells",peak,falling+0.05,rc+1);
            // Compare spreading and splash height over the entire post-impact
            // window, avoiding a result-dependent choice of a flattering frame.
            for f in rc..a.len() {
                checks.upper("splash height vs fine",(a[f].extent(1,true)-r[f].extent(1,true)).abs(),3.0,f+1);
                for side in [false,true] {
                    checks.upper("spread vs fine",(a[f].extent(0,side)-r[f].extent(0,side)).abs(),3.0,f+1);
                }
            }
        } else {
            checks.require("far wall leaves a rebound window",rc+6<r.len(),rc);
            if rc>=2 {
                let s=&a[rc-2];
                checks.lower("far wall receivers refine before arrival",s.fraction(|i|
                    i%s.nx>=s.nx-4 && i/s.nx<6,|w|w>0.0 && w<=2.0),0.75,rc-1);
            }
            if rc+6<r.len() {
                let height=|s:&Shot|s.contour.iter().filter(|p|p[0]>=s.nx as f64-8.0).map(|p|p[1]).fold(0.0,f64::max);
                let base=height(&r[rc]);
                let peak=r[rc+1..].iter().map(height).fold(0.0,f64::max);
                checks.lower("reference produces far wall upturn",peak-base,2.0,rc+1);
                checks.lower("adaptive preserves far wall upturn",a[ac..].iter().map(height).fold(0.0,f64::max),peak-3.0,ac+1);
            }
        }
    }
    if lane==Lane::Circle {
        let average_width=|s:&Shot|s.fill.iter().zip(&s.width).map(|(f,w)|f*w).sum::<f64>()/s.area;
        let early=a[2..8].iter().map(average_width).sum::<f64>()/6.0;
        let late=a[10..16].iter().map(average_width).sum::<f64>()/6.0;
        checks.lower("falling speed does not undo coarsening",late/early,0.80,16);
    }
}

#[test] fn long_dam_release_traversal_and_far_wall() {run(Lane::Dam);}
#[test] fn cm12_figure_2_fall_approach_impact_and_splash() {run(Lane::Circle);}
#[test] fn large_hydrostatic_coarse_surface_and_bulk() {run(Lane::Hydro);}
#[test] fn half_slab_pool_impact_keeps_remote_water_coarse() {run(Lane::Pool);}
