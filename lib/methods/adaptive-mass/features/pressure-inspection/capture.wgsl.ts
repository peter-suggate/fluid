/** Capture kernels consume the resident solver arena and PCG scalars. */
export const PRESSURE_JOURNAL_CONSTANTS_WGSL = /* wgsl */ `// Pressure-journal region, a tail range of state past every physics field.
// It lives there rather than in a buffer of its own because the compute bind
// group is already at the device's ten-storage-buffer ceiling — the same
// reason the tracer lattice sits past the physics fields.
const JOURNAL_HEADER_FLOATS:u32=8u;
const JOURNAL_ITERATION_FLOATS:u32=16u;
const JOURNAL_FIELD_COUNT:u32=4u;
`;

export const PRESSURE_JOURNAL_ACCESS_WGSL = /* wgsl */ `// True only on the pipeline variant the host encodes at a snapshot iteration.
// A dispatch cannot be told which iteration it is — the uniform is written once
// per frame and WebGPU has no push constant — so "is this a snapshot" is a
// pipeline property and "which snapshot" is a device-side cursor.
override JOURNAL_SNAPSHOT:bool=false;
fn journalBase()->u32{return p.journal.x;}
fn journalIterationCapacity()->u32{return p.journal.y;}
fn journalSnapshotCapacity()->u32{return p.journal.z;}
fn journalCellStride()->u32{return p.journal.w;}
fn journalArmed()->bool{return p.journal.x!=0u&&p.journal.y!=0u&&p.journal.w!=0u;}
fn journalSnapshotField(slot:u32,field:u32)->u32{
  return journalBase()+JOURNAL_HEADER_FLOATS
    +journalIterationCapacity()*JOURNAL_ITERATION_FLOATS
    +(slot*JOURNAL_FIELD_COUNT+field)*journalCellStride();
}
`;

export const PRESSURE_JOURNAL_CAPTURE_WGSL = /* wgsl */ `// One journal record per *encoded* iteration.
//
// Deliberately ungated: the solve encodes a fixed ceiling and the residual gate
// zeroes the tail, so a gated kernel would stop recording exactly where the
// interesting thing — the gate closing — happens. This runs every encoded
// iteration and stores the gate rather than obeying it, which is what lets the
// film distinguish "converged at 43" from "ran 128 times".
//
// The cursor is the encoded iteration index because this kernel is dispatched
// exactly once per encoded iteration, in order, on a queue with an implicit
// barrier between dispatches.
@compute @workgroup_size(64)
fn journalIteration(@builtin(local_invocation_id)lid:vec3u){
  if(lid.x!=0u||!journalArmed()){return;}
  let base=journalBase();
  let cursor=u32(max(0.0,state[base]));
  if(cursor>=journalIterationCapacity()){return;}
  var snapshot=-1.0;
  if(JOURNAL_SNAPSHOT){
    let slot=u32(max(0.0,state[base+1u]));
    if(slot<journalSnapshotCapacity()){snapshot=f32(slot);state[base+1u]=f32(slot+1u);}
  }
  let at=base+JOURNAL_HEADER_FLOATS+cursor*JOURNAL_ITERATION_FLOATS;
  state[at]=select(0.0,1.0,pipelinedPressureActive());
  state[at+1u]=scalars[0];
  state[at+2u]=scalars[2];
  state[at+3u]=scalars[3];
  state[at+4u]=scalars[4];
  state[at+5u]=scalars[1];
  state[at+6u]=scalars[12];
  state[at+7u]=scalars[14];
  state[at+8u]=scalars[10];
  state[at+9u]=scalars[11];
  state[at+10u]=scalars[18];
  state[at+11u]=scalars[13];
  state[at+12u]=scalars[8];
  state[at+13u]=scalars[9];
  state[at+14u]=scalars[16];
  state[at+15u]=snapshot;
  state[base]=f32(cursor+1u);
  state[base+2u]=1.0;
  state[base+3u]=1.0;
}

// A whole-field snapshot of the four cell fields the picture is made of.
//
// Runs over accepted cells rather than the compacted pressure worklist, so a
// dry cell inside the topology reads as an explicit zero instead of keeping
// whatever the previous capture left there. The slot is the snapshot cursor the
// paired journalIteration dispatch just advanced.
@compute @workgroup_size(64)
fn journalSnapshot(@builtin(global_invocation_id)gid:vec3u){
  if(!journalArmed()){return;}
  let base=journalBase();
  let cursorValue=state[base+1u];
  if(cursorValue<0.5){return;}
  let slot=u32(cursorValue)-1u;
  if(slot>=journalSnapshotCapacity()){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);
  if(cell==INVALID||cell>=journalCellStride()){return;}
  let live=peiPressureCellMember(cell);
  state[journalSnapshotField(slot,0u)+cell]=
    select(0.0,state[p.stateOffsets2.x+cell],live);
  state[journalSnapshotField(slot,1u)+cell]=
    select(0.0,state[p.stateOffsets3.y+cell],live);
  state[journalSnapshotField(slot,2u)+cell]=
    select(0.0,state[p.stateOffsets3.z+cell],live);
  state[journalSnapshotField(slot,3u)+cell]=
    select(0.0,state[p.stateOffsets3.w+cell],live);
}
`;
