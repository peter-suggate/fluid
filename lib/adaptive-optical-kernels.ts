export const adaptiveOpticalLayerShader = /* wgsl */ `
struct AdaptiveParams {
  paper: vec4f,
  grid: vec4f,
}

@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var volumeIn: texture_3d<f32>;
@group(0) @binding(2) var columnBaseIn: texture_2d<f32>;
@group(0) @binding(3) var motionIn: texture_2d<f32>;
@group(0) @binding(4) var motionOut: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var envelopeIn: texture_3d<f32>;
@group(0) @binding(6) var envelopeOut: texture_storage_3d<r32float, write>;
@group(0) @binding(7) var rawBaseIn: texture_2d<f32>;
@group(0) @binding(8) var rawBaseOut: texture_storage_2d<r32float, write>;
@group(0) @binding(9) var smoothIn: texture_2d<f32>;
@group(0) @binding(10) var smoothOut: texture_storage_2d<r32float, write>;
@group(0) @binding(11) var<storage, read_write> nextColumnBases: array<u32>;
@group(0) @binding(12) var<storage, read_write> diagnostics: array<atomic<u32>, 16>;
@group(0) @binding(13) var<uniform> params: AdaptiveParams;

fn nx() -> i32 { return i32(textureDimensions(volumeIn).x); }
fn fineNy() -> i32 { return i32(textureDimensions(volumeIn).y) - 2; }
fn nz() -> i32 { return i32(textureDimensions(volumeIn).z); }
fn packedNy() -> i32 { return i32(textureDimensions(volumeIn).y); }
fn alphaScale() -> f32 { return params.paper.x; }
fn minimumDilation() -> i32 { return i32(round(params.paper.y)); }
fn maximumDilation() -> i32 { return i32(round(params.paper.z)); }
fn airborneOffset() -> i32 { return i32(round(params.paper.w)); }
fn airborneDilation() -> i32 { return i32(round(params.grid.x)); }
fn smoothingRadius() -> i32 { return i32(round(params.grid.y)); }
fn gridWidth() -> f32 { return params.grid.z; }
fn maximumBase() -> i32 { return i32(round(params.grid.w)); }
fn validColumn(x: i32, z: i32) -> bool { return x >= 0 && x < nx() && z >= 0 && z < nz(); }
fn baseAt(x: i32, z: i32) -> i32 {
  if (!validColumn(x, z)) { return 0; }
  return i32(round(textureLoad(columnBaseIn, vec2i(x, z), 0).x));
}
fn volumeCell(q: vec3i) -> f32 {
  if (!validColumn(q.x, q.z) || q.y < 0 || q.y >= fineNy()) { return 0.0; }
  let base = baseAt(q.x, q.z);
  if (base > 0 && q.y < base) { return textureLoad(volumeIn, vec3i(q.x, 0, q.z), 0).x; }
  let packedY = 2 + q.y - base;
  if (packedY < 2 || packedY >= packedNy()) { return 0.0; }
  return textureLoad(volumeIn, vec3i(q.x, packedY, q.z), 0).x;
}
fn velocityCell(q: vec3i) -> vec3f {
  if (!validColumn(q.x, q.z) || q.y < 0 || q.y >= fineNy()) { return vec3f(0.0); }
  let base = baseAt(q.x, q.z);
  if (base > 0 && q.y < base) {
    let t = clamp(f32(q.y) / f32(max(base - 1, 1)), 0.0, 1.0);
    return mix(textureLoad(velocityIn, vec3i(q.x, 0, q.z), 0).xyz,
               textureLoad(velocityIn, vec3i(q.x, 1, q.z), 0).xyz, t);
  }
  let packedY = 2 + q.y - base;
  if (packedY < 2 || packedY >= packedNy()) { return vec3f(0.0); }
  return textureLoad(velocityIn, vec3i(q.x, packedY, q.z), 0).xyz;
}
fn wet(q: vec3i) -> bool { return volumeCell(q) >= 0.5; }
fn surfaceCell(q: vec3i) -> bool {
  if (!wet(q)) { return false; }
  return !wet(q + vec3i(0, 1, 0)) || !wet(q - vec3i(1, 0, 0))
      || !wet(q + vec3i(1, 0, 0)) || !wet(q - vec3i(0, 0, 1))
      || !wet(q + vec3i(0, 0, 1));
}
fn finiteScalar(value: f32) -> bool { return value == value && abs(value) <= 3.402823e38; }

@compute @workgroup_size(8, 8, 1)
fn estimateMotion(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u32(nx()) || gid.y >= u32(nz())) { return; }
  let x = i32(gid.x); let z = i32(gid.y);
  if (gid.x == 0u && gid.y == 0u) {
    atomicOr(&diagnostics[4], 1u);
    atomicStore(&diagnostics[12], u32(max(0, minimumDilation())));
    atomicStore(&diagnostics[13], u32(max(0, maximumDilation())));
    atomicStore(&diagnostics[14], u32(max(0, maximumBase())));
    atomicStore(&diagnostics[15], u32(max(0, fineNy())));
  }
  var groundSurface = -1;
  if (wet(vec3i(x, 0, z))) {
    for (var y = 0; y < fineNy(); y += 1) {
      if (!wet(vec3i(x, y, z))) { break; }
      groundSurface = y;
    }
  }

  var error = 0.0;
  var dilation = minimumDilation();
  if (groundSurface >= 0) {
    let count = groundSurface + 1;
    var sumT = 0.0; var sumTT = 0.0;
    var sumX = 0.0; var sumTX = 0.0;
    var sumY = 0.0;
    var sumZ = 0.0; var sumTZ = 0.0;
    for (var y = 0; y <= groundSurface; y += 1) {
      var t = 0.0;
      if (count > 1) { t = f32(y) / f32(count - 1); }
      let value = velocityCell(vec3i(x, y, z));
      sumT += t; sumTT += t * t;
      sumX += value.x; sumTX += t * value.x;
      sumY += value.y;
      sumZ += value.z; sumTZ += t * value.z;
    }
    let n = f32(count);
    let denominator = n * sumTT - sumT * sumT;
    var slopeX = 0.0; var slopeZ = 0.0;
    if (denominator > 1e-6) {
      slopeX = (n * sumTX - sumT * sumX) / denominator;
      slopeZ = (n * sumTZ - sumT * sumZ) / denominator;
    }
    let interceptX = (sumX - slopeX * sumT) / n;
    let interceptZ = (sumZ - slopeZ * sumT) / n;
    let averageY = sumY / n;
    for (var y = 0; y <= groundSurface; y += 1) {
      var t = 0.0;
      if (count > 1) { t = f32(y) / f32(count - 1); }
      let original = velocityCell(vec3i(x, y, z));
      let fitted = vec3f(interceptX + slopeX * t, averageY, interceptZ + slopeZ * t);
      let difference = abs(original - fitted);
      error += difference.x + difference.y + difference.z;
    }
    dilation = i32(round(clamp(alphaScale() * error * gridWidth(), f32(minimumDilation()), f32(maximumDilation()))));
  }

  if (!finiteScalar(error)) { error = 0.0; atomicAdd(&diagnostics[11], 1u); }
  // buildVerticalSeeds enumerates every surface independently. Its contract
  // needs the ground-connected dilation here, not a radius selected from one
  // (possibly airborne) seed.
  textureStore(motionOut, vec2i(x, z), vec4f(error, 0.0, f32(dilation), f32(groundSurface)));
}

// For every surface cell and every already-consumed Manhattan radius, retain
// the lowest layer boundary it can induce. Keeping the radius budget as a
// dimension preserves overlapping seeds with different radii exactly.
@compute @workgroup_size(4, 4, 4)
fn buildVerticalSeeds(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u32(nx()) || gid.z >= u32(nz()) || gid.y > u32(maximumDilation())) { return; }
  if (all(gid == vec3u(0u))) { atomicOr(&diagnostics[4], 2u); }
  let x = i32(gid.x); let consumed = i32(gid.y); let z = i32(gid.z);
  let motion = textureLoad(motionIn, vec2i(x, z), 0);
  let groundSurface = i32(round(motion.w));
  let groundRadius = i32(round(motion.z));
  var best = f32(maximumBase());
  for (var y = 0; y < fineNy(); y += 1) {
    if (!surfaceCell(vec3i(x, y, z))) { continue; }
    var radius = airborneDilation();
    if (groundSurface >= 0 && abs(y - groundSurface) <= airborneOffset()) { radius = groundRadius; }
    if (consumed <= radius) { best = min(best, f32(max(0, y - radius + consumed))); }
  }
  textureStore(envelopeOut, vec3i(x, consumed, z), vec4f(best));
}

// Exact separable x pass. The envelope's y coordinate records radius already
// consumed by the final z pass plus the current x distance.
@compute @workgroup_size(4, 4, 4)
fn buildHorizontalEnvelope(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u32(nx()) || gid.z >= u32(nz()) || gid.y > u32(maximumDilation())) { return; }
  if (all(gid == vec3u(0u))) { atomicOr(&diagnostics[4], 4u); }
  let x = i32(gid.x); let consumed = i32(gid.y); let z = i32(gid.z);
  var best = f32(maximumBase());
  for (var sourceX = max(0, x - maximumDilation()); sourceX <= min(nx() - 1, x + maximumDilation()); sourceX += 1) {
    let distance = abs(sourceX - x);
    let budget = consumed + distance;
    if (budget > maximumDilation()) { continue; }
    best = min(best, textureLoad(envelopeIn, vec3i(sourceX, budget, z), 0).x);
  }
  textureStore(envelopeOut, vec3i(x, consumed, z), vec4f(best));
}

@compute @workgroup_size(8, 8, 1)
fn finishManhattanDilation(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u32(nx()) || gid.y >= u32(nz())) { return; }
  if (gid.x == 0u && gid.y == 0u) { atomicOr(&diagnostics[4], 8u); }
  let x = i32(gid.x); let z = i32(gid.y);
  var best = f32(maximumBase());
  for (var sourceZ = max(0, z - maximumDilation()); sourceZ <= min(nz() - 1, z + maximumDilation()); sourceZ += 1) {
    let distance = abs(sourceZ - z);
    let horizontal = textureLoad(envelopeIn, vec3i(x, distance, sourceZ), 0).x;
    best = min(best, horizontal);
  }
  textureStore(rawBaseOut, vec2i(x, z), vec4f(clamp(best, 0.0, f32(maximumBase()))));
}

@compute @workgroup_size(8, 8, 1)
fn smoothLayer(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u32(nx()) || gid.y >= u32(nz())) { return; }
  if (gid.x == 0u && gid.y == 0u) { atomicOr(&diagnostics[4], 16u); }
  let x = i32(gid.x); let z = i32(gid.y);
  var sum = 0.0; var count = 0.0;
  for (var dz = -smoothingRadius(); dz <= smoothingRadius(); dz += 1) {
    for (var dx = -smoothingRadius(); dx <= smoothingRadius(); dx += 1) {
      let q = vec2i(x + dx, z + dz);
      if (!validColumn(q.x, q.y)) { continue; }
      sum += textureLoad(smoothIn, q, 0).x; count += 1.0;
    }
  }
  let raw = textureLoad(rawBaseIn, vec2i(x, z), 0).x;
  textureStore(smoothOut, vec2i(x, z), vec4f(min(raw, sum / max(count, 1.0))));
}

@compute @workgroup_size(8, 8, 1)
fn finalizeLayer(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u32(nx()) || gid.y >= u32(nz())) { return; }
  if (gid.x == 0u && gid.y == 0u) { atomicOr(&diagnostics[4], 32u); }
  let x = i32(gid.x); let z = i32(gid.y); let index = u32(x + nx() * z);
  let raw = clamp(i32(round(textureLoad(rawBaseIn, vec2i(x, z), 0).x)), 0, maximumBase());
  var base = clamp(i32(round(textureLoad(smoothIn, vec2i(x, z), 0).x)), 0, maximumBase());
  if (base == 1) { base = 0; }
  nextColumnBases[index] = u32(base);
  let motion = textureLoad(motionIn, vec2i(x, z), 0);
  if (motion.w >= 0.0) {
    atomicAdd(&diagnostics[0], u32(max(0.0, motion.z) * 256.0 + 0.5));
    atomicMax(&diagnostics[1], u32(max(0.0, motion.z)));
    atomicAdd(&diagnostics[10], 1u);
  }
  atomicMax(&diagnostics[2], bitcast<u32>(max(0.0, motion.x)));
  atomicAdd(&diagnostics[3], u32(base));
  atomicMax(&diagnostics[5], u32(base));
  let activeSamples = select(fineNy(), fineNy() - base + 2, base > 0);
  atomicAdd(&diagnostics[6], u32(activeSamples));
  atomicAdd(&diagnostics[7], u32(fineNy() - base));
  if (base > 0) { atomicAdd(&diagnostics[8], 1u); }
  atomicAdd(&diagnostics[9], u32(max(0, raw - base)));
}
`;
