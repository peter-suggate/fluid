/**
 * Solver-grid cross-section rendered as an independent presentation layer.
 *
 * Keeping this out of both water renderers lets the same scientific overlay
 * compose over raster optics and the legacy ray marcher. The shared view
 * uniform supplies the slice axis/position through `debug.xy`; this pipeline
 * only owns the grid-specific sampling and alpha blend.
 */

export const gridOverlayShader = /* wgsl */ `
struct Uniforms {
  viewport: vec4f,
  cameraPosition: vec4f,
  cameraTarget: vec4f,
  container: vec4f,
  options: vec4f,
  gridInfo: vec4f,
  debug: vec4f,
}
struct BodyGPU {
  positionRadius: vec4f,
  halfSizeShape: vec4f,
  orientation: vec4f,
  colorSelected: vec4f,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> bodies: array<BodyGPU, 12>;
@group(0) @binding(2) var fluidField: texture_3d<f32>;
@group(0) @binding(3) var tallCellBases: texture_2d<f32>;
@group(0) @binding(4) var adaptiveCells: texture_3d<u32>;

struct VertexOutput { @builtin(position) position: vec4f, @location(0) uv: vec2f }
@vertex fn vertexMain(@builtin(vertex_index) index: u32) -> VertexOutput {
  var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var result: VertexOutput;
  result.position = vec4f(positions[index], 0.0, 1.0);
  result.uv = positions[index] * 0.5 + 0.5;
  return result;
}

fn boxIntersection(ro: vec3f, rd: vec3f, boundsMin: vec3f, boundsMax: vec3f) -> vec2f {
  let inv = 1.0 / rd;
  let a = (boundsMin - ro) * inv;
  let b = (boundsMax - ro) * inv;
  let near3 = min(a, b);
  let far3 = max(a, b);
  return vec2f(max(max(near3.x, near3.y), near3.z), min(min(far3.x, far3.y), far3.z));
}
fn quatRotate(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.yzw, v);
  return v + q.x * t + cross(q.yzw, t);
}
fn quatInverseRotate(q: vec4f, v: vec3f) -> vec3f { return quatRotate(vec4f(q.x, -q.yzw), v); }

fn bodyDistance(ro: vec3f, rd: vec3f, body: BodyGPU) -> f32 {
  let origin = quatInverseRotate(body.orientation, ro - body.positionRadius.xyz);
  let direction = quatInverseRotate(body.orientation, rd);
  let shape = i32(round(body.halfSizeShape.w));
  if (shape == 0) {
    let radius = body.halfSizeShape.x;
    let projected = dot(origin, direction);
    let discriminant = projected * projected - dot(origin, origin) + radius * radius;
    if (discriminant < 0.0) { return 1e20; }
    let root = sqrt(discriminant);
    let near = -projected - root;
    let far = -projected + root;
    let candidate = select(far, near, near > 1e-4);
    return select(1e20, candidate, candidate > 1e-4);
  }
  if (shape == 1) {
    let hit = boxIntersection(origin, direction, -body.halfSizeShape.xyz, body.halfSizeShape.xyz);
    if (hit.x > hit.y) { return 1e20; }
    let candidate = select(hit.y, hit.x, hit.x > 1e-4);
    return select(1e20, candidate, candidate > 1e-4);
  }
  let radius = body.halfSizeShape.x;
  let halfHeight = body.halfSizeShape.y;
  var best = 1e20;
  let a = dot(direction.xz, direction.xz);
  let b = dot(origin.xz, direction.xz);
  let c = dot(origin.xz, origin.xz) - radius * radius;
  if (a > 1e-7) {
    let discriminant = b * b - a * c;
    if (discriminant >= 0.0) {
      let side = (-b - sqrt(discriminant)) / a;
      let y = origin.y + direction.y * side;
      if (side > 1e-4 && abs(y) <= halfHeight) { best = side; }
    }
  }
  if (shape == 2) {
    for (var end = -1.0; end <= 1.0; end += 2.0) {
      let offset = origin - vec3f(0.0, end * halfHeight, 0.0);
      let projected = dot(offset, direction);
      let discriminant = projected * projected - dot(offset, offset) + radius * radius;
      if (discriminant >= 0.0) {
        let cap = -projected - sqrt(discriminant);
        if (cap > 1e-4 && cap < best) { best = cap; }
      }
    }
  } else if (abs(direction.y) > 1e-7) {
    for (var end = -1.0; end <= 1.0; end += 2.0) {
      let cap = (end * halfHeight - origin.y) / direction.y;
      let point = origin + direction * cap;
      if (cap > 1e-4 && cap < best && dot(point.xz, point.xz) <= radius * radius) { best = cap; }
    }
  }
  return best;
}

fn nearestBodyDistance(ro: vec3f, rd: vec3f) -> f32 {
  var nearest = 1e20;
  let bodyCount = u32(round(u.options.z));
  for (var index = 0u; index < 12u; index += 1u) {
    if (index >= bodyCount) { break; }
    nearest = min(nearest, bodyDistance(ro, rd, bodies[index]));
  }
  return nearest;
}

fn fluidSample(cell: vec3i) -> f32 {
  let dims = vec3i(u.gridInfo.xyz);
  let q = clamp(cell, vec3i(0), dims - vec3i(1));
  if (u.gridInfo.w < 1.5) { return textureLoad(fluidField, q, 0).x; }
  let cellSizeY = u.container.y / max(u.gridInfo.y, 1.0);
  if (u.gridInfo.w > 2.5) { return clamp(0.5 - textureLoad(fluidField, q, 0).x / cellSizeY, 0.0, 1.0); }
  let base = i32(round(textureLoad(tallCellBases, q.xz, 0).x));
  if (q.y < base && base > 0) { let t=clamp(f32(q.y)/f32(max(base-1,1)),0.0,1.0);let phi=mix(textureLoad(fluidField,vec3i(q.x,0,q.z),0).x,textureLoad(fluidField,vec3i(q.x,1,q.z),0).x,t);return clamp(0.5-phi/cellSizeY,0.0,1.0); }
  let packedY = 2 + q.y - base;
  let stored = vec3i(textureDimensions(fluidField));
  if (packedY < 2 || packedY >= stored.y) { return 0.0; }
  return clamp(0.5-textureLoad(fluidField, vec3i(q.x, packedY, q.z), 0).x/cellSizeY,0.0,1.0);
}

fn adaptiveCellKey(cell: vec3i, dims: vec3i) -> vec2u {
  return textureLoad(adaptiveCells, clamp(cell, vec3i(0), dims - vec3i(1)), 0).xy;
}

struct RepresentedCell {
  lower: vec3f,
  upper: vec3f,
}

fn representedCell(cell: vec3i, dims: vec3i, boundsMin: vec3f, size: vec3f, adaptiveGrid: bool, tallGrid: bool) -> RepresentedCell {
  var lower = cell;
  var upper = cell + vec3i(1);
  if (adaptiveGrid) {
    let key = adaptiveCellKey(cell, dims);
    lower = vec3i(i32(key.x & 1023u), i32(key.y & 1023u), i32((key.x >> 10u) & 1023u));
    let leafSize = i32((key.x >> 20u) & 1023u);
    upper = vec3i(lower.x + leafSize, i32((key.y >> 10u) & 1023u), lower.z + leafSize);
  } else if (tallGrid) {
    let base = i32(round(textureLoad(tallCellBases, cell.xz, 0).x));
    if (cell.y < base) {
      lower.y = 0;
      upper.y = base;
    }
  }
  let cellSize = size / vec3f(dims);
  return RepresentedCell(boundsMin + vec3f(lower) * cellSize, boundsMin + vec3f(upper) * cellSize);
}

fn bodySignedDistance(point: vec3f, body: BodyGPU) -> f32 {
  let local = quatInverseRotate(body.orientation, point - body.positionRadius.xyz);
  let shape = i32(round(body.halfSizeShape.w));
  if (shape == 0) { return length(local) - body.halfSizeShape.x; }
  if (shape == 1) {
    let q = abs(local) - body.halfSizeShape.xyz;
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
  }
  if (shape == 2) {
    let axisPoint = vec3f(0.0, clamp(local.y, -body.halfSizeShape.y, body.halfSizeShape.y), 0.0);
    return length(local - axisPoint) - body.halfSizeShape.x;
  }
  let q = vec2f(length(local.xz) - body.halfSizeShape.x, abs(local.y) - body.halfSizeShape.y);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}

struct GridBodySample {
  occupied: bool,
  selected: f32,
}

fn gridBodySample(cellBounds: RepresentedCell) -> GridBodySample {
  let center = (cellBounds.lower + cellBounds.upper) * 0.5;
  let cellRadius = length((cellBounds.upper - cellBounds.lower) * 0.5);
  var nearest = 1e20;
  var result = GridBodySample(false, 0.0);
  let bodyCount = u32(round(u.options.z));
  for (var index = 0u; index < 12u; index += 1u) {
    if (index >= bodyCount) { break; }
    // A represented cell is solid when its circumscribed volume reaches the
    // analytic primitive. This conservative voxelization keeps small bodies
    // visible and, for adaptive grids, fills the complete quadtree leaf rather
    // than leaking the hidden dense backing-grid resolution into the view.
    let closest = clamp(bodies[index].positionRadius.xyz, cellBounds.lower, cellBounds.upper);
    let sphereDistance = length(closest - bodies[index].positionRadius.xyz) - bodies[index].positionRadius.w;
    let shapeDistance = bodySignedDistance(center, bodies[index]) - cellRadius;
    let distance = max(sphereDistance, shapeDistance);
    if (distance <= cellRadius && distance < nearest) {
      nearest = distance;
      result = GridBodySample(true, bodies[index].colorSelected.w);
    }
  }
  return result;
}

struct GridSample {
  color: vec3f,
  alpha: f32,
  solid: bool,
}

fn gridSample(point: vec3f, boundsMin: vec3f, size: vec3f, axis: i32, footprint: f32) -> GridSample {
  let dims = vec3i(u.gridInfo.xyz);
  let local3 = clamp((point - boundsMin) / size, vec3f(0.0), vec3f(0.99999)) * vec3f(dims);
  let cell = clamp(vec3i(floor(local3)), vec3i(0), dims - vec3i(1));
  var samplePosition = local3.xy;
  var cellPerPixel = vec2f(footprint * f32(dims.x) / size.x, footprint * f32(dims.y) / size.y);
  if (axis == 2) {
    samplePosition.x = local3.z;
    cellPerPixel.x = footprint * f32(dims.z) / size.z;
  }
  let derivative = max(cellPerPixel, vec2f(1e-5));
  let pixelsPerCell = 1.0 / max(derivative.x, derivative.y);
  let lineFade = smoothstep(2.5, 6.0, pixelsPerCell);
  let dotFade = smoothstep(9.0, 18.0, pixelsPerCell);
  let adaptiveGrid = u.debug.z > 0.5;
  let tallGrid = u.gridInfo.w > 1.5 && u.gridInfo.w < 2.5;
  var base = 0.0;
  if (tallGrid) { base = round(textureLoad(tallCellBases, cell.xz, 0).x); }
  let stored = vec3i(textureDimensions(fluidField));
  let bandLayers = select(dims.y, stored.y - 2, tallGrid);
  let bandTop = min(i32(base) + bandLayers, dims.y);
  let columnLine = 1.0 - smoothstep(0.4, 1.2, (0.5 - abs(fract(samplePosition.x) - 0.5)) / derivative.x);
  var fill = vec3f(0.0);
  var alpha = 0.0;
  var line = 0.0;
  var sampleDot = 0.0;
  if (adaptiveGrid) {
    // The simulation transports its fields on a dense cubic backing texture,
    // but pressure is represented by adaptive quadtree/tall cells. The id
    // texture maps each backing voxel to that represented cell. Only an id
    // transition is a real grid edge; the hidden fine-grid boundaries must
    // not appear in this scientific overlay.
    let own = adaptiveCellKey(cell, dims);
    let horizontalAxis = select(0, 2, axis == 2);
    var lowerHorizontal = cell;
    var upperHorizontal = cell;
    lowerHorizontal[horizontalAxis] -= 1;
    upperHorizontal[horizontalAxis] += 1;
    var lowerHorizontalEdge = cell[horizontalAxis] == 0;
    var upperHorizontalEdge = cell[horizontalAxis] == dims[horizontalAxis] - 1;
    if (!lowerHorizontalEdge) { lowerHorizontalEdge = any(adaptiveCellKey(lowerHorizontal, dims) != own); }
    if (!upperHorizontalEdge) { upperHorizontalEdge = any(adaptiveCellKey(upperHorizontal, dims) != own); }
    var below = cell;
    var above = cell;
    below.y -= 1;
    above.y += 1;
    var lowerVerticalEdge = cell.y == 0;
    var upperVerticalEdge = cell.y == dims.y - 1;
    if (!lowerVerticalEdge) { lowerVerticalEdge = any(adaptiveCellKey(below, dims) != own); }
    if (!upperVerticalEdge) { upperVerticalEdge = any(adaptiveCellKey(above, dims) != own); }
    let horizontalFraction = fract(samplePosition.x);
    let verticalFraction = fract(samplePosition.y);
    let horizontalDistance = min(select(1e6, horizontalFraction / derivative.x, lowerHorizontalEdge), select(1e6, (1.0 - horizontalFraction) / derivative.x, upperHorizontalEdge));
    let verticalDistance = min(select(1e6, verticalFraction / derivative.y, lowerVerticalEdge), select(1e6, (1.0 - verticalFraction) / derivative.y, upperVerticalEdge));
    line = 1.0 - smoothstep(0.4, 1.2, min(horizontalDistance, verticalDistance));
    let isTall = !lowerVerticalEdge || !upperVerticalEdge;
    let wet = fluidSample(cell) > 0.5;
    fill = select(select(vec3f(0.85, 0.91, 0.89), vec3f(0.20, 0.50, 0.74), wet), select(vec3f(0.10, 0.23, 0.22), vec3f(0.03, 0.52, 0.47), wet), isTall);
    alpha = select(select(0.18, 0.55, wet), select(0.40, 0.78, wet), isTall);
  } else if (cell.y < i32(base)) {
    let wet = fluidSample(cell) > 0.5;
    fill = select(vec3f(0.10, 0.23, 0.22), vec3f(0.03, 0.52, 0.47), wet);
    alpha = select(0.40, 0.78, wet);
    let baseEdge = 1.0 - smoothstep(0.4, 1.4, min(samplePosition.y, abs(base - samplePosition.y)) / derivative.y);
    line = max(columnLine * lineFade * 0.45, baseEdge);
    let dy = min(abs(samplePosition.y - 0.5), abs(samplePosition.y - (base - 0.5)));
    let distance = length(vec2f(fract(samplePosition.x) - 0.5, dy));
    sampleDot = (1.0 - smoothstep(0.17, 0.17 + max(derivative.x, derivative.y) * 1.6, distance)) * dotFade;
  } else if (cell.y < bandTop) {
    let wet = fluidSample(cell) > 0.5;
    fill = select(vec3f(0.85, 0.91, 0.89), vec3f(0.20, 0.50, 0.74), wet);
    alpha = select(0.18, 0.55, wet);
    line = max(columnLine, 1.0 - smoothstep(0.4, 1.2, (0.5 - abs(fract(samplePosition.y) - 0.5)) / derivative.y));
    let distance = length(fract(samplePosition.xy) - vec2f(0.5));
    sampleDot = (1.0 - smoothstep(0.17, 0.17 + max(derivative.x, derivative.y) * 1.6, distance)) * dotFade;
  } else {
    let stripe = smoothstep(0.38, 0.5, abs(fract((samplePosition.x + samplePosition.y) * 0.25) - 0.5));
    fill = vec3f(0.62, 0.24, 0.22);
    alpha = 0.08 + 0.10 * stripe;
    line = columnLine * 0.35;
  }
  line *= lineFade;
  let gridBody = gridBodySample(representedCell(cell, dims, boundsMin, size, adaptiveGrid, tallGrid));
  if (gridBody.occupied) {
    fill = mix(vec3f(0.96, 0.43, 0.12), vec3f(1.0, 0.78, 0.38), 0.18 * gridBody.selected);
    alpha = 0.97;
    sampleDot = 0.0;
  }
  var color = mix(fill, vec3f(0.03, 0.08, 0.09), line);
  color = mix(color, vec3f(0.02, 0.05, 0.06), sampleDot);
  alpha = max(alpha, max(line * 0.85, sampleDot * 0.92));
  return GridSample(color, alpha, gridBody.occupied);
}

fn displayColor(linear: vec3f) -> vec3f {
  let mapped = linear / (linear + vec3f(1.0));
  return pow(max(mapped, vec3f(0.0)), vec3f(1.0 / 2.2));
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let axis = i32(round(u.debug.x));
  if (axis <= 0 || u.gridInfo.w <= 0.5) { discard; }
  let ndc = input.uv * 2.0 - 1.0;
  let origin = u.cameraPosition.xyz;
  let forward = normalize(u.cameraTarget.xyz - origin);
  let right = normalize(cross(forward, vec3f(0.0, 1.0, 0.0)));
  let up = normalize(cross(right, forward));
  let direction = normalize(forward + right * ndc.x * u.viewport.x / max(u.viewport.y, 1.0) * 0.72 + up * ndc.y * 0.72);
  let size = u.container.xyz;
  let boundsMin = vec3f(-0.5 * size.x, 0.0, -0.5 * size.z);
  let boundsMax = boundsMin + size;
  let dims = vec3f(u.gridInfo.xyz);
  var denominator = direction.z;
  var rayOrigin = origin.z;
  var planeCoordinate = 0.0;
  if (axis == 1) {
    let layer = clamp(floor(u.debug.y * dims.z), 0.0, dims.z - 1.0);
    planeCoordinate = boundsMin.z + (layer + 0.5) * size.z / dims.z;
  } else {
    let layer = clamp(floor(u.debug.y * dims.x), 0.0, dims.x - 1.0);
    planeCoordinate = boundsMin.x + (layer + 0.5) * size.x / dims.x;
    denominator = direction.x;
    rayOrigin = origin.x;
  }
  if (abs(denominator) <= 1e-5) { discard; }
  let distance = (planeCoordinate - rayOrigin) / denominator;
  let point = origin + direction * distance;
  let inside = all(point >= boundsMin - vec3f(1e-4)) && all(point <= boundsMax + vec3f(1e-4));
  if (distance <= 0.0 || !inside) { discard; }
  let footprint = distance * 1.44 / max(u.viewport.y, 1.0);
  var overlay = gridSample(point, boundsMin, size, axis, footprint);
  if (distance >= nearestBodyDistance(origin, direction) && !overlay.solid) { discard; }
  let grip = clamp(1.0 - (boundsMax.y - point.y) / (0.03 * size.y), 0.0, 1.0) * 0.8;
  overlay.color = mix(overlay.color, vec3f(0.51, 0.95, 0.82), grip);
  overlay.alpha = max(overlay.alpha, grip);
  return vec4f(displayColor(overlay.color), overlay.alpha);
}
`;

export class GridOverlayPipeline {
  private pipeline?: GPURenderPipeline;
  private bindGroup?: GPUBindGroup;
  private volume?: GPUTexture;
  private columnBases?: GPUTexture;
  private adaptiveCells?: GPUTexture;

  constructor(
    private readonly device: GPUDevice,
    private readonly targetFormat: GPUTextureFormat,
    private readonly uniformBuffer: GPUBuffer,
    private readonly bodyBuffer: GPUBuffer
  ) {}

  async initialize() {
    const shaderModule = this.device.createShaderModule({ label: "Solver grid overlay", code: gridOverlayShader });
    const compilation = await shaderModule.getCompilationInfo();
    const errors = compilation.messages.filter((message) => message.type === "error");
    if (errors.length > 0) throw new Error(errors.map((error) => `${error.lineNum}:${error.linePos} ${error.message}`).join("\n"));
    this.pipeline = await this.device.createRenderPipelineAsync({
      label: "Solver grid overlay composite",
      layout: "auto",
      vertex: { module: shaderModule, entryPoint: "vertexMain" },
      fragment: {
        module: shaderModule,
        entryPoint: "fragmentMain",
        targets: [{
          format: this.targetFormat,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" }
          }
        }]
      },
      primitive: { topology: "triangle-list" }
    });
    this.rebuildBindGroup();
  }

  setVolume(volume: GPUTexture, columnBases: GPUTexture, adaptiveCells: GPUTexture) {
    if (this.volume === volume && this.columnBases === columnBases && this.adaptiveCells === adaptiveCells) return;
    this.volume = volume;
    this.columnBases = columnBases;
    this.adaptiveCells = adaptiveCells;
    this.rebuildBindGroup();
  }

  private rebuildBindGroup() {
    if (!this.pipeline || !this.volume || !this.columnBases || !this.adaptiveCells) return;
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.bodyBuffer } },
        { binding: 2, resource: this.volume.createView({ dimension: "3d" }) },
        { binding: 3, resource: this.columnBases.createView() },
        { binding: 4, resource: this.adaptiveCells.createView({ dimension: "3d" }) }
      ]
    });
  }

  encode(encoder: GPUCommandEncoder, target: GPUTextureView): boolean {
    if (!this.pipeline || !this.bindGroup) return false;
    const pass = encoder.beginRenderPass({
      label: "Solver grid overlay",
      colorAttachments: [{ view: target, loadOp: "load", storeOp: "store" }]
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
    return true;
  }
}
