/** Serialize production 32-byte water vertices without changing the surface.
 * The right-handed rotation (x,y,z)->(x,-z,y) preserves triangle winding.
 */
export function waterMeshObj(vertices: Float32Array, label: string, options: { opticalNormals?: boolean } = {}): {
  obj: string; bounds: { min: number[]; max: number[] };
} {
  if (vertices.length === 0 || vertices.length % 24 !== 0) throw new Error("Water mesh must contain complete triangles");
  const lines = [`# ${label.replace(/[\r\n]/g, " ")}`, "o NBFLIP_Water"];
  const bounds = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  for (let i = 0; i < vertices.length; i += 8) {
    const point = [vertices[i]!, -vertices[i + 2]!, vertices[i + 1]!];
    if (!point.every(Number.isFinite)) throw new Error("Water mesh contains non-finite positions");
    point.forEach((value, axis) => {
      bounds.min[axis] = Math.min(bounds.min[axis]!, value);
      bounds.max[axis] = Math.max(bounds.max[axis]!, value);
    });
    lines.push(`v ${point.join(" ")}`);
  }
  if (options.opticalNormals) {
    for (let i = 0; i < vertices.length; i += 8) {
      const normal = [vertices[i + 4]!, -vertices[i + 6]!, vertices[i + 5]!];
      const length = Math.hypot(...normal);
      if (!normal.every(Number.isFinite) || !(length > 1e-12)) throw new Error("Water mesh contains invalid optical normals");
      // The production vertex shader normalizes this lane before interpolation.
      lines.push(`vn ${normal.map(value => value / length).join(" ")}`);
    }
  }
  for (let i = 1; i <= vertices.length / 8; i += 3) {
    lines.push(options.opticalNormals ? `f ${i}//${i} ${i + 1}//${i + 1} ${i + 2}//${i + 2}` : `f ${i} ${i + 1} ${i + 2}`);
  }
  return { obj: lines.join("\n") + "\n", bounds };
}
