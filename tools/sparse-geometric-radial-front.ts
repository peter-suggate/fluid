export interface RadialFrontReceipt {
  readonly semantics: "binary cell-centre phase mask, bilinear mask sampling at 1/8 cell";
  readonly angleCount: number;
  readonly radii_cells: readonly (number | null)[];
  readonly meanRadius_cells: number | null;
  readonly radiusRange_cells: number | null;
  readonly d4MaximumError_cells: number;
  readonly roughnessAfterLowModes: { readonly removedThroughMode: number;
    readonly rms_cells: number | null; readonly range_cells: number | null };
}

/** Farthest occupied phase-mask sample along fixed rays from the domain centre.
 * This is a jaggedness/footprint measure, not a subcell phi-zero reconstruction. */
export function radialFront(mask: ArrayLike<number>, nx: number, nz: number,
  angleCount = 128): RadialFrontReceipt {
  const cx = nx / 2, cz = nz / 2;
  const radii: Array<number | null> = [];
  const step = 0.125;
  const maximum = Math.hypot(nx, nz);
  const sample = (px: number, pz: number) => {
    const gx = px - 0.5, gz = pz - 0.5;
    const x0 = Math.floor(gx), z0 = Math.floor(gz);
    const tx = gx - x0, tz = gz - z0;
    let value = 0;
    for (let dz = 0; dz < 2; dz++) for (let dx = 0; dx < 2; dx++) {
      const x = x0 + dx, z = z0 + dz;
      if (x < 0 || z < 0 || x >= nx || z >= nz) continue;
      value += Number(mask[x + nx * z]) * (dx ? tx : 1 - tx) * (dz ? tz : 1 - tz);
    }
    return value;
  };
  for (let angle = 0; angle < angleCount; angle++) {
    const theta = 2 * Math.PI * angle / angleCount;
    let radius: number | null = null;
    for (let r = 0; r <= maximum; r += step) {
      const px = cx + r * Math.cos(theta), pz = cz + r * Math.sin(theta);
      if (px < 0 || pz < 0 || px >= nx || pz >= nz) break;
      if (sample(px, pz) >= 0.5 - 1e-6) radius = r;
    }
    radii.push(radius);
  }
  const finite = radii.filter((value): value is number => value !== null);
  const mean = finite.reduce((sum, value) => sum + value, 0) / Math.max(1, finite.length);
  const index = (angle: number) => ((Math.round(angle) % angleCount) + angleCount) % angleCount;
  let d4MaximumError_cells = 0;
  for (let i = 0; i < angleCount; i++) for (const target of [
    index(angleCount / 2 - i), index(-i), index(angleCount / 4 - i),
  ]) {
    if (radii[i] !== null && radii[target] !== null) d4MaximumError_cells = Math.max(
      d4MaximumError_cells, Math.abs(radii[i]! - radii[target]!));
    else if (radii[i] !== radii[target]) d4MaximumError_cells = Infinity;
  }
  // Retain only angular frequencies finer than the square's smooth k=4/8/12
  // shape harmonics; this keeps the receipt about jaggedness rather than
  // penalising the authored square for not being circular.
  const removedThroughMode = 12;
  const residuals = finite.length === angleCount ? radii.map((value, i) => {
    const theta = 2 * Math.PI * i / angleCount;
    let smooth = 0;
    for (let mode = 0; mode <= removedThroughMode; mode++) {
      const cosine = 2 / angleCount * radii.reduce<number>((sum, radius, j) => sum
        + Number(radius) * Math.cos(mode * 2 * Math.PI * j / angleCount), 0);
      const sine = mode === 0 ? 0 : 2 / angleCount * radii.reduce<number>((sum, radius, j) => sum
        + Number(radius) * Math.sin(mode * 2 * Math.PI * j / angleCount), 0);
      smooth += (mode === 0 ? 0.5 : 1) * cosine * Math.cos(mode * theta)
        + sine * Math.sin(mode * theta);
    }
    return value! - smooth;
  }) : [];
  return { semantics: "binary cell-centre phase mask, bilinear mask sampling at 1/8 cell",
    angleCount, radii_cells: radii,
    meanRadius_cells: finite.length ? mean : null,
    radiusRange_cells: finite.length ? Math.max(...finite) - Math.min(...finite) : null,
    d4MaximumError_cells,
    roughnessAfterLowModes: { removedThroughMode,
      rms_cells: residuals.length ? Math.sqrt(residuals.reduce(
        (sum, value) => sum + value * value, 0) / residuals.length) : null,
      range_cells: residuals.length ? Math.max(...residuals) - Math.min(...residuals) : null } };
}

export function expectedCentredSquareRadialFront(halfWidth_cells: number,
  angleCount = 128): readonly number[] {
  return Array.from({ length: angleCount }, (_, index) => {
    const theta = 2 * Math.PI * index / angleCount;
    return halfWidth_cells / Math.max(Math.abs(Math.cos(theta)), Math.abs(Math.sin(theta)));
  });
}

export function radialFrontsFromFields(phi: ArrayLike<number | null>, density: ArrayLike<number>,
  [nx, ny, nz]: readonly [number, number, number]) {
  const phiGround = new Uint8Array(nx * nz), phiProjected = new Uint8Array(nx * nz);
  const densityIntegrated = new Uint8Array(nx * nz);
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
    const groundPhi = phi[x + nx * (ny * z)];
    phiGround[x + nx * z] = groundPhi !== null && Number.isFinite(Number(groundPhi))
      && Number(groundPhi) <= 0 ? 1 : 0;
    let volume = 0;
    for (let y = 0; y < ny; y++) {
      const at = x + nx * (y + ny * z);
      if (phi[at] !== null && Number.isFinite(Number(phi[at])) && Number(phi[at]) <= 0) {
        phiProjected[x + nx * z] = 1;
      }
      volume += Math.max(0, Number(density[at]));
    }
    densityIntegrated[x + nx * z] = volume >= 0.5 ? 1 : 0;
  }
  return { densityIntegrated: radialFront(densityIntegrated, nx, nz),
    phiGround: radialFront(phiGround, nx, nz),
    phiProjected: radialFront(phiProjected, nx, nz) };
}

export function radialOutlineSvg(series: readonly { label: string; color: string;
  radii_cells: readonly (number | null)[] }[], title: string): string {
  const size = 640, centre = size / 2, scale = 0.42 * size / 24;
  const lines = series.map(({ label, color, radii_cells }, seriesIndex) => {
    const points = radii_cells.flatMap((radius, index) => radius === null ? [] : [[
      centre + scale * radius * Math.cos(2 * Math.PI * index / radii_cells.length),
      centre - scale * radius * Math.sin(2 * Math.PI * index / radii_cells.length),
    ].map(value => value.toFixed(2)).join(",")]).join(" ");
    return `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2"/>`
      + `<text x="20" y="${48 + 22 * seriesIndex}" fill="${color}">${label}</text>`;
  }).join("\n");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
<rect width="100%" height="100%" fill="#10151d"/><text x="20" y="24" fill="white">${title}</text>
<path d="M ${centre} 30 V ${size - 30} M 30 ${centre} H ${size - 30}" stroke="#384352"/>
${lines}
</svg>\n`;
}

export function radialComparisonSvg(input: readonly { run: "before" | "after";
  step: number; fronts: ReturnType<typeof radialFrontsFromFields> }[]): string {
  const width = 1200, height = 450, panel = 380, centreY = 245, scale = 7;
  const colors = ["#94a3b8", "#38bdf8", "#f59e0b", "#f43f5e", "#a78bfa"];
  const fields = [["densityIntegrated", "Integrated-density phase mask"],
    ["phiGround", "Ground published-φ phase mask"],
    ["phiProjected", "Projected published-φ phase mask"]] as const;
  const paths = fields.flatMap(([field], panelIndex) => input.map((item, index) => {
    const radii = item.fronts[field].radii_cells;
    const points = radii.flatMap((radius, angle) => radius === null ? [] : [[
      panelIndex * panel + panel / 2 + scale * radius * Math.cos(2 * Math.PI * angle / radii.length),
      centreY - scale * radius * Math.sin(2 * Math.PI * angle / radii.length),
    ].map(value => value.toFixed(2)).join(",")]).join(" ");
    return `<polyline points="${points}" fill="none" stroke="${colors[index % colors.length]}"`
      + ` stroke-width="${item.run === "after" ? 2.2 : 1.2}"`
      + ` stroke-dasharray="${item.run === "before" ? "5 4" : "none"}" opacity="0.9"/>`;
  })).join("\n");
  const titles = fields.map(([, label], index) =>
    `<text x="${index * panel + panel / 2}" y="32" text-anchor="middle" fill="white">${label}</text>`)
    .join("\n");
  const legend = input.map((item, index) => `<text x="20" y="${height - 18 - 18 * index}"`
    + ` fill="${colors[index % colors.length]}">${item.run} step ${item.step}</text>`).join("\n");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
<rect width="100%" height="100%" fill="#10151d"/>${titles}
${paths}
${legend}
</svg>\n`;
}
