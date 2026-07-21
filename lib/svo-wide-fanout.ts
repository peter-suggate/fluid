/**
 * Deterministic CPU plan for the renderer's 4x4x4 view of a binary SVO.
 *
 * A wide page collapses two canonical octree levels. A terminal at an odd
 * canonical level therefore covers 2x2x2 page slots while an even-level
 * terminal covers one. The source octree remains authoritative: terminal
 * records only point back to source node/leaf indices.
 */

export type SvoWideCoordinate = readonly [number, number, number];
export type SvoWideVec3 = readonly [number, number, number];

export const SVO_WIDE_FANOUT = 64;
export const SVO_WIDE_AXIS = 4;
export const SVO_WIDE_INVALID_INDEX = 0xffff_ffff;
export const SVO_WIDE_MICRO_MIP_WORDS = 64 + 8 + 1;

export interface SvoWideTerminalInput {
  sourceNodeIndex: number;
  sourceLeafIndex?: number;
  /** Canonical binary-octree level. */
  level: number;
  coordinate: SvoWideCoordinate;
  solidOpacity?: number;
  fluidFraction?: number;
}
export interface SvoWideFanoutPlanInput {
  sourceGeneration: number;
  generation: number;
  maximumDepth: number;
  terminals: readonly SvoWideTerminalInput[];
}

export interface SvoWideOpacity {
  solidMean: number;
  solidMaximum: number;
  fluidMean: number;
  fluidMaximum: number;
}

export interface SvoWideTerminalDescriptor {
  kind: "terminal";
  slot: number;
  sourceNodeIndex: number;
  sourceLeafIndex: number;
  sourceLevel: number;
  opacity: SvoWideOpacity;
}

export interface SvoWidePageDescriptor {
  kind: "page";
  slot: number;
  pageIndex: number;
  sourceLevel: number;
  opacity: SvoWideOpacity;
}

export type SvoWideDescriptor = SvoWideTerminalDescriptor | SvoWidePageDescriptor;

export interface SvoWidePagePlan {
  index: number;
  /** Canonical level of the region covered by this page. Always even. */
  level: number;
  coordinate: SvoWideCoordinate;
  morton: bigint;
  occupancyLow: number;
  occupancyHigh: number;
  descriptors: readonly SvoWideDescriptor[];
  /** 4^3, 2^3, then 1^3 packed opacity values. */
  microMips: readonly SvoWideOpacity[];
}

export interface SvoWideFanoutPlan {
  sourceGeneration: number;
  generation: number;
  maximumDepth: number;
  pages: readonly SvoWidePagePlan[];
  descriptorCount: number;
  terminalSlotCount: number;
}

export interface SvoWideWorldMapping {
  origin: SvoWideVec3;
  cellSize: SvoWideVec3;
  brickSize: 4 | 8;
  maximumDepth: number;
}

export interface SvoWideRay {
  origin: SvoWideVec3;
  direction: SvoWideVec3;
  tMin?: number;
  tMax?: number;
}

export interface SvoWideTraversalOptions {
  maximumPageVisits?: number;
  stackCapacity?: number;
}

export type SvoWideTraversalResult =
  | { status: "hit"; pageVisits: number; descriptorTests: number; sourceNodeIndex: number; sourceLeafIndex: number;
    sourceLevel: number; coordinate: SvoWideCoordinate; tEnter: number; tExit: number }
  | { status: "miss"; pageVisits: number; descriptorTests: number }
  | { status: "work-exhausted" | "stack-overflow"; pageVisits: number; descriptorTests: number };

const UINT32_MAX = 0xffff_ffff;
const EMPTY_OPACITY: SvoWideOpacity = Object.freeze({ solidMean: 0, solidMaximum: 0, fluidMean: 0, fluidMaximum: 0 });

function uint32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > UINT32_MAX) throw new RangeError(`${label} must fit uint32`);
  return value >>> 0;
}

function unit(value: number | undefined, label: string): number {
  const resolved = value ?? 0;
  if (!Number.isFinite(resolved) || resolved < 0 || resolved > 1) throw new RangeError(`${label} must be in [0, 1]`);
  return resolved;
}

function validateLevel(level: number, maximumDepth: number): void {
  if (!Number.isInteger(level) || level < 0 || level > maximumDepth) {
    throw new RangeError("Terminal level must be within the canonical octree depth");
  }
}

function validateCoordinate(coordinate: SvoWideCoordinate, level: number): void {
  const width = 2 ** level;
  if (coordinate.length !== 3 || coordinate.some((value) => !Number.isInteger(value) || value < 0 || value >= width)) {
    throw new RangeError("Terminal coordinate is outside its canonical level");
  }
}

function mortonAtLevel(coordinate: SvoWideCoordinate, level: number): bigint {
  let result = 0n;
  for (let bit = 0; bit < level; bit += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      result |= BigInt((coordinate[axis] >> bit) & 1) << BigInt(3 * bit + axis);
    }
  }
  return result;
}

function pageKey(level: number, coordinate: SvoWideCoordinate): string {
  return `${level}:${coordinate[0]},${coordinate[1]},${coordinate[2]}`;
}

function slotIndex(x: number, y: number, z: number): number {
  return x + y * 4 + z * 16;
}

export function svoWideSlotCoordinate(slot: number): SvoWideCoordinate {
  if (!Number.isInteger(slot) || slot < 0 || slot >= SVO_WIDE_FANOUT) throw new RangeError("Wide slot must be 0..63");
  return [slot & 3, (slot >> 2) & 3, (slot >> 4) & 3];
}

function opacityForTerminal(terminal: SvoWideTerminalInput): SvoWideOpacity {
  const solid = unit(terminal.solidOpacity, "Solid opacity");
  const fluid = unit(terminal.fluidFraction, "Fluid fraction");
  return { solidMean: solid, solidMaximum: solid, fluidMean: fluid, fluidMaximum: fluid };
}

function reduceOpacity(values: readonly SvoWideOpacity[]): SvoWideOpacity {
  if (values.length === 0) return EMPTY_OPACITY;
  let solidMean = 0;
  let solidMaximum = 0;
  let fluidMean = 0;
  let fluidMaximum = 0;
  for (const value of values) {
    solidMean += value.solidMean;
    fluidMean += value.fluidMean;
    solidMaximum = Math.max(solidMaximum, value.solidMaximum);
    fluidMaximum = Math.max(fluidMaximum, value.fluidMaximum);
  }
  return { solidMean: solidMean / values.length, solidMaximum, fluidMean: fluidMean / values.length, fluidMaximum };
}

function buildMicroMips(base: readonly SvoWideOpacity[]): SvoWideOpacity[] {
  const result = [...base];
  const middle: SvoWideOpacity[] = [];
  for (let z = 0; z < 2; z += 1) {
    for (let y = 0; y < 2; y += 1) {
      for (let x = 0; x < 2; x += 1) {
        const children: SvoWideOpacity[] = [];
        for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) for (let dx = 0; dx < 2; dx += 1) {
          children.push(base[slotIndex(x * 2 + dx, y * 2 + dy, z * 2 + dz)]);
        }
        middle.push(reduceOpacity(children));
      }
    }
  }
  result.push(...middle, reduceOpacity(middle));
  return result;
}

function terminalPageLevel(level: number): number {
  if (level === 0) return 0;
  return Math.floor((level - 1) / 2) * 2;
}

function terminalSlots(terminal: SvoWideTerminalInput, pageLevel: number): number[] {
  if (terminal.level === pageLevel) return Array.from({ length: 64 }, (_, slot) => slot);
  const delta = terminal.level - pageLevel;
  const divisor = 2 ** delta;
  const local = terminal.coordinate.map((value) => value % divisor) as [number, number, number];
  if (delta === 2) return [slotIndex(local[0], local[1], local[2])];
  if (delta !== 1) throw new Error("Internal wide-terminal level mismatch");
  const result: number[] = [];
  for (let z = 0; z < 2; z += 1) for (let y = 0; y < 2; y += 1) for (let x = 0; x < 2; x += 1) {
    result.push(slotIndex(local[0] * 2 + x, local[1] * 2 + y, local[2] * 2 + z));
  }
  return result.sort((a, b) => a - b);
}

/** Build a level-major, Morton-sorted 4^3 directory from canonical terminals. */
export function planSvoWideFanout(input: SvoWideFanoutPlanInput): SvoWideFanoutPlan {
  const sourceGeneration = uint32(input.sourceGeneration, "Source generation");
  const generation = uint32(input.generation, "Wide generation");
  if (!Number.isInteger(input.maximumDepth) || input.maximumDepth < 0 || input.maximumDepth > 21) {
    throw new RangeError("Maximum depth must be an integer from 0 to 21");
  }
  const terminals = input.terminals.map((value) => {
    validateLevel(value.level, input.maximumDepth);
    validateCoordinate(value.coordinate, value.level);
    uint32(value.sourceNodeIndex, "Source node index");
    if (value.sourceLeafIndex !== undefined) uint32(value.sourceLeafIndex, "Source leaf index");
    opacityForTerminal(value);
    return value;
  }).sort((a, b) => a.level - b.level || (mortonAtLevel(a.coordinate, a.level) < mortonAtLevel(b.coordinate, b.level) ? -1 : 1)
    || a.sourceNodeIndex - b.sourceNodeIndex);

  // Canonical leaves must be disjoint. Detect equal and ancestor/descendant support.
  for (let a = 0; a < terminals.length; a += 1) for (let b = a + 1; b < terminals.length; b += 1) {
    const shallow = terminals[a].level <= terminals[b].level ? terminals[a] : terminals[b];
    const deep = shallow === terminals[a] ? terminals[b] : terminals[a];
    const divisor = 2 ** (deep.level - shallow.level);
    if (shallow.coordinate.every((value, axis) => value === Math.floor(deep.coordinate[axis] / divisor))) {
      throw new RangeError("Canonical wide-fanout terminals must not overlap");
    }
  }

  type MutablePage = { level: number; coordinate: SvoWideCoordinate; morton: bigint; entries: Map<number, SvoWideDescriptor> };
  const pageByKey = new Map<string, MutablePage>();
  const ensurePage = (level: number, coordinate: SvoWideCoordinate): MutablePage => {
    const key = pageKey(level, coordinate);
    let page = pageByKey.get(key);
    if (!page) {
      page = { level, coordinate, morton: mortonAtLevel(coordinate, level), entries: new Map() };
      pageByKey.set(key, page);
    }
    return page;
  };
  for (const terminal of terminals) {
    const finalPageLevel = terminalPageLevel(terminal.level);
    for (let level = 0; level <= finalPageLevel; level += 2) {
      const shift = terminal.level - level;
      ensurePage(level, terminal.coordinate.map((value) => Math.floor(value / 2 ** shift)) as unknown as SvoWideCoordinate);
    }
  }
  const mutablePages = [...pageByKey.values()].sort((a, b) => a.level - b.level || (a.morton < b.morton ? -1 : a.morton > b.morton ? 1 : 0));
  const pageIndex = new Map(mutablePages.map((page, index) => [pageKey(page.level, page.coordinate), index]));

  for (let index = 1; index < mutablePages.length; index += 1) {
    const child = mutablePages[index];
    const parentLevel = child.level - 2;
    const parentCoordinate = child.coordinate.map((value) => Math.floor(value / 4)) as unknown as SvoWideCoordinate;
    const parent = pageByKey.get(pageKey(parentLevel, parentCoordinate));
    if (!parent) throw new Error("Internal wide-page ancestry is incomplete");
    const slot = slotIndex(child.coordinate[0] & 3, child.coordinate[1] & 3, child.coordinate[2] & 3);
    if (parent.entries.has(slot)) throw new RangeError("Canonical terminals conflict with a wide child page");
    parent.entries.set(slot, { kind: "page", slot, pageIndex: index, sourceLevel: child.level, opacity: EMPTY_OPACITY });
  }

  let terminalSlotCount = 0;
  for (const terminal of terminals) {
    const level = terminalPageLevel(terminal.level);
    const shift = terminal.level - level;
    const coordinate = terminal.coordinate.map((value) => Math.floor(value / 2 ** shift)) as unknown as SvoWideCoordinate;
    const page = pageByKey.get(pageKey(level, coordinate));
    if (!page) throw new Error("Internal wide terminal page is missing");
    for (const slot of terminalSlots(terminal, level)) {
      if (page.entries.has(slot)) throw new RangeError("Canonical terminals conflict within a wide page");
      page.entries.set(slot, {
        kind: "terminal", slot, sourceNodeIndex: terminal.sourceNodeIndex,
        sourceLeafIndex: terminal.sourceLeafIndex ?? SVO_WIDE_INVALID_INDEX,
        sourceLevel: terminal.level, opacity: opacityForTerminal(terminal),
      });
      terminalSlotCount += 1;
    }
  }

  // Child summaries are computed deepest-first and become the parent's base texel.
  const completed = new Array<SvoWidePagePlan>(mutablePages.length);
  let descriptorCount = 0;
  for (let index = mutablePages.length - 1; index >= 0; index -= 1) {
    const source = mutablePages[index];
    const descriptors = [...source.entries.values()].sort((a, b) => a.slot - b.slot).map((descriptor) => {
      if (descriptor.kind !== "page") return descriptor;
      const child = completed[descriptor.pageIndex];
      return { ...descriptor, opacity: child.microMips[SVO_WIDE_MICRO_MIP_WORDS - 1] };
    });
    const base = Array.from({ length: 64 }, () => EMPTY_OPACITY);
    let occupancyLow = 0;
    let occupancyHigh = 0;
    for (const descriptor of descriptors) {
      base[descriptor.slot] = descriptor.opacity;
      if (descriptor.slot < 32) occupancyLow = (occupancyLow | (1 << descriptor.slot)) >>> 0;
      else occupancyHigh = (occupancyHigh | (1 << (descriptor.slot - 32))) >>> 0;
    }
    descriptorCount += descriptors.length;
    completed[index] = {
      index, level: source.level, coordinate: source.coordinate, morton: source.morton,
      occupancyLow, occupancyHigh, descriptors, microMips: buildMicroMips(base),
    };
  }
  return { sourceGeneration, generation, maximumDepth: input.maximumDepth, pages: completed, descriptorCount, terminalSlotCount };
}

function rayAabb(ray: SvoWideRay, minimum: SvoWideVec3, maximum: SvoWideVec3): readonly [number, number] | null {
  let enter = ray.tMin ?? 0;
  let exit = ray.tMax ?? Number.POSITIVE_INFINITY;
  if (ray.origin.some((value) => !Number.isFinite(value)) || ray.direction.some((value) => !Number.isFinite(value))
      || ray.direction.every((value) => value === 0) || !Number.isFinite(enter) || exit < enter) {
    throw new RangeError("Wide traversal ray is invalid");
  }
  for (let axis = 0; axis < 3; axis += 1) {
    if (ray.direction[axis] === 0) {
      if (ray.origin[axis] < minimum[axis] || ray.origin[axis] > maximum[axis]) return null;
      continue;
    }
    let near = (minimum[axis] - ray.origin[axis]) / ray.direction[axis];
    let far = (maximum[axis] - ray.origin[axis]) / ray.direction[axis];
    if (near > far) [near, far] = [far, near];
    enter = Math.max(enter, near);
    exit = Math.min(exit, far);
    if (exit < enter) return null;
  }
  return [enter, exit];
}

function canonicalBounds(level: number, coordinate: SvoWideCoordinate, mapping: SvoWideWorldMapping): readonly [SvoWideVec3, SvoWideVec3] {
  const scale = 2 ** (mapping.maximumDepth - level) * mapping.brickSize;
  const minimum = coordinate.map((value, axis) => mapping.origin[axis] + value * scale * mapping.cellSize[axis]) as unknown as SvoWideVec3;
  const maximum = coordinate.map((value, axis) => mapping.origin[axis] + (value + 1) * scale * mapping.cellSize[axis]) as unknown as SvoWideVec3;
  return [minimum, maximum];
}

/** Near-to-far CPU oracle over the planned 4^3 directory. */
export function traverseSvoWideFanout(
  ray: SvoWideRay,
  plan: SvoWideFanoutPlan,
  mapping: SvoWideWorldMapping,
  options: SvoWideTraversalOptions = {},
): SvoWideTraversalResult {
  if (mapping.maximumDepth !== plan.maximumDepth) throw new RangeError("Wide plan and world mapping depths differ");
  if (mapping.origin.some((value) => !Number.isFinite(value)) || mapping.cellSize.some((value) => !Number.isFinite(value) || value <= 0)
      || (mapping.brickSize !== 4 && mapping.brickSize !== 8)) throw new RangeError("Wide world mapping is invalid");
  if (plan.pages.length === 0) return { status: "miss", pageVisits: 0, descriptorTests: 0 };
  const maximumPageVisits = options.maximumPageVisits ?? 128;
  const stackCapacity = options.stackCapacity ?? 64;
  if (!Number.isInteger(maximumPageVisits) || maximumPageVisits < 1 || !Number.isInteger(stackCapacity) || stackCapacity < 1) {
    throw new RangeError("Wide traversal budgets must be positive integers");
  }
  type Candidate = { kind: "page"; pageIndex: number; enter: number; exit: number }
    | { kind: "terminal"; descriptor: SvoWideTerminalDescriptor; coordinate: SvoWideCoordinate; enter: number; exit: number };
  const rootBounds = canonicalBounds(0, [0, 0, 0], mapping);
  const rootInterval = rayAabb(ray, rootBounds[0], rootBounds[1]);
  if (!rootInterval) return { status: "miss", pageVisits: 0, descriptorTests: 0 };
  const stack: Candidate[] = [{ kind: "page", pageIndex: 0, enter: rootInterval[0], exit: rootInterval[1] }];
  let pageVisits = 0;
  let descriptorTests = 0;
  while (stack.length > 0) {
    const candidate = stack.pop() as Candidate;
    if (candidate.kind === "terminal") {
      return { status: "hit", pageVisits, descriptorTests, sourceNodeIndex: candidate.descriptor.sourceNodeIndex,
        sourceLeafIndex: candidate.descriptor.sourceLeafIndex, sourceLevel: candidate.descriptor.sourceLevel,
        coordinate: candidate.coordinate, tEnter: candidate.enter, tExit: candidate.exit };
    }
    if (pageVisits >= maximumPageVisits) return { status: "work-exhausted", pageVisits, descriptorTests };
    pageVisits += 1;
    const page = plan.pages[candidate.pageIndex];
    const next: Candidate[] = [];
    for (const descriptor of page.descriptors) {
      descriptorTests += 1;
      const slot = svoWideSlotCoordinate(descriptor.slot);
      const slotCoordinate = page.coordinate.map((value, axis) => value * 4 + slot[axis]) as unknown as SvoWideCoordinate;
      const slotBounds = canonicalBounds(page.level + 2, slotCoordinate, mapping);
      const interval = rayAabb(ray, slotBounds[0], slotBounds[1]);
      if (!interval) continue;
      if (descriptor.kind === "page") next.push({ kind: "page", pageIndex: descriptor.pageIndex, enter: interval[0], exit: interval[1] });
      else {
        const divisor = 2 ** ((page.level + 2) - descriptor.sourceLevel);
        const terminalCoordinate = slotCoordinate.map((value) => Math.floor(value / divisor)) as unknown as SvoWideCoordinate;
        const terminalBounds = canonicalBounds(descriptor.sourceLevel, terminalCoordinate, mapping);
        const terminalInterval = rayAabb(ray, terminalBounds[0], terminalBounds[1]);
        if (terminalInterval) next.push({ kind: "terminal", descriptor, coordinate: terminalCoordinate,
          enter: terminalInterval[0], exit: terminalInterval[1] });
      }
    }
    next.sort((a, b) => b.enter - a.enter || (a.kind === "terminal" ? 1 : 0) - (b.kind === "terminal" ? 1 : 0));
    if (stack.length + next.length > stackCapacity) return { status: "stack-overflow", pageVisits, descriptorTests };
    stack.push(...next);
  }
  return { status: "miss", pageVisits, descriptorTests };
}
