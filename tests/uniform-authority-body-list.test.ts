/**
 * The bodies-only authority head (UniformMixedPressureAuthority.encodeBodies)
 * surfaces each owner whose 3x3x3-tile box touches a body-reached region.
 * It used to scan an eighth of a held region a workgroup and run that
 * eighth's h tiles in series; it now scans a region a workgroup (a lane a
 * tile), surfaces 4h owners there and appends reached h tiles to one list
 * that an indirect launch runs (zero workgroups when empty). This mirrors
 * both selections on randomized regions and requires the same owners, each
 * surfaced once, with the list inside its ceil(owners/64) capacity.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

type Region = { live: boolean[]; width: number[]; near: number };
const corner = (k: number, side: number) => [k % side, Math.floor(k / side) % side, Math.floor(k / (side * side))];
function reached(near: number, tile: number): boolean {
  const local = corner(tile, 4);
  for (let k = 0; k < 27; k++) {
    if (((near >>> k) & 1) === 0) continue;
    const d = corner(k, 3).map(v => v - 1);
    if (d.every((dk, i) => (dk >= 0 || local[i] === 0) && (dk <= 0 || local[i] === 3))) return true;
  }
  return false;
}
function random(seed: number) { let s = seed >>> 0; return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 2 ** 32; }; }

/** Old surfaceRegions: item = region*8 + eighth; eight tiles a workgroup, then the eighth's h tiles in series. */
function eighths(regions: Region[]): string[] {
  const out: string[] = [];
  for (let item = 0; item < 8 * regions.length; item++) {
    const r = regions[item >> 3], first = 8 * (item & 7);
    if (r.near === 0 || !r.live[0]) continue;
    const fine: number[] = [];
    for (let l = 0; l < 8; l++) {
      const t = first + l;
      if (!reached(r.near, t) || !r.live[t]) continue;
      if (r.width[t] === 1) fine.push(t); else out.push(`${item >> 3}:${t}:4h`);
    }
    for (const t of fine) for (let l = 0; l < 64; l++) out.push(`${item >> 3}:${t}:${l}`);
  }
  return out;
}

/** New bodyRegions + bodyFine: a region a workgroup, h tiles appended (any order) to a capped list. */
function listed(regions: Region[], capacity: number, next: () => number): { owners: string[]; count: number } {
  const owners: string[] = [], list: [number, number][] = [];let count = 0;
  regions.forEach((r, region) => {
    if (r.near === 0 || !r.live[0]) return;
    const lanes = Array.from({ length: 64 }, (_, l) => l).sort(() => next() - 0.5);
    for (const t of lanes) {
      if (!reached(r.near, t) || !r.live[t]) continue;
      if (r.width[t] === 1) { const at = count++; if (at < capacity) list[at] = [region, t]; }
      else owners.push(`${region}:${t}:4h`);
    }
  });
  for (let j = 0; j < Math.min(count, capacity); j++) { const [region, t] = list[j]; for (let l = 0; l < 64; l++) owners.push(`${region}:${t}:${l}`); }
  return { owners, count };
}

test('list-driven body authority surfaces exactly the owners the per-eighth scan did', () => {
  const next = random(0x5eed);
  for (let trial = 0; trial < 200; trial++) {
    const fill = next(), fineShare = next(), nearShare = next() * 0.3;
    const regions: Region[] = Array.from({ length: 1 + Math.floor(next() * 40) }, () => ({
      live: Array.from({ length: 64 }, () => next() < fill),
      width: Array.from({ length: 64 }, () => (next() < fineShare ? 1 : 4)),
      near: Array.from({ length: 27 }, (_, k) => (next() < nearShare ? 1 << k : 0)).reduce((a, b) => a | b, 0) >>> 0,
    }));
    // The ownership's owner capacity counts 64 per live h tile and one per live 4h tile.
    const owners = regions.reduce((sum, r) => sum + r.live.reduce((s, live, t) => s + (live ? (r.width[t] === 1 ? 64 : 1) : 0), 0), 0);
    const capacity = Math.max(1, Math.ceil(owners / 64));
    const before = eighths(regions), after = listed(regions, capacity, next);
    assert.ok(after.count <= capacity, `trial ${trial}: ${after.count} listed h tiles exceed ${capacity}`);
    assert.equal(new Set(after.owners).size, after.owners.length, `trial ${trial}: an owner surfaced twice`);
    assert.deepEqual([...after.owners].sort(), [...before].sort(), `trial ${trial}`);
  }
});

test('an empty body reach lists no h tile, so the indirect launch has zero workgroups', () => {
  const regions: Region[] = [{ live: Array(64).fill(true), width: Array(64).fill(1), near: 0 }];
  const { owners, count } = listed(regions, 64, random(1));
  assert.equal(count, 0); assert.equal(owners.length, 0); assert.deepEqual(eighths(regions), []);
});
