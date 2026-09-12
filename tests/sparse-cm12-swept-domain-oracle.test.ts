import assert from "node:assert/strict";
import test from "node:test";
import {
  exactSweptDomainPages2D,
  exactSweptDomainPages3D,
  extrudedDonors3D,
  SWEPT_DOMAIN_PAGE_FINE_SIZE,
  SWEPT_DOMAIN_ORACLE_CASES,
  SWEPT_DOMAIN_ORACLE_CASES_3D,
  type SweptDomainDonor2,
  type SweptDomainDonor3,
  type SweptDomainPage2,
} from
  "./support/sparse-cm12-swept-domain-oracle";

for (const fixture of SWEPT_DOMAIN_ORACLE_CASES) test(
  `analytic swept-domain fixture ${fixture.id}`,
  () => assert.deepEqual(exactSweptDomainPages2D(fixture), fixture.expectedPages2D),
);

for (const fixture of SWEPT_DOMAIN_ORACLE_CASES_3D) test(
  `analytic 3-D swept-domain fixture ${fixture.id}`,
  () => assert.deepEqual(exactSweptDomainPages3D(fixture.donors), fixture.expectedPages3D),
);

test("true 3-D extrusion equals the independently evaluated 2-D sweep", () => {
  for (const fixture of SWEPT_DOMAIN_ORACLE_CASES) {
    const actual = exactSweptDomainPages3D(extrudedDonors3D(fixture));
    assert.deepEqual(actual, fixture.expectedPages2D.map(([x, y]) => [x, y, 0]), fixture.id);
    assert.ok(actual.every(page => page[2] === 0), `${fixture.id} leaked across z`);
  }
});

const translated = (donor: SweptDomainDonor2, pages: readonly SweptDomainPage2[],
  offset: SweptDomainPage2) => ({
  donor: {
    minimumFine: donor.minimumFine.map(
      (value, axis) => value + SWEPT_DOMAIN_PAGE_FINE_SIZE * offset[axis]!) as [number, number],
    maximumExclusiveFine: donor.maximumExclusiveFine.map(
      (value, axis) => value + SWEPT_DOMAIN_PAGE_FINE_SIZE * offset[axis]!) as [number, number],
    displacementFine: donor.displacementFine,
  },
  pages: pages.map(([x, y]) => [x + offset[0], y + offset[1]] as const),
});

test("page translation and reflection preserve the swept-domain geometry", () => {
  const fixture = SWEPT_DOMAIN_ORACLE_CASES.find(candidate => candidate.id === "diagonal-10")!;
  const moved = translated(fixture.donors[0]!, fixture.expectedPages2D, [-3, 2]);
  assert.deepEqual(exactSweptDomainPages2D({ donors: [moved.donor] }), moved.pages);

  const donor = fixture.donors[0]!;
  const reflected: SweptDomainDonor2 = {
    minimumFine: [-donor.maximumExclusiveFine[0], donor.minimumFine[1]],
    maximumExclusiveFine: [-donor.minimumFine[0], donor.maximumExclusiveFine[1]],
    displacementFine: [-donor.displacementFine[0], donor.displacementFine[1]],
  };
  const expected = fixture.expectedPages2D.map(([x, y]) => [-x - 1, y] as const)
    .sort((a, b) => a[1] - b[1] || a[0] - b[0]);
  assert.deepEqual(exactSweptDomainPages2D({ donors: [reflected] }), expected);
});

test("true 3-D sweep preserves signed page translation and central reflection", () => {
  const donor: SweptDomainDonor3 = {
    minimumFine: [8, 8, 8],
    maximumExclusiveFine: [16, 16, 16],
    displacementFine: [10, -2.5, 18],
  };
  const original = exactSweptDomainPages3D([donor]);
  const offset = [-4, 2, -3] as const;
  const moved: SweptDomainDonor3 = {
    minimumFine: donor.minimumFine.map((value, axis) =>
      value + SWEPT_DOMAIN_PAGE_FINE_SIZE * offset[axis]!) as [number, number, number],
    maximumExclusiveFine: donor.maximumExclusiveFine.map((value, axis) =>
      value + SWEPT_DOMAIN_PAGE_FINE_SIZE * offset[axis]!) as [number, number, number],
    displacementFine: donor.displacementFine,
  };
  assert.deepEqual(exactSweptDomainPages3D([moved]), original.map(([x, y, z]) =>
    [x + offset[0], y + offset[1], z + offset[2]]));

  const reflected: SweptDomainDonor3 = {
    minimumFine: donor.maximumExclusiveFine.map(value => -value) as [number, number, number],
    maximumExclusiveFine: donor.minimumFine.map(value => -value) as [number, number, number],
    displacementFine: donor.displacementFine.map(value => -value) as [number, number, number],
  };
  const expectedReflection = original.map(([x, y, z]) => [-x - 1, -y - 1, -z - 1] as const)
    .sort((a, b) => a[2] - b[2] || a[1] - b[1] || a[0] - b[0]);
  assert.deepEqual(exactSweptDomainPages3D([reflected]), expectedReflection);
});

test("disconnected geometry is the union of component sweeps without bridge pages", () => {
  const fixture = SWEPT_DOMAIN_ORACLE_CASES.find(candidate => candidate.id === "disconnected")!;
  const componentPages = fixture.donors.flatMap(
    donor => exactSweptDomainPages2D({ donors: [donor] }));
  const union = [...new Map(componentPages.map(page => [page.join(","), page])).values()]
    .sort((a, b) => a[1] - b[1] || a[0] - b[0]);
  assert.deepEqual(exactSweptDomainPages2D(fixture), union);
});

test("positive axis travel grows monotonically beyond one page", () => {
  const ids = ["axis-2.5", "axis-10", "axis-18"];
  const pages = ids.map(id => new Set(exactSweptDomainPages2D(
    SWEPT_DOMAIN_ORACLE_CASES.find(candidate => candidate.id === id)!).map(String)));
  for (let index = 1; index < pages.length; index += 1) {
    for (const page of pages[index - 1]!) assert.ok(pages[index]!.has(page), `${ids[index]} lost ${page}`);
    assert.ok(pages[index]!.size > pages[index - 1]!.size);
  }
});

test("half-open page contacts do not create zero-measure neighbours", () => {
  const page: SweptDomainDonor2 = {
    minimumFine: [8, 8], maximumExclusiveFine: [16, 16], displacementFine: [0, 0],
  };
  assert.deepEqual(exactSweptDomainPages2D({ donors: [page] }),
    [[1, 1]]);
  assert.deepEqual(exactSweptDomainPages2D({ donors: [{ ...page, displacementFine: [8, 0] }],
  }), [[1, 1], [2, 1]]);
});

test("half-open 3-D page contacts exclude endpoint-only pages", () => {
  const donor: SweptDomainDonor3 = {
    minimumFine: [8, 8, 8], maximumExclusiveFine: [16, 16, 16],
    displacementFine: [8, 8, 8],
  };
  const actual = exactSweptDomainPages3D([donor]);
  assert.equal(actual.length, 8);
  assert.ok(actual.every(page => page.every(coordinate => coordinate === 1 || coordinate === 2)));
});
