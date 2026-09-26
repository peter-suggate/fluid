/** Sec. 3.4 row/column balancing, shared by the uniform stencil and seam proof.
 * Expressions are WGSL, with i the receiver and k the edge slot. Keeping the
 * storage ABI outside the arithmetic lets a seam have more than eight donors.
 * Native calls retain their fixed loop and exact arithmetic order.
 */
export interface VolumeRowExpressions {
  readonly count: string;
  readonly weight: string;
  readonly donor: string;
  readonly target: string;
  readonly storeWeight?: (value: string) => string;
}
export function volumeNormalizeRowsWGSL(row: VolumeRowExpressions): string {
  return `var sum=0.0;
  for(var k=0u;k<${row.count};k++){sum+=${row.weight};}
  let scale=${row.target}/max(sum,1e-20);
  for(var k=0u;k<${row.count};k++){let weight=${row.weight}*scale;
    ${row.storeWeight?.("weight") ?? `${row.weight}=weight;`}uvAddDonor(${row.donor},weight);}`;
}
export function volumeNormalizeDonorsWGSL(row: VolumeRowExpressions, sum: string,
  donorCapacity?: string): string {
  const value = `${row.weight}${donorCapacity ? `*${donorCapacity}` : ""}/max(sum,1e-20)`;
  const write = row.storeWeight ? row.storeWeight(value)
    : donorCapacity ? `${row.weight}=${value};` : `${row.weight}/=max(sum,1e-20);`;
  return `for(var k=0u;k<${row.count};k++){let donor=${row.donor};
    let sum=${sum};
    ${write}}`;
}
