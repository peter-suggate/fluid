/** Exact-coordinate sparse directory hash shared by fluid and solid worlds. */
export function signedSpatialCoordinateHash(
  coordinate: readonly [number, number, number],
  discriminator = 0,
): number {
  if (!coordinate.every((value) => Number.isSafeInteger(value)
    && value >= -0x8000_0000 && value <= 0x7fff_ffff)
    || !Number.isSafeInteger(discriminator)
    || discriminator < 0 || discriminator > 0xffff_ffff) {
    throw new RangeError("Signed spatial hash input must fit the i32/u32 ABI");
  }
  let hash = 0x811c_9dc5;
  for (const value of [...coordinate, discriminator]) {
    hash = Math.imul(hash ^ (value >>> 0), 0x0100_0193) >>> 0;
    hash ^= hash >>> 16;
  }
  return (hash | 1) >>> 0;
}
