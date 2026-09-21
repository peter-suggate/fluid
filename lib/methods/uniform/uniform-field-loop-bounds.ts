/** Fixed algorithmic trip counts supplied through immutable bind-group metadata.
 * Keeping these out of shader constants prevents Metal from expanding nested
 * quadrature/stencil loops into many copies of page/geometry sampling code.
 * Values and iteration order are identical to the original literal loops.
 */
export const UNIFORM_FIELD_LOOP_BOUNDS = [2, 3, 6, 8] as const;
// Binding 34 owns the layout uniform; entry 35 is reserved for these counts.
export const UNIFORM_FIELD_LOOP_METADATA = 35;
export function uniformFieldRuntimeLoops(source: string): string {
  return source.replace(
    /(for\s*\(\s*var\s+(\w+)\s*=\s*\d+u?;\s*\2\s*<\s*)([23468])(u?)(\s*;)/g,
    (_match, prefix: string, _variable: string, bound: string, unsigned: string, end: string) => {
      const index = UNIFORM_FIELD_LOOP_BOUNDS.indexOf(Number(bound) as 2|3|6|8);
      const value = bound === "4"
        ? `(uniformFieldPages[${UNIFORM_FIELD_LOOP_METADATA}][0] * 2u)`
        : `uniformFieldPages[${UNIFORM_FIELD_LOOP_METADATA}][${index}]`;
      return `${prefix}${unsigned ? value : `i32(${value})`}${end}`;
    },
  );
}
