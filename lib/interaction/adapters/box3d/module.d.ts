declare module 'box3d-wasm/standard' {
  const load: (options?: Record<string, unknown>) => Promise<unknown>;
  export default load;
}
