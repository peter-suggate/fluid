/** Instrument a call without replacing its arguments, return value/promise,
 * or thrown error. Observer failures must not change application behavior. */
export function observeMethodCalls(owner: Record<string, any>, name: string,
  observer: (phase: "begin" | "end", outcome?: "returned" | "threw") => void): () => void {
  const descriptor = Object.getOwnPropertyDescriptor(owner, name);
  if (!descriptor || typeof descriptor.value !== "function") throw new Error(`Cannot observe method ${name}`);
  const original = descriptor.value;
  const notify = (phase: "begin" | "end", outcome?: "returned" | "threw") => {
    try { observer(phase, outcome); } catch { /* Diagnostics do not own the call. */ }
  };
  const wrapped = function (this: unknown, ...args: unknown[]) {
    notify("begin");
    let result: unknown;
    try { result = Reflect.apply(original, this, args); }
    catch (error) { notify("end", "threw"); throw error; }
    if (result instanceof Promise) {
      // Observe settlement but return the original promise identity. Both
      // handlers return normally so the observer creates no rejected promise.
      void result.then(() => notify("end", "returned"), () => notify("end", "threw"));
    } else notify("end", "returned");
    return result;
  };
  Object.defineProperty(owner, name, { ...descriptor, value: wrapped });
  return () => {
    if (owner[name] === wrapped) Object.defineProperty(owner, name, descriptor);
  };
}
