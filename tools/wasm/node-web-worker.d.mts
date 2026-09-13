export class NodeWebWorker {
  constructor(url: URL | string, options?: { type?: string; name?: string });
  postMessage(value: unknown, transfer?: readonly Transferable[]): void;
  addEventListener(type: string, listener: (event: MessageEvent) => void): void;
  removeEventListener(type: string, listener: (event: MessageEvent) => void): void;
  terminate(): void;
}
