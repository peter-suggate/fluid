import { prepareQuadtreeProjectionCPU, preparedProjectionTransferables, type QuadtreeCPUPreparationInput } from "./webgpu-quadtree-tall-cell";

interface Request {
  id: number;
  input: QuadtreeCPUPreparationInput;
}

const scope = self as unknown as {
  onmessage: ((event: MessageEvent<Request>) => void) | null;
  postMessage(message: unknown, transfer?: ArrayBuffer[]): void;
};

scope.onmessage = (event: MessageEvent<Request>) => {
  const { id, input } = event.data;
  try {
    const value = prepareQuadtreeProjectionCPU(input);
    scope.postMessage({ id, value }, preparedProjectionTransferables(value));
  } catch (error) {
    scope.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
};
