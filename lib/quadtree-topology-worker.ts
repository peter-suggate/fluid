import { prepareQuadtreeProjectionCPU, type QuadtreeCPUPreparationInput } from "./webgpu-quadtree-tall-cell";

interface Request {
  id: number;
  input: QuadtreeCPUPreparationInput;
}

const scope = self as unknown as {
  onmessage: ((event: MessageEvent<Request>) => void) | null;
  postMessage(message: unknown): void;
};

scope.onmessage = (event: MessageEvent<Request>) => {
  const { id, input } = event.data;
  try {
    scope.postMessage({ id, value: prepareQuadtreeProjectionCPU(input) });
  } catch (error) {
    scope.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
};
