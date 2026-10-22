import { createAssetRegistry } from "./asset-registry";
import { bundledAssetPlugins } from "./bundled-asset-plugins";

/** Shared composition root for browser, workers and Node tools. The generated
 * plugins contain only metadata; heavy mesh/volume payloads load on demand. */
export const assetRegistry = createAssetRegistry(bundledAssetPlugins);
