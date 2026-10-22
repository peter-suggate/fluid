import { resolve, relative, sep } from "node:path";
import type { Plugin } from "vite";
import { generateAssetCatalog } from "../tools/asset-catalog";

/** Install every authored collection at startup and rediscover newly exported
 * files while developing. Reload the page when its installation changes so
 * retained workers and GPU tables cannot use different asset ordinals. */
export function assetCatalog(): Plugin {
  return {
    name: "fluid-asset-catalog",
    enforce: "pre",
    async configResolved(config) { await generateAssetCatalog(config.root); },
    configureServer(server) {
      const root = server.config.root;
      server.watcher.add([resolve(root, "assets"), resolve(root, "public/assets"), resolve(root, "lib/core/imported-assets/metadata")]);
      let timer: ReturnType<typeof setTimeout> | undefined;
      let generation = Promise.resolve();
      let metadataChanged = false;
      const changed = (file: string) => {
        const path = relative(root, file).split(sep).join("/");
        const relevant = (path.startsWith("public/assets/") && /(?:manifest\.json|\.glb|\.fvx\.gz)$/.test(path))
          || (path.startsWith("assets/") && /\.blend$/.test(path))
          || (path.startsWith("lib/core/imported-assets/metadata/") && /-volume\.json$/.test(path));
        if (!relevant) return;
        metadataChanged ||= path.startsWith("lib/core/imported-assets/metadata/");
        clearTimeout(timer);
        timer = setTimeout(() => {
          const reloadMetadata = metadataChanged;
          metadataChanged = false;
          generation = generation.then(async () => {
            const result = await generateAssetCatalog(root);
            // Packed volume ordinals belong to one installation. Retained GPU
            // tables cannot coexist with a changed catalog in another module.
            if (result.changed || reloadMetadata) server.ws.send({ type: "full-reload" });
          }).catch((error: unknown) => {
            server.config.logger.error(`Asset discovery: ${error instanceof Error ? error.message : String(error)}`);
          });
        }, 150);
      };
      for (const event of ["add", "change", "unlink"] as const) server.watcher.on(event, changed);
      server.httpServer?.once("close", () => {
        clearTimeout(timer);
        for (const event of ["add", "change", "unlink"] as const) server.watcher.off(event, changed);
      });
    },
  };
}
