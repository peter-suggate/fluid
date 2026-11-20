import process from "node:process";
import { open } from "node:fs/promises";
import { pipeline } from "node:stream/promises";
import { resolve } from "node:path";
import { startProdServer } from "vinext/server/prod-server";

const argument = (name) => {
  const prefix = `--${name}=`;
  const inline = process.argv.find(value => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : undefined;
};

const parsedPort = Number(argument("port") ?? process.env.PORT ?? 3000);
if (!Number.isSafeInteger(parsedPort) || parsedPort < 1 || parsedPort > 65535) {
  throw new RangeError("Production server port must be an integer from 1 through 65535");
}

const { server } = await startProdServer({
  port: parsedPort,
  host: argument("hostname") ?? "0.0.0.0",
  outDir: resolve("dist"),
});

// Vinext serves hashed `/assets/*` before the app worker can add response
// headers. Set the isolation policy at the Node response boundary so render,
// simulation and Rayon worker entry scripts receive the same policy as HTML
// and public Wasm artifacts. `_headers` remains the deployed static-host rule.
// Vinext indexes .gz as transport sidecars. FVX and FVM archives are
// themselves public files and the scene loader requests their literal URL.
const [serveApp] = server.listeners("request");
server.removeListener("request", serveApp);
server.on("request", async (request, response) => {
  response.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  response.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  response.setHeader("Cross-Origin-Resource-Policy", "same-origin");
  // Vinext's static MIME table currently omits Wasm. Its writeHead headers
  // override earlier setHeader calls, so correct the type at that boundary.
  const pathname = new URL(request.url ?? "/", "http://localhost").pathname;
  if ((request.method === "GET" || request.method === "HEAD")
    && /^\/assets\/(?:[\w-]+\/)+[\w-]+\.fv[xm]\.gz$/.test(pathname)) {
    let file;
    try {
      file = await open(resolve("dist/client", "." + pathname));
      const { size } = await file.stat();
      response.writeHead(200, { "Content-Type": "application/gzip", "Content-Length": size });
      if (request.method === "HEAD") response.end();
      else await pipeline(file.createReadStream({ autoClose: false }), response);
    } catch (error) {
      if (!response.headersSent) response.writeHead(error.code === "ENOENT" ? 404 : 500).end();
      else response.destroy();
    } finally {
      await file?.close();
    }
    return;
  }
  if (/^\/wasm\/fluid-wasm\/[^/]+\/fluid_wasm_bg\.wasm$/.test(pathname)) {
    const writeHead = response.writeHead;
    response.writeHead = function (statusCode, ...args) {
      if (statusCode === 200 || statusCode === 206) {
        const headersIndex = typeof args[0] === "string" ? 1 : 0;
        const headers = new Headers(args[headersIndex] ?? {});
        headers.set("Content-Type", "application/wasm");
        headers.set("Cache-Control", "no-cache");
        args[headersIndex] = Object.fromEntries(headers);
      }
      return writeHead.call(this, statusCode, ...args);
    };
  }
  serveApp.call(server, request, response);
});
