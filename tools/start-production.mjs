import process from "node:process";
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
server.prependListener("request", (_request, response) => {
  response.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  response.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  response.setHeader("Cross-Origin-Resource-Policy", "same-origin");
});
