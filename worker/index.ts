import handler from "vinext/server/app-router-entry";

const worker = {
  async fetch(request: Request, env: Parameters<typeof handler.fetch>[1], context: Parameters<typeof handler.fetch>[2]) {
    const response = await handler.fetch(request, env, context);
    const headers = new Headers(response.headers);
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    headers.set("Cross-Origin-Embedder-Policy", "require-corp");
    headers.set("Cross-Origin-Resource-Policy", "same-origin");
    if (new URL(request.url).pathname.startsWith("/wasm/fluid-wasm/")) {
      // Artifact names are stable across builds. Revalidate glue, binaries and
      // Rayon helpers so a release cannot combine generations for an hour.
      headers.set("Cache-Control", "no-cache");
    }
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  }
};

export default worker;
