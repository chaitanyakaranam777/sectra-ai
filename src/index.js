export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (!url.pathname.startsWith("/api/")) {
      return env.ASSETS.fetch(request);
    }

    if (!env.BACKEND_URL) {
      return Response.json({ error: "Backend tunnel is not configured." }, { status: 503 });
    }

    try {
      const backend = new URL(env.BACKEND_URL);
      const target = new URL(request.url);

      target.protocol = backend.protocol;
      target.host = backend.host;

      const headers = new Headers(request.headers);
      headers.delete("host");
      headers.delete("origin");

      const init = {
        method: request.method,
        headers,
        redirect: "manual"
      };

      if (request.method !== "GET" && request.method !== "HEAD") {
        init.body = request.body;
      }

      const upstreamRequest = new Request(target.toString(), init);
      const response = await fetch(upstreamRequest);

      const responseHeaders = new Headers(response.headers);
      responseHeaders.set("Cache-Control", "no-store");

      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: responseHeaders
      });

    } catch (error) {
      return Response.json({
        error: "FastAPI backend connection failed.",
        details: error.message
      }, { status: 502 });
    }
  }
};
