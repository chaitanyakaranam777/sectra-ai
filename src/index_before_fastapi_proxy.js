const MODEL = "@cf/huggingface/distilbert-sst-2-int8";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    if (!path.startsWith("/api/")) {
      return env.ASSETS.fetch(request);
    }

    const json = (data, status = 200) =>
      Response.json(data, {
        status,
        headers: { "Cache-Control": "no-store" }
      });

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204 });
    }

    if (path === "/api/health") {
      return json({
        status: "online",
        service: "SENTRA AI Cloudflare API",
        provider: "Cloudflare Workers AI",
        model: MODEL,
        device: "Cloudflare Workers AI",
        database: "Not connected",
        torch: "Not running in Worker",
        cuda_available: false
      });
    }

    if (path === "/api/models") {
      return json({
        models: [{
          name: "Cloudflare DistilBERT SST-2",
          modality: "TEXT",
          description: "Cloudflare-hosted binary sentiment model. Separate from the locally trained SENTRA CNN+BERT model.",
          runs: []
        }]
      });
    }

    if (path === "/api/predict/text" && request.method === "POST") {
      try {
        const body = await request.json();
        const text = typeof body.text === "string" ? body.text.trim() : "";

        if (!text) {
          return json({ error: "Text is required." }, 400);
        }

        if (text.length > 5000) {
          return json({ error: "Text must be 5000 characters or fewer." }, 400);
        }

        const result = await env.AI.run(MODEL, { text });

        const predictions = Array.isArray(result) ? result : [result];
        const best = predictions.reduce((a, b) =>
          (b?.score ?? 0) > (a?.score ?? 0) ? b : a
        );

        return json({
          sentiment: best?.label ?? "Unknown",
          score: best?.score ?? null,
          model: MODEL,
          provider: "Cloudflare Workers AI",
          note: "This prediction uses Cloudflare's hosted DistilBERT model, not the locally trained SENTRA CNN+BERT checkpoint.",
          raw: result
        });
      } catch (error) {
        return json({
          error: "Sentiment prediction failed.",
          details: error.message
        }, 500);
      }
    }

    const unavailable = {
      error: "This endpoint is not connected to the SENTRA backend yet.",
      status: "not_configured"
    };

    if (
      path === "/api/datasets" ||
      path === "/api/training/runs" ||
      path === "/api/evaluation/comparison"
    ) {
      return json(unavailable, 501);
    }

    return json({ error: "API route not found." }, 404);
  }
};
