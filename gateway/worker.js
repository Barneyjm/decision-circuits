// Circuit API gateway: self-serve keys, per-key and global daily quotas, proxy to the Modal endpoints.
//
//   POST /v1/keys      {"email": "you@example.com"}   -> {"key": "dc-...", ...}   (shown once; only its hash is stored)
//   POST /v1/systemone  Authorization: Bearer dc-...  -> the System One response from the chosen model
//   GET  /v1/models                                    -> the models this gateway serves
//
// Secrets: S1_API_KEY (the bearer the Modal services require).

const json = (body, status = 200, headers = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", "access-control-allow-origin": "*", ...headers } });

const cors = () =>
  new Response(null, { status: 204, headers: { "access-control-allow-origin": "*", "access-control-allow-methods": "GET, POST, OPTIONS", "access-control-allow-headers": "authorization, content-type" } });

async function sha256(text) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function newKey() {
  const bytes = crypto.getRandomValues(new Uint8Array(24));
  return "dc-" + btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function issueKey(request, env) {
  const ip = request.headers.get("cf-connecting-ip") || "unknown";
  const { success: okIp } = await env.SIGNUP_LIMIT.limit({ key: ip });
  if (!okIp) return json({ error: "too many keys requested from this address; try again in a minute" }, 429);
  let email = null;
  try {
    const body = await request.json();
    if (typeof body.email === "string" && /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(body.email)) email = body.email.trim().toLowerCase();
  } catch {}
  const key = newKey();
  const record = { email, created: new Date().toISOString(), quota: parseInt(env.KEY_DAILY_QUOTA, 10) };
  await env.KEYS.put(`key:${await sha256(key)}`, JSON.stringify(record));
  return json({ key, rate: "60 questions a minute", models: Object.keys(JSON.parse(env.MODAL_URLS)), endpoint: "https://api.decisioncircuits.com/v1/systemone" });
}

async function proxy(request, env) {
  const auth = request.headers.get("authorization") || "";
  if (!auth.toLowerCase().startsWith("bearer ")) return json({ error: "missing bearer token" }, 401);
  const key = auth.slice(7).trim();
  const rec = await env.KEYS.get(`key:${await sha256(key)}`, { type: "json" });
  if (!rec) return json({ error: "unknown api key" }, 401);
  const { success: okKey } = await env.KEY_LIMIT.limit({ key: await sha256(key) });
  if (!okKey) return json({ error: "this key is over its rate (60 questions a minute); slow down" }, 429, { "retry-after": "10" });
  const { success: okAll } = await env.GLOBAL_LIMIT.limit({ key: "all" });
  if (!okAll) return json({ error: "the free tier is at capacity right now; retry shortly" }, 503, { "retry-after": "10" });

  const bodyText = await request.text();
  let body;
  try {
    body = JSON.parse(bodyText);
  } catch {
    return json({ error: "body must be JSON" }, 400);
  }
  const urls = JSON.parse(env.MODAL_URLS);
  const model = body.model && urls[body.model] ? body.model : env.DEFAULT_MODEL;
  body.model = model;
  let upstream;
  try {
    upstream = await fetch(`${urls[model]}/v1/systemone`, {
      method: "POST",
      headers: { authorization: `Bearer ${env.S1_API_KEY}`, "content-type": "application/json" },
      body: JSON.stringify(body),
      redirect: "follow", // Modal answers a slow first boot with a 303 to the result; follow it
    });
  } catch (e) {
    return json({ error: "the model is starting up; retry in 30 s", model }, 503, { "retry-after": "30" });
  }
  const headers = { "x-circuit-model": model, "x-circuit-rate": "60/min per key" };
  const lat = upstream.headers.get("x-s1-latency-ms");
  if (lat) headers["x-s1-latency-ms"] = lat;
  const text = await upstream.text();
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    // a non-JSON body is the platform, not the model: a cold start that outran a timeout. Say so and ask for a retry.
    return json({ error: "the model is starting up; retry in 30 s", model, upstream_status: upstream.status }, 503, { ...headers, "retry-after": "30" });
  }
  return json(payload, upstream.status, headers);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "OPTIONS") return cors();
    if (url.pathname === "/v1/keys" && request.method === "POST") return issueKey(request, env);
    if (url.pathname === "/v1/systemone" && request.method === "POST") return proxy(request, env);
    if (url.pathname === "/v1/models") return json({ models: Object.keys(JSON.parse(env.MODAL_URLS)), default: env.DEFAULT_MODEL });
    if (url.pathname === "/") return json({ name: "circuit api", docs: "https://decisioncircuits.com", agent_skill: "https://decisioncircuits.com/skill.md", signup: "POST /v1/keys", call: "POST /v1/systemone with Authorization: Bearer <key>" });
    return json({ error: "not found" }, 404);
  },
};
