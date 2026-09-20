// Circuit API gateway: self-serve keys, per-key and global daily quotas, proxy to the Modal endpoints.
//
//   POST /v1/keys      {"email": "you@example.com"}   -> {"key": "dc-...", ...}   (key shown once; only its hash is stored)
//   POST /v1/systemone  Authorization: Bearer dc-...  -> the System One response from the chosen model
//   GET  /v1/models                                    -> the models this gateway serves
//
// Secrets: S1_API_KEY (the bearer both upstreams require), and for the local tier
// LOCAL_ACCESS_ID / LOCAL_ACCESS_SECRET (a Cloudflare Access service token).
//
// Tiers. Every key is "free" until someone writes in and says what they are doing;
// raising one is a hand edit, so there is no billing and no accounts to run:
//
//   npx wrangler kv key get "pfx:dc-XXXXXXX" --namespace-id <KEYS> --remote  # -> the hash
//   npx wrangler kv key get "key:<hash>" --namespace-id <KEYS> --remote      # -> the record
//   npx wrangler kv key put "key:<hash>" '<record with "tier":"pro">' --namespace-id <KEYS> --remote
//
// The pfx: entry exists so a key can be identified from the ten characters its owner
// can safely paste into an email, instead of from the key itself.
//
// Where a question goes. When LOCAL_URLS names a model, hardware at home answers it and
// no GPU is rented at all: the occasional question costs nothing and waits for nothing.
// Modal is the overflow, woken only when home cannot take the work —
//
//   home answers                -> done, and Modal stays asleep
//   home says it is at capacity -> Modal, immediately (it declines with 503 + x-s1-busy)
//   home is unreachable         -> Modal, immediately
//   home is merely slow         -> Modal starts after LOCAL_PATIENCE_MS, first answer wins
//
// so a burst that outruns one machine spills onto GPUs, and a quiet day never touches them.

const LOCAL_PATIENCE_MS = 4000; // home is answering, just slowly: past this, waking a GPU beats waiting
const LOCAL_TIMEOUT_MS = 20000; // home is asleep, updating, or gone: give up on it entirely

const json = (body, status = 200, headers = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", "access-control-allow-origin": "*", ...headers } });

const cors = () =>
  new Response(null, { status: 204, headers: { "access-control-allow-origin": "*", "access-control-allow-methods": "GET, POST, OPTIONS", "access-control-allow-headers": "authorization, content-type" } });

async function sha256(text) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// Per-minute allowance by tier. "free" is what self-serve gets; anything else is set by hand.
const TIERS = { free: "KEY_LIMIT", pro: "PRO_LIMIT" };
const TIER_RATE = { free: "60 questions a minute", pro: "1,200 questions a minute" };

const UPGRADE =
  "Email hello@decisioncircuits.com for a higher limit. Include the first ten characters of your key, " +
  "what you are building, how many questions a day and at peak per minute, and which models you use.";

function newKey() {
  const bytes = crypto.getRandomValues(new Uint8Array(24));
  return "dc-" + btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function issueKey(request, env) {
  const ip = request.headers.get("cf-connecting-ip") || "unknown";
  const { success: okIp } = await env.SIGNUP_LIMIT.limit({ key: ip });
  if (!okIp) return json({ error: "too many keys requested from this address; try again in a minute" }, 429);
  // An email is optional and kept only so a key has an owner to write to. Everything else
  // about a key record is the date it was issued, what it may spend, and its tier.
  let email = null;
  try {
    const body = await request.json();
    if (typeof body.email === "string" && /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(body.email)) email = body.email.trim().toLowerCase();
  } catch {}
  const key = newKey();
  const prefix = key.slice(0, 10);
  const record = { email, created: new Date().toISOString(), quota: parseInt(env.KEY_DAILY_QUOTA, 10), tier: "free", prefix };
  const hash = await sha256(key);
  await env.KEYS.put(`key:${hash}`, JSON.stringify(record));
  await env.KEYS.put(`pfx:${prefix}`, hash); // find a key from what its owner can safely paste
  return json({
    key,
    rate: TIER_RATE.free,
    upgrade: UPGRADE,
    models: Object.keys(JSON.parse(env.MODAL_URLS)),
    endpoint: "https://api.decisioncircuits.com/v1/systemone",
    terms: "https://decisioncircuits.com/terms",
  });
}

async function proxy(request, env, ctx) {
  const auth = request.headers.get("authorization") || "";
  if (!auth.toLowerCase().startsWith("bearer ")) return json({ error: "missing bearer token" }, 401);
  const key = auth.slice(7).trim();
  const rec = await env.KEYS.get(`key:${await sha256(key)}`, { type: "json" });
  if (!rec) return json({ error: "unknown api key" }, 401);
  const tier = TIERS[rec.tier] ? rec.tier : "free";
  const { success: okKey } = await env[TIERS[tier]].limit({ key: await sha256(key) });
  if (!okKey) return json({ error: `this key is over its rate (${TIER_RATE[tier]}); slow down`, tier, upgrade: UPGRADE }, 429, { "retry-after": "10" });
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
  const payloadText = JSON.stringify(body);

  const toModal = () => ask(urls[model], { authorization: `Bearer ${env.S1_API_KEY}` }, payloadText, 120000).then(tag("modal"));
  const localBase = localUrlFor(env, model);
  let answer = null;
  if (!localBase) {
    answer = await toModal().catch(() => null);
  } else {
    const local = ask(localBase, localHeaders(env), payloadText, LOCAL_TIMEOUT_MS).then(tag("local"));
    const waited = Symbol("slow");
    const first = await Promise.race([local.catch(() => null), new Promise((r) => setTimeout(() => r(waited), LOCAL_PATIENCE_MS))]);
    if (first && first !== waited) {
      answer = first; // home took it: no GPU was rented for this question
    } else if (first === waited) {
      answer = await Promise.any([local, toModal()]).catch(() => null); // slow, not refused: let both run
    } else {
      answer = await toModal().catch(() => null); // at capacity or unreachable
    }
  }
  if (!answer) return json({ error: "the model is starting up; retry in 30 s", model }, 503, { "retry-after": "30" });

  const headers = { "x-circuit-model": model, "x-circuit-rate": TIER_RATE[tier], "x-circuit-served-by": answer.via };
  if (answer.latency) headers["x-s1-latency-ms"] = answer.latency;
  if (answer.payload && answer.payload.request_id) headers["x-request-id"] = answer.payload.request_id;
  return json(answer.payload, answer.status, headers);
}

// The tunnel hostname for a model, when one is configured. An empty map turns the tier off.
function localUrlFor(env, model) {
  if (!env.LOCAL_URLS) return null;
  try {
    return JSON.parse(env.LOCAL_URLS)[model] || null;
  } catch {
    return null;
  }
}

// Access sits in front of the tunnel, so the service token rides along with the model's own bearer.
function localHeaders(env) {
  const h = { authorization: `Bearer ${env.S1_API_KEY}` };
  if (env.LOCAL_ACCESS_ID && env.LOCAL_ACCESS_SECRET) {
    h["cf-access-client-id"] = env.LOCAL_ACCESS_ID;
    h["cf-access-client-secret"] = env.LOCAL_ACCESS_SECRET;
  }
  return h;
}

const tag = (via) => (r) => ({ ...r, via });

// One upstream call. Rejects on anything that means "not answered" — a timeout, a dead
// host, a 5xx, a cold start's non-JSON body — so the hedge can take the other branch.
// A 4xx is an answer (a malformed question), and comes back as one.
async function ask(base, headers, body, timeoutMs) {
  const r = await fetch(`${base}/v1/systemone`, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body,
    redirect: "follow", // Modal answers a slow first boot with a 303 to the result; follow it
    signal: AbortSignal.timeout(timeoutMs),
  });
  const text = await r.text();
  if (r.status >= 500) throw new Error(`upstream ${r.status}`);
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    throw new Error("non-JSON body"); // the platform, not the model: a cold start that outran a timeout
  }
  return { status: r.status, payload, latency: r.headers.get("x-s1-latency-ms") };
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (request.method === "OPTIONS") return cors();
    if (url.pathname === "/v1/keys" && request.method === "POST") return issueKey(request, env);
    if (url.pathname === "/v1/systemone" && request.method === "POST") return proxy(request, env, ctx);
    if (url.pathname === "/v1/models") return json({ models: Object.keys(JSON.parse(env.MODAL_URLS)), default: env.DEFAULT_MODEL });
    if (url.pathname === "/") return json({ name: "circuit api", docs: "https://decisioncircuits.com", agent_skill: "https://decisioncircuits.com/skill.md", terms: "https://decisioncircuits.com/terms", signup: "POST /v1/keys", call: "POST /v1/systemone with Authorization: Bearer <key>" });
    return json({ error: "not found" }, 404);
  },
};
