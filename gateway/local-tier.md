# The local tier: covering Modal's cold start with a machine at home

Modal scales to zero, which is why the API costs almost nothing to run and
why the first call after a quiet spell waits about a minute for a GPU. That
minute is the worst thing about the free API. This closes it with hardware
you already own, without turning that hardware into production.

**How it works.** Home answers first, and a GPU is rented only when home can't
take the work:

| what happens at home | where the question goes |
|---|---|
| it answers | done — no GPU is woken |
| it declines: at capacity (503 with `x-s1-busy`) | Modal, immediately |
| it is unreachable | Modal, immediately |
| it is merely slow | Modal starts after `LOCAL_PATIENCE_MS` (4 s); first answer wins |

So occasional traffic costs nothing and waits for nothing, and a burst that
outruns one machine spills onto GPUs rather than queueing behind it.

`x-circuit-served-by: modal | local` on every response says which answered.

Tailscale can't do this job: the Workers runtime has no WireGuard, so the
gateway cannot reach a tailnet address. Tailscale Funnel would work, but it
puts a public `ts.net` hostname and a DERP relay hop in the path. A
Cloudflare Tunnel keeps the traffic inside the network the gateway already
runs in.

## 1. Serve the models on the Mac

One process per model, each on its own port, from the circuit repo with the
weights already in `runs/`. `S1_API_KEY` must match the gateway's secret.

```bash
cd ~/Documents/code/s1-proto
export S1_API_KEY=...                                    # same value as the Worker secret
S1_MODEL=lora:runs/circuit-8b PORT=8902 \
  S1_MAX_INFLIGHT=3 S1_CONCURRENCY=1 S1_QUEUE_WAIT_S=8 \
  uv run python -m s1proto &
curl -s localhost:8902/healthz
# {"ok":true,"model":"lora:circuit-8b","inflight":0,"max_inflight":3,"concurrency":1}
```

Host only the models worth covering. Each one holds its weights in memory
for as long as it runs, and the audio and vision models are the largest.

**The three numbers are the whole capacity policy.** `S1_CONCURRENCY` is how
many forward passes run at once: **leave it at 1 on Apple silicon**. Two
concurrent passes on MPS do not run twice as fast, they hang, holding both
requests until the process is killed — which then makes the box permanently
"busy" and sends everything to Modal. `S1_MAX_INFLIGHT` is running plus
waiting, past which the box declines instantly, and `S1_QUEUE_WAIT_S` caps how
long a queued request waits for its turn before it declines too. Measured on
one Mac at 3/1/8, a burst of eight arrived as three served locally in about
three seconds each and five overflowed to Modal.

`deploy/serve_local.sh circuit-8b 8902` in the circuit repo is the same thing
with the defaults baked in.

**Surviving a reboot is awkward on macOS, and the reason is worth knowing.** A
launchd agent cannot open anything under `~/Documents` — that directory is
protected by the privacy system, and an agent has none of the consent a
terminal has. Measured with a probe agent: it could stat the files and could
not list the directory, and `zsh` reported `can't open input file` for a script
sitting right there. So a plain launchd job pointed at a repo in `~/Documents`
crash-loops, which is what happened here before anyone worked out why.

Three ways out, in the order I would consider them:

1. Keep the weights and the runner outside `~/Documents` — a copy of `runs/`
   under `~/Library/Application Support/` with `S1_MODEL` pointing at it — and
   let launchd start that. Nothing needs a privacy exception.
2. Grant Full Disk Access to whatever launchd executes. It works and it is one
   checkbox, but granting it to `/bin/zsh` grants it to every script anything
   runs through zsh.
3. Start it by hand after a reboot, which is what `nohup deploy/serve_local.sh`
   is for. The gateway treats an absent local tier as overflow and sends the
   questions to Modal, so the cost of forgetting is money, not downtime.

## 2. Put a tunnel in front of them

```bash
brew install cloudflared
cloudflared tunnel login                                  # opens a browser; pick decisioncircuits.com
cloudflared tunnel create circuit-mac
cloudflared tunnel route dns circuit-mac mac-8b.decisioncircuits.com
cloudflared tunnel route dns circuit-mac mac-vl-4b.decisioncircuits.com
```

`~/.cloudflared/config.yml`:

```yaml
tunnel: circuit-mac
credentials-file: /Users/jbarney/.cloudflared/<TUNNEL-ID>.json
ingress:
  - hostname: mac-8b.decisioncircuits.com
    service: http://localhost:8902
  - hostname: mac-vl-4b.decisioncircuits.com
    service: http://localhost:8903
  - service: http_status:404
```

```bash
cloudflared tunnel run circuit-mac        # then: sudo cloudflared service install
```

Use hostnames the gateway does not serve. Pointing a tunnel at
`api.decisioncircuits.com` makes the Worker fetch itself.

## 3. Lock it to the gateway

Without this, anyone who learns the hostname can run your GPU at home.

1. Zero Trust → Access → Service Auth → create a service token, `circuit-gateway`.
2. Zero Trust → Access → Applications → self-hosted, domain `mac-8b.decisioncircuits.com`
   (one per hostname), policy action **Service Auth**, rule: that token.
3. Give the Worker the token:

```bash
npx wrangler secret put LOCAL_ACCESS_ID -c gateway/wrangler.toml
npx wrangler secret put LOCAL_ACCESS_SECRET -c gateway/wrangler.toml
```

The gateway sends them as `CF-Access-Client-Id` / `CF-Access-Client-Secret`
alongside the model's own bearer token. Two locks, both required.

## 4. Turn it on

In `gateway/wrangler.toml`, name the models the Mac serves:

```toml
LOCAL_URLS = '{"circuit-8b":"https://mac-8b.decisioncircuits.com","circuit-vl-4b":"https://mac-vl-4b.decisioncircuits.com"}'
```

```bash
npx wrangler deploy -c gateway/wrangler.toml
```

## 5. Check it

With Modal cold (leave it more than two minutes), ask a question and watch
the header:

```bash
curl -si https://api.decisioncircuits.com/v1/systemone \
  -H "Authorization: Bearer dc-..." -H "Content-Type: application/json" \
  -d '{"model":"circuit-8b","state":"the pipe burst on Elm","questions":{"urgent":{"type":"noul","instructions":"Does this need someone today?"}}}' \
  | grep -i 'x-circuit-served-by\|HTTP/'
```

First call `served-by: local` in a couple of seconds. Repeat within two
minutes: `served-by: modal` in well under one. That is the whole feature.

## Turning it off

Set `LOCAL_URLS = '{}'` and deploy. The Mac can also simply go to sleep —
an unreachable tunnel costs one failed subrequest inside a window the caller
was already going to spend waiting.

## What this is not

Not a load tier and not a failover for real traffic. It serves strangers'
request bodies — receipts, recordings, whatever people upload — from a
machine in your house. That is a reasonable trade for a free research API
and the wrong one the moment real customer data shows up.
