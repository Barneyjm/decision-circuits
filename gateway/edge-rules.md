# Edge rules

Cloudflare sits in front of both hostnames. Its bot protection is written for
websites, and an API is not a website: the clients are scripts, and some of
them get refused for looking like scripts. These rules carve out the paths
that machines are supposed to fetch. They live in the dashboard (Security →
WAF → Custom rules) and are written down here because otherwise the only copy
is in someone's memory.

## 1. API clients skip browser checks

| | |
|---|---|
| Expression | `(http.host eq "api.decisioncircuits.com")` |
| Action | Skip → Browser Integrity Check, and Super Bot Fight Mode if enabled |
| Order | First |

**Why.** Browser Integrity Check answers `Python-urllib/3.x` with a 403 before
the request reaches the Worker. Measured against the live API, identical
requests differing only in User-Agent:

```
Python-urllib/3.12       403      <- blocked at the edge, no body, nothing to act on
python-requests/2.32.3   401      <- normal auth failure
curl/8.7.1               401
decision-circuits/0.4    401
no User-Agent at all     401
```

This is not hypothetical: the first outside user of the API spent their first
hour on 60 of those 403s before getting through. The API's real limits are in
the Worker — 60 questions a minute per key, 600 across all keys — and those are
unaffected by this rule. The website keeps every protection it has.

## 2. Hosted sample media is fetchable by machines

| | |
|---|---|
| Expression | `(http.host eq "decisioncircuits.com" and starts_with(http.request.uri.path, "/bench/"))` |
| Action | Skip → Browser Integrity Check |
| Order | After rule 1 |

**Why.** `/bench/call.wav` is in the published examples as an audio state, so
the thing fetching it is a model server or somebody's script, never a browser.
Our own vision and audio endpoints could not read it until the service started
sending a named User-Agent, and every other caller has the same problem.

Everything outside `/bench/` on the website is untouched.

## Checking them

```bash
curl -s -o /dev/null -w '%{http_code}\n' -A 'Python-urllib/3.12' \
  -X POST https://api.decisioncircuits.com/v1/systemone -d '{}'      # want 401, not 403
curl -s -o /dev/null -w '%{http_code}\n' -A 'Python-urllib/3.12' \
  https://decisioncircuits.com/bench/call.wav                        # want 200
```

A 403 from either means the rule is missing, disabled, or ordered below
something that matches first.
