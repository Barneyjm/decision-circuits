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

## 2. What is published for machines is fetchable by machines

| | |
|---|---|
| Expression | `(http.host eq "decisioncircuits.com" and (starts_with(http.request.uri.path, "/bench/") or http.request.uri.path in {"/skill.md" "/llms.txt" "/robots.txt" "/sitemap.xml"}))` |
| Action | Skip → Browser Integrity Check (and Super Bot Fight Mode) |
| Order | After rule 1 |

**Why.** Every path here exists for something that is not a browser.
`/bench/call.wav` is the audio state in the published example, fetched by
whichever model server the caller points at it — our own vision and audio
endpoints could not read it until the service started sending a named
User-Agent. `skill.md` is written for agents and tells them how to sign up;
`llms.txt` exists only so automated readers find the docs; `robots.txt` and
`sitemap.xml` are for crawlers by definition. All four answered 403 to a plain
client until this rule, which is the whole failure in one line: the edge
refusing the exact clients a resource was published for.

The rest of the website — the front page, `/terms`, the assets — is untouched
and still answers 403 to `Python-urllib`, which is how you can tell the rule is
scoped rather than global.

## Checking them

```bash
ua='Python-urllib/3.12'
curl -s -o /dev/null -w '%{http_code}\n' -A "$ua" \
  -X POST https://api.decisioncircuits.com/v1/systemone -d '{}'   # want 401, not 403
for p in /skill.md /llms.txt /robots.txt /sitemap.xml /bench/call.wav; do
  curl -s -o /dev/null -w "%{http_code} $p\n" -A "$ua" "https://decisioncircuits.com$p"   # want 200
done
curl -s -o /dev/null -w '%{http_code}\n' -A "$ua" https://decisioncircuits.com/   # want 403: still protected
```

Measured after both rules were in place, a script using Python's `urllib` with
no User-Agent of its own read the skill, signed itself up, asked a question and
thresholded the answer — four steps from a bare URL, no human anywhere in it.

A 403 from either means the rule is missing, disabled, or ordered below
something that matches first.
