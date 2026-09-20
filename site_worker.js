// One canonical host. Everything else (www, http) redirects to https://decisioncircuits.com;
// static assets serve the rest, with a real 404.
//
// Section paths: /scoreboard is the front page served at that URL, not a redirect, because
// the places people paste links (Threads, LinkedIn, Slack) strip #fragments. The page reads
// the path on load and scrolls to the matching section.
const SECTIONS = new Set(["family", "quiz", "api", "try", "write", "numbers", "scoreboard", "agents", "models"]);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.hostname !== "decisioncircuits.com" || url.protocol !== "https:") {
      url.hostname = "decisioncircuits.com";
      url.protocol = "https:";
      return Response.redirect(url.toString(), 301);
    }
    const seg = url.pathname.replace(/^\/+|\/+$/g, "");
    if (SECTIONS.has(seg)) {
      const res = await env.ASSETS.fetch(new Request(new URL("/", url), request));
      return new Response(res.body, res);
    }
    return env.ASSETS.fetch(request);
  },
};
