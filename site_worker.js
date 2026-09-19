// One canonical host. Everything else (www, http) redirects to https://decisioncircuits.com; static assets serve the rest, with a real 404.
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.hostname !== "decisioncircuits.com" || url.protocol !== "https:") {
      url.hostname = "decisioncircuits.com"; url.protocol = "https:";
      return Response.redirect(url.toString(), 301);
    }
    return env.ASSETS.fetch(request);
  },
};
