"""Self-contained demo web page for the serving app.

Everything the browser needs -- CSS, JavaScript, and an inline SVG map of
Poland -- is embedded in a single HTML string with no external CDN, font, or
image reference, so the page works behind an air-gap and adds no runtime
dependency. The two ingredients that carry real logic (the lat/lon -> SVG
projection and the whole-page builder) are plain pure functions so they can be
unit-tested without standing up a server.
"""

from __future__ import annotations

import html
import json

from tulip.labels.geo import POLAND_BOUNDS, REGION_CENTROIDS, GeoPoint
from tulip.labels.taxonomy import display_name

#: SVG canvas size for the map, in user units. The width/height ratio is chosen
#: to roughly match Poland's bounding box in degrees so the linear projection
#: does not visibly squash the country.
_MAP_WIDTH = 500
_MAP_HEIGHT = 291

#: Radius of an idle (pre-prediction) region marker, shared with the JS so the
#: client can scale up from the same baseline.
_BASE_RADIUS = 4


def project(
    point: GeoPoint,
    bounds: tuple[float, float, float, float],
    *,
    width: float,
    height: float,
) -> tuple[float, float]:
    """Linearly project a WGS84 coordinate onto an SVG canvas.

    Longitude maps to ``x`` (west edge -> 0), latitude maps to ``y`` with the
    axis inverted (north edge -> 0) because SVG's y grows downward while
    latitude grows upward. The projection is deliberately linear (no map
    reprojection): for a country the size of Poland at display scale the
    distortion is invisible, and a pure function is trivially testable.

    Args:
        point: The coordinate to place.
        bounds: ``(south, west, north, east)`` bounding box, as in
            :data:`tulip.labels.geo.POLAND_BOUNDS`.
        width: Canvas width in user units.
        height: Canvas height in user units.

    Returns:
        The ``(x, y)`` position in SVG user units.
    """
    south, west, north, east = bounds
    x = (point.lon - west) / (east - west) * width
    y = (north - point.lat) / (north - south) * height
    return (x, y)


def demo_page(*, title: str) -> str:
    """Render the complete, self-contained demo HTML page.

    Region markers are positioned server-side via :func:`project` so the SVG is
    correct even before any JavaScript runs; the client only rescales existing
    markers once a prediction returns.

    Args:
        title: Page ``<title>`` and heading text.

    Returns:
        A full HTML document as a string.
    """
    dots: list[str] = []
    region_names: dict[str, str] = {}
    for dialect, centroid in REGION_CENTROIDS.items():
        x, y = project(centroid, POLAND_BOUNDS, width=_MAP_WIDTH, height=_MAP_HEIGHT)
        value = dialect.value
        name = display_name(dialect)
        region_names[value] = name
        dots.append(
            f'<circle class="region-dot" data-region="{html.escape(value, quote=True)}" '
            f'cx="{x:.1f}" cy="{y:.1f}" r="{_BASE_RADIUS}">'
            f"<title>{html.escape(name)}</title></circle>"
        )

    return (
        _HTML_TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("__MAP_W__", str(_MAP_WIDTH))
        .replace("__MAP_H__", str(_MAP_HEIGHT))
        .replace("__BASE_R__", str(_BASE_RADIUS))
        .replace("__DOTS__", "\n".join(dots))
        .replace("__REGIONS__", json.dumps(region_names, ensure_ascii=False))
    )


# The template is a plain (non-f) string: its CSS and JS contain literal braces,
# and only the "__UPPER__" sentinels are substituted in demo_page(). Keeping the
# markup here rather than concatenating fragments keeps the page readable.
_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #0f1220; --panel: #181c2e; --ink: #eef1ff; --muted: #9aa3c7;
    --accent: #6d8bff; --top: #ffd166; --bar: #2b3358; --land: #232a47;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  }
  main { max-width: 960px; margin: 0 auto; padding: 24px 20px 64px; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  p.sub { color: var(--muted); margin: 0 0 24px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid #2a3153; border-radius: 14px; padding: 18px; }
  form { display: flex; gap: 8px; margin-bottom: 14px; }
  textarea {
    flex: 1; min-height: 84px; resize: vertical; padding: 10px 12px; border-radius: 10px;
    border: 1px solid #333c66; background: #10152a; color: var(--ink); font: inherit;
  }
  button {
    align-self: flex-start; padding: 10px 18px; border: 0; border-radius: 10px;
    background: var(--accent); color: #0b1020; font-weight: 600; cursor: pointer;
  }
  button:disabled { opacity: .5; cursor: progress; }
  .verdict { min-height: 26px; margin: 4px 0 14px; font-weight: 600; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 999px;
    background: #3a2b12; color: var(--top); font-size: .8rem; }
  .bars { display: flex; flex-direction: column; gap: 8px; }
  .row { display: grid; grid-template-columns: 130px 1fr 52px; align-items: center; gap: 10px; }
  .row .name {
    color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .track { height: 12px; border-radius: 6px; background: var(--bar); overflow: hidden; }
  .fill { height: 100%; width: 0; background: var(--accent); transition: width .35s ease; }
  .row.top .fill { background: var(--top); }
  .row.top .name { color: var(--ink); }
  .pct { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
  svg { width: 100%; height: auto; display: block; }
  .land { fill: var(--land); stroke: #39406b; stroke-width: 1; }
  .region-dot { fill: var(--accent); opacity: .28; transition: r .35s ease, opacity .35s ease; }
  .region-dot.top { fill: var(--top); }
  .hint { color: var(--muted); font-size: .85rem; margin-top: 10px; }
</style>
</head>
<body>
<main>
  <h1>__TITLE__</h1>
  <p class="sub">Type a Polish sentence and see which regional dialect the model predicts.</p>
  <div class="grid">
    <section class="card">
      <form id="classify-form">
        <textarea id="text-input" placeholder="Hej, baca sie pyto, kaj sie owce pasa na holi."
          >Hej, baca się pyto, kaj się owce pasą na holi.</textarea>
        <button id="classify-btn" type="submit">Classify</button>
      </form>
      <div class="verdict" id="verdict"></div>
      <div class="bars" id="bars"></div>
    </section>
    <section class="card">
      <svg viewBox="0 0 __MAP_W__ __MAP_H__" role="img" aria-label="Map of Polish dialect regions">
        <rect class="land" x="1" y="1" width="__MAP_W__" height="__MAP_H__" rx="16"
          transform="translate(-1,-1)"></rect>
        __DOTS__
      </svg>
      <p class="hint">Marker size and brightness track the predicted probability
        for each region.</p>
    </section>
  </div>
</main>
<script>
  const REGIONS = __REGIONS__;
  const BASE_R = __BASE_R__;
  const form = document.getElementById("classify-form");
  const input = document.getElementById("text-input");
  const btn = document.getElementById("classify-btn");
  const verdict = document.getElementById("verdict");
  const bars = document.getElementById("bars");

  function prettify(label) {
    if (REGIONS[label]) return REGIONS[label];
    return label.replace(/_/g, " ").replace(/\\b\\w/g, (c) => c.toUpperCase());
  }

  function renderBars(probs, abstained) {
    bars.textContent = "";
    probs.forEach((p, i) => {
      const row = document.createElement("div");
      row.className = "row" + (i === 0 && !abstained ? " top" : "");
      const pct = (p.probability * 100).toFixed(1);
      // Build via textContent/.title, never innerHTML: p.label is a model class
      // name that can contain arbitrary characters (a dataset controls it), so
      // concatenating it into markup would be an HTML/JS injection sink.
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = prettify(p.label);
      name.title = prettify(p.label);
      const track = document.createElement("span");
      track.className = "track";
      const fill = document.createElement("span");
      fill.className = "fill";
      fill.style.width = pct + "%";
      track.appendChild(fill);
      const pctEl = document.createElement("span");
      pctEl.className = "pct";
      pctEl.textContent = pct + "%";
      row.appendChild(name);
      row.appendChild(track);
      row.appendChild(pctEl);
      bars.appendChild(row);
    });
  }

  function updateMap(probs, abstained) {
    const byLabel = {};
    probs.forEach((p) => { byLabel[p.label] = p.probability; });
    const topLabel = abstained ? null : (probs[0] && probs[0].label);
    document.querySelectorAll(".region-dot").forEach((dot) => {
      const value = byLabel[dot.dataset.region] || 0;
      dot.setAttribute("r", (BASE_R + value * 18).toFixed(1));
      dot.style.opacity = (0.22 + value * 0.78).toFixed(2);
      dot.classList.toggle("top", dot.dataset.region === topLabel);
    });
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = input.value;
    btn.disabled = true;
    verdict.textContent = "Classifying...";
    try {
      const res = await fetch("/predict/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        verdict.textContent = "Error: " + (err.detail || res.status);
        return;
      }
      const data = await res.json();
      if (data.abstained) {
        verdict.innerHTML = '<span class="badge">abstained</span> too uncertain to label';
      } else {
        const conf = data.probabilities.length
          ? (data.probabilities[0].probability * 100).toFixed(1)
          : "0";
        verdict.textContent = "Predicted: " + prettify(data.label) + " (" + conf + "%)";
      }
      renderBars(data.probabilities || [], data.abstained);
      updateMap(data.probabilities || [], data.abstained);
    } catch (e) {
      verdict.textContent = "Request failed: " + e;
    } finally {
      btn.disabled = false;
    }
  });
</script>
</body>
</html>
"""


__all__ = ["demo_page", "project"]
