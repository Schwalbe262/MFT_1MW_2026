from __future__ import annotations

import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from rl_mft.records import INSIGHT_PATH, NOTE_PATH, append_token_usage, load_state, read_token_usage


def read_tail(path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[-lines:])


def token_chart(points: list[dict]) -> str:
    if not points:
        return "<p>No token usage records yet.</p>"
    width = 760
    height = 220
    pad = 28
    totals = [max(0, int(point.get("total_tokens", 0))) for point in points]
    max_total = max(totals) or 1
    if len(totals) == 1:
        coords = [(width // 2, height - pad - int((totals[0] / max_total) * (height - 2 * pad)))]
    else:
        coords = [
            (
                pad + int(index * ((width - 2 * pad) / (len(totals) - 1))),
                height - pad - int((total / max_total) * (height - 2 * pad)),
            )
            for index, total in enumerate(totals)
        ]
    polyline = " ".join(f"{x},{y}" for x, y in coords)
    circles = "".join(f"<circle cx='{x}' cy='{y}' r='3'></circle>" for x, y in coords)
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Token usage over time'>"
        f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}'></line>"
        f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height-pad}'></line>"
        f"<polyline points='{polyline}'></polyline>{circles}</svg>"
    )


def token_summary(points: list[dict]) -> list[dict]:
    totals: dict[tuple[str, str, str], int] = {}
    for point in points:
        key = (point.get("provider", ""), point.get("project", ""), point.get("reset_cycle", ""))
        totals[key] = totals.get(key, 0) + int(point.get("total_tokens", 0))
    return [
        {"provider": key[0], "project": key[1], "reset_cycle": key[2], "total_tokens": value}
        for key, value in sorted(totals.items())
    ]


def page() -> str:
    state = load_state()
    tokens = read_token_usage()
    loops = list(reversed(state.get("loops", [])[-20:]))
    loop_rows = "\n".join(
        f"<tr><td>{loop.get('loop')}</td><td>{html.escape(str(loop.get('status')))}</td><td>{html.escape(str(loop.get('backend')))}</td>"
        f"<td>{loop.get('completed')}</td><td>{loop.get('failed')}</td><td>{loop.get('best_reward')}</td>"
        f"<td>{html.escape(str(loop.get('best_candidate_id') or ''))}</td></tr>"
        for loop in loops
    )
    best = state.get("best_parameters") or {}
    best_rows = "\n".join(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>" for key, value in best.items())
    token_rows = "\n".join(
        f"<tr><td>{html.escape(str(item.get('recorded_at')))}</td><td>{html.escape(str(item.get('provider')))}</td>"
        f"<td>{html.escape(str(item.get('project')))}</td><td>{html.escape(str(item.get('reset_cycle')))}</td>"
        f"<td>{item.get('input_tokens')}</td><td>{item.get('output_tokens')}</td><td>{item.get('total_tokens')}</td>"
        f"<td>{html.escape(str(item.get('note')))}</td></tr>"
        for item in reversed(tokens[-50:])
    )
    token_summary_rows = "\n".join(
        f"<tr><td>{html.escape(item['provider'])}</td><td>{html.escape(item['project'])}</td>"
        f"<td>{html.escape(item['reset_cycle'])}</td><td>{item['total_tokens']}</td></tr>"
        for item in token_summary(tokens)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="20">
  <title>MFT RL Dashboard</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #1d252d; }}
    header {{ padding: 18px 24px; background: #18202a; color: white; }}
    main {{ padding: 20px 24px; display: grid; grid-template-columns: minmax(0, 2fr) minmax(320px, 1fr); gap: 18px; }}
    section {{ background: white; border: 1px solid #d9dee5; border-radius: 6px; padding: 16px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e6e9ee; padding: 8px; text-align: left; vertical-align: top; }}
    input {{ box-sizing: border-box; width: 100%; padding: 7px; border: 1px solid #cbd2da; border-radius: 4px; }}
    button {{ padding: 8px 12px; border: 0; border-radius: 4px; background: #244b75; color: white; cursor: pointer; }}
    form.grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; align-items: end; }}
    pre {{ white-space: pre-wrap; font-size: 13px; background: #f1f3f5; padding: 12px; overflow: auto; max-height: 460px; }}
    svg {{ width: 100%; height: 220px; }}
    svg line {{ stroke: #aeb8c4; stroke-width: 1; }}
    svg polyline {{ fill: none; stroke: #244b75; stroke-width: 2; }}
    svg circle {{ fill: #244b75; }}
    .metric {{ display: flex; gap: 24px; margin-top: 8px; }}
    .metric div {{ font-size: 14px; }}
    .metric strong {{ display: block; font-size: 22px; }}
  </style>
</head>
<body>
  <header>
    <h1>MFT RL Dashboard</h1>
    <div class="metric">
      <div><strong>{state.get('current_loop', 0)}</strong>Current loop</div>
      <div><strong>{state.get('best_reward', '-')}</strong>Best reward</div>
      <div><strong>{html.escape(str(state.get('best_candidate_id') or '-'))}</strong>Best candidate</div>
      <div><strong>{state.get('live_best_reward', '-')}</strong>Live best</div>
      <div><strong>{html.escape(str(state.get('live_best_candidate_id') or '-'))}</strong>Live candidate</div>
      <div><strong>{state.get('failure_rate', 0)}</strong>Failure rate</div>
    </div>
  </header>
  <main>
    <div>
      <section>
        <h2>Recent Loops</h2>
        <table>
          <thead><tr><th>Loop</th><th>Status</th><th>Backend</th><th>Done</th><th>Failed</th><th>Best Reward</th><th>Candidate</th></tr></thead>
          <tbody>{loop_rows}</tbody>
        </table>
      </section>
      <section style="margin-top:18px">
        <h2>Token Usage</h2>
        <form class="grid" action="/token-usage" method="get">
          <label>Provider<input name="provider" value="codex"></label>
          <label>Project<input name="project" value="MFT_1MW_RL"></label>
          <label>Input<input name="input_tokens" type="number" min="0" value="0"></label>
          <label>Output<input name="output_tokens" type="number" min="0" value="0"></label>
          <label>Total<input name="total_tokens" type="number" min="0" value="0"></label>
          <label>Cycle<input name="reset_cycle" placeholder="2026-W25"></label>
          <label>Note<input name="note"></label>
          <button type="submit">Record</button>
        </form>
        {token_chart(tokens)}
        <h3>Summary</h3>
        <table><thead><tr><th>Provider</th><th>Project</th><th>Cycle</th><th>Total</th></tr></thead><tbody>{token_summary_rows}</tbody></table>
        <h3>Records</h3>
        <table><thead><tr><th>Time</th><th>Provider</th><th>Project</th><th>Cycle</th><th>Input</th><th>Output</th><th>Total</th><th>Note</th></tr></thead><tbody>{token_rows}</tbody></table>
      </section>
      <section style="margin-top:18px">
        <h2>Notes</h2>
        <pre>{html.escape(read_tail(NOTE_PATH))}</pre>
      </section>
      <section style="margin-top:18px">
        <h2>Insights</h2>
        <pre>{html.escape(read_tail(INSIGHT_PATH))}</pre>
      </section>
    </div>
    <section>
      <h2>Best Parameters</h2>
      <table><tbody>{best_rows}</tbody></table>
    </section>
  </main>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/token-usage":
            query = parse_qs(parsed.query)
            append_token_usage(
                provider=query.get("provider", ["codex"])[0],
                project=query.get("project", ["MFT_1MW_RL"])[0],
                input_tokens=int(query.get("input_tokens", ["0"])[0] or 0),
                output_tokens=int(query.get("output_tokens", ["0"])[0] or 0),
                total_tokens=int(query.get("total_tokens", ["0"])[0] or 0) or None,
                reset_cycle=query.get("reset_cycle", [""])[0],
                note=query.get("note", [""])[0],
            )
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
            return
        if parsed.path != "/":
            self.send_error(404)
            return
        body = page().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8010), Handler)
    print("MFT RL dashboard: http://127.0.0.1:8010", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
