from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse

from rl_mft.records import INSIGHT_PATH, NOTE_PATH, append_token_usage, load_state, read_token_usage


app = FastAPI(title="MFT RL Dashboard")


def read_tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(content[-lines:])


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


@app.get("/api/state")
def api_state() -> dict:
    return load_state()


@app.get("/api/token-usage")
def api_token_usage() -> list[dict]:
    return read_token_usage()


@app.get("/token-usage")
def create_token_usage(
    provider: str = "codex",
    project: str = "MFT_1MW_RL",
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    reset_cycle: str = "",
    note: str = "",
) -> RedirectResponse:
    append_token_usage(
        provider=provider,
        project=project,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens or None,
        reset_cycle=reset_cycle,
        note=note,
    )
    return RedirectResponse("/", status_code=303)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    state = load_state()
    tokens = read_token_usage()
    loops = list(reversed(state.get("loops", [])[-20:]))
    rows = "\n".join(
        f"<tr><td>{loop.get('loop')}</td><td>{loop.get('status')}</td><td>{loop.get('backend')}</td>"
        f"<td>{loop.get('completed')}</td><td>{loop.get('failed')}</td><td>{loop.get('best_reward')}</td>"
        f"<td>{loop.get('best_candidate_id') or ''}</td></tr>"
        for loop in loops
    )
    best = state.get("best_parameters") or {}
    best_outputs = state.get("best_outputs") or {}
    live_outputs = state.get("live_best_outputs") or {}
    best_rows = "\n".join(f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in best.items())
    best_output_rows = "\n".join(f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in best_outputs.items())
    live_output_rows = "\n".join(f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in live_outputs.items())
    token_rows = "\n".join(
        f"<tr><td>{item.get('recorded_at')}</td><td>{item.get('provider')}</td><td>{item.get('project')}</td>"
        f"<td>{item.get('reset_cycle')}</td><td>{item.get('input_tokens')}</td><td>{item.get('output_tokens')}</td>"
        f"<td>{item.get('total_tokens')}</td><td>{item.get('note')}</td></tr>"
        for item in reversed(tokens[-50:])
    )
    token_summary_rows = "\n".join(
        f"<tr><td>{item['provider']}</td><td>{item['project']}</td><td>{item['reset_cycle']}</td><td>{item['total_tokens']}</td></tr>"
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
      <div><strong>{state.get('best_candidate_id') or '-'}</strong>Best candidate</div>
      <div><strong>{state.get('live_best_reward', '-')}</strong>Live best</div>
      <div><strong>{state.get('live_best_candidate_id') or '-'}</strong>Live candidate</div>
      <div><strong>{state.get('failure_rate', 0)}</strong>Failure rate</div>
    </div>
  </header>
  <main>
    <div>
      <section>
        <h2>Recent Loops</h2>
        <table>
          <thead><tr><th>Loop</th><th>Status</th><th>Backend</th><th>Done</th><th>Failed</th><th>Best Reward</th><th>Candidate</th></tr></thead>
          <tbody>{rows}</tbody>
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
          <label>Cycle<input name="reset_cycle" placeholder="2026-W24"></label>
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
        <pre>{read_tail(NOTE_PATH)}</pre>
      </section>
      <section style="margin-top:18px">
        <h2>Insights</h2>
        <pre>{read_tail(INSIGHT_PATH)}</pre>
      </section>
    </div>
    <section>
      <h2>Best Parameters</h2>
      <table><tbody>{best_rows}</tbody></table>
      <h2>Best Outputs</h2>
      <table><tbody>{best_output_rows}</tbody></table>
      <h2>Live Best Outputs</h2>
      <table><tbody>{live_output_rows}</tbody></table>
    </section>
  </main>
</body>
</html>"""
