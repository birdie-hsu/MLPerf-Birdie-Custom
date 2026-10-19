"""Separate lanes, explicit missing values, attempt denominators, and raw exports."""
import csv
import html
import json
import math
from pathlib import Path
import random
import statistics

from .runner import is_timing_sample
from .transport import server_metrics
from . import __version__


def percentile(values, p):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def case_bootstrap_ci(rows):
    groups = {}
    for row in rows:
        if is_timing_sample(row) and row.get("e2e_s") is not None:
            groups.setdefault(row["case_id"], []).append(row["e2e_s"])
    centers = [statistics.median(group) for group in groups.values()]
    if len(centers) < 2:
        return None
    rng = random.Random(42)
    means = [statistics.mean(rng.choices(centers, k=len(centers))) for _ in range(1000)]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def summarize(rows):
    groups = {}
    for row in rows:
        if row["phase"] == "measured":
            key = (row["target"], row["mode"], row["category"], row.get("replay_tools_enabled", False))
            groups.setdefault(key, []).append(row)
    summaries = []
    for (target, mode, category, replay_tools), group in sorted(groups.items()):
        good = [r for r in group if r["status"] == "ok"]
        perf = [r for r in group if is_timing_sample(r)]
        metrics = [m for r in perf for m in r.get("turn_metrics", [])]
        scored = [r for r in group if r.get("quality", {}).get("pass") is not None]
        quality_passes = sum(r["quality"]["pass"] is True and r["status"] == "ok" for r in scored)
        successes = [r for r in good if r.get("quality", {}).get("pass") is True]
        summary = {"target": target, "mode": mode, "category": category,
                   "replay_tools": replay_tools, "attempts": len(group), "completed": len(good),
                   "valid_measurements": len(perf),
                   "truncated": sum(r["status"] == "truncated" for r in group),
                   "failures": len(group) - len(perf), "scored_attempts": len(scored),
                   "local_thinking": group[0].get("local_thinking"),
                   "quality_passes": quality_passes,
                   "quality_pass_rate": quality_passes / len(scored) if scored else None,
                   "e2e_p50_s": percentile([r.get("e2e_s") for r in perf], 0.5),
                   "e2e_p95_s": percentile([r.get("e2e_s") for r in perf], 0.95),
                   "success_e2e_p50_s": percentile([r["e2e_s"] for r in successes], 0.5),
                   "ttft_visible_p50_s": percentile([m.get("ttft_visible_s") for m in metrics], 0.5),
                   "ttft_visible_p95_s": percentile([m.get("ttft_visible_s") for m in metrics], 0.95),
                   "native_tps_estimate_p50": percentile([m.get("native_tps_estimate") for m in metrics], 0.5),
                   "prefill_p50_s": percentile([m.get("prefill_s") for m in metrics], 0.5),
                   "prefill_tps_p50": percentile([m.get("prefill_tps") for m in metrics], 0.5),
                   "generation_tps_p50": percentile([m.get("generation_tps", m.get("server_decode_tps")) for m in metrics], 0.5),
                   "server_decode_tps_p50": percentile([m.get("generation_tps", m.get("server_decode_tps")) for m in metrics], 0.5),
                   "chars_per_s_p50": percentile([m.get("visible_chars_per_s") for m in metrics], 0.5),
                   "llm_request_p50_s": percentile([r.get("llm_request_s") for r in perf], 0.5),
                   "tool_p50_s": percentile([r.get("tool_s") for r in perf], 0.5),
                   "case_bootstrap_mean_ci95_s": case_bootstrap_ci(group),
                   "distinct_cases": len({r["case_id"] for r in group}),
                   "measured_estimated_usd": (sum(r["estimated_usd"] for r in group)
                       if all(r.get("estimated_usd") is not None for r in group) else None),
                   "synthetic": any(r.get("synthetic") for r in group)}
        summaries.append(summary)
    return summaries


def csv_export(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            converted = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()}
            # Prevent spreadsheet formula execution in model/file names.
            for key, value in converted.items():
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
                    converted[key] = "'" + value
            writer.writerow(converted)


PAGE = r'''<!doctype html>
<html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ClientBench 模型比較報告</title>
<style>
:root{color-scheme:dark;font:15px/1.65 system-ui,sans-serif;background:#0c1422;color:#e7edf6}
body{margin:0;padding:36px;max-width:1600px;margin:auto}h1{font-size:32px;margin:0}.eyebrow{letter-spacing:2px;color:#77d4cb}
p{color:#afbed2;max-width:1100px}.banner{background:#453512;border:1px solid #d59b32;border-radius:10px;padding:15px;margin:18px 0;color:#ffe5a0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:24px 0}.card{background:#142239;border:1px solid #243b59;padding:20px;border-radius:12px}.card b{display:block;font-size:28px}.card span{color:#a9bdd6}
select,input,button{font:inherit;background:#15253b;border:1px solid #35516c;color:#fff;border-radius:7px;padding:8px;margin:0 10px 12px 0}button.active{background:#26726d}.scroll{overflow:auto;border:1px solid #2a405b;border-radius:10px}table{border-collapse:collapse;width:100%;white-space:nowrap}th,td{padding:13px 14px;border-bottom:1px solid #243850;text-align:left}th{background:#142238;color:#a9d5ec}tbody tr:hover{background:#172840}.ok{color:#79d9b3}.fail{color:#ff9b93}.muted{color:#92a5bf}a{color:#81ccc8}details{padding:16px;background:#142238;margin-top:18px}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:440px;overflow:auto}.bar{display:inline-block;background:#58b4ad;height:8px;border-radius:4px;margin-right:8px;min-width:2px}.legend{font-size:13px}h2{font-size:21px;margin-top:32px}.controls{margin-top:25px}footer{margin:30px 0;color:#96aac4}
</style>
<div class="eyebrow">CLIENTBENCH · MLPerf Client inspired</div><h1>LLM 與 Agent 模型比較</h1>
<p class="legend" id="base"></p>
<p>雲端 API 與本機 GGUF 共用工作負載。各測試路徑分開呈現；本報告不是官方 MLPerf 成績。雲端時間包含網路與服務排隊，本機時間包含 HTTP 與推論。</p>
<div id="banner"></div><div class="cards" id="cards"></div>
<div class="controls"><button data-mode="llm" class="active">LLM 原始 prompts</button><button data-mode="replay">Agent 固定重播</button><button data-mode="autonomous">自主 Agent</button><select id="target"><option value="">所有模型</option></select><input id="search" placeholder="篩選 scenario 或模型"></div>
<p class="legend">TTFT 是首個可見文字片段的等待時間。Prefill 秒數與 Prefill TPS 為伺服器回報的輸入處理時間與速率，只計實際處理的 tokens，排除快取重用。Generation TPS 為伺服器回報的生成速率，包含 thinking tokens；Native TPS 是可見文字的估計值。上述彙整欄位顯示有效測量中各模型請求的 p50（中位數），不是多回合任務的合計時間。tokenizer 不同；串流片段不等於單一 token。LLM 達到 token 上限仍保留有效計時，但不算完整完成或品質通過。— 代表無資料或不適用。品質檢查與語意品質不同；重播不提供任務成功率。</p>
<div class="scroll"><table><thead><tr><th>模型</th><th>Scenario</th><th>有效計時／嘗試</th><th>完整完成</th><th>達到上限</th><th>檢查通過</th><th>TTFT p50 秒</th><th>Prefill p50 秒</th><th>Prefill TPS p50</th><th>Generation TPS p50</th><th>Native TPS 估計</th><th>E2E p50 秒</th><th>E2E p95 秒</th><th>LLM 秒</th><th>Tool 秒</th></tr></thead><tbody id="summary"></tbody></table></div>
<h2>逐次執行紀錄</h2><p class="legend">Warmup 不納入下表與統計。失敗仍保留在分母；完整成功延遲與有效計時延遲分開保存在 summary.json。少量樣本的 p95 僅供探索。</p>
<div class="scroll"><table><thead><tr><th>模型</th><th>案例</th><th>次數</th><th>狀態／診斷</th><th>檢查</th><th>回合</th><th>工具次數</th><th>E2E 秒</th><th>Trace</th></tr></thead><tbody id="attempts"></tbody></table></div>
<details><summary>版本、環境與完整設定</summary><pre id="metadata"></pre></details>
<footer>輸出：<a href="results.csv">逐次 CSV</a> · <a href="summary.csv">彙整 CSV</a> · <a href="results.jsonl">原始紀錄</a> · <a href="run.json">重現設定</a></footer>
<script>
const data=__DATA__;const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const n=x=>x==null?'—':Number(x).toFixed(3);let mode='llm';
const measured=data.rows.filter(r=>r.phase==='measured');
const upstream=data.meta.upstream;document.getElementById('base').textContent=upstream?'MLPerf Client base: '+(upstream.version||(upstream.tag==='v2.0'?'2.0.0':upstream.tag||'未記錄'))+' · commit: '+(upstream.commit||'未記錄'):'MLPerf Client base: 未記錄';
document.getElementById('banner').innerHTML=data.meta.synthetic?'<div class="banner"><strong>MOCK 離線驗證</strong>：計時、回覆與工具執行皆為合成示例，不能用來比較真實模型效能或能力。</div>':(data.meta.stopped_reason?'<div class="banner">部分執行：'+esc(data.meta.stopped_reason)+'</div>':'');
document.getElementById('cards').innerHTML=[['模型',new Set(measured.map(r=>r.target)).size],['測量次數',measured.length],['無有效計時',measured.filter(r=>!r.measurement_valid&&r.status!=='ok').length],['達到 token 上限',measured.filter(r=>r.status==='truncated').length],['工作負載',new Set(measured.map(r=>r.case_id)).size]].map(([a,b])=>'<div class="card"><b>'+b+'</b><span>'+a+'</span></div>').join('');
const target=document.getElementById('target');[...new Set(measured.map(r=>r.target))].sort().forEach(name=>{let option=document.createElement('option');option.value=option.textContent=name;target.appendChild(option)});
function matches(r){const q=document.getElementById('search').value.toLowerCase();return r.mode===mode&&(!target.value||r.target===target.value)&&JSON.stringify([r.target,r.category,r.case_id]).toLowerCase().includes(q)}
function render(){const summaries=data.summary.filter(matches);const max=Math.max(...summaries.map(r=>r.e2e_p50_s||0),1);
document.getElementById('summary').innerHTML=summaries.map(r=>'<tr><td>'+esc(r.target)+(r.local_thinking?'<div class="muted">thinking: '+esc(r.local_thinking)+'</div>':'')+'</td><td>'+esc(r.category)+(r.mode==='replay'?'<div class="muted">'+(r.replay_tools?'含固定工具執行':'僅推論重播')+'</div>':'')+'</td><td>'+r.valid_measurements+'/'+r.attempts+'</td><td>'+r.completed+'</td><td>'+r.truncated+'</td><td>'+(r.scored_attempts?r.quality_passes+'/'+r.scored_attempts:'未評分')+'</td><td>'+n(r.ttft_visible_p50_s)+'</td><td>'+n(r.prefill_p50_s)+'</td><td>'+n(r.prefill_tps_p50)+'</td><td>'+n(r.generation_tps_p50)+'</td><td>'+n(r.native_tps_estimate_p50)+'</td><td><span class="bar" style="width:'+(80*(r.e2e_p50_s||0)/max)+'px"></span>'+n(r.e2e_p50_s)+'</td><td>'+n(r.e2e_p95_s)+'</td><td>'+n(r.llm_request_p50_s)+'</td><td>'+n(r.tool_p50_s)+'</td></tr>').join('')||'<tr><td colspan="15">沒有此測試路徑的資料。</td></tr>';
document.getElementById('attempts').innerHTML=measured.filter(matches).map(r=>'<tr><td>'+esc(r.target)+'</td><td>'+esc(r.case_id)+'</td><td>'+r.repeat+'</td><td class="'+(r.status==='ok'?'ok':r.measurement_valid?'muted':'fail')+'">'+esc(r.status)+(r.diagnostic?'<div class="muted">'+esc(r.diagnostic)+'</div>':'')+'</td><td>'+(r.quality?.pass==null?'—':r.quality.pass&&r.status==='ok'?'通過':'未通過')+'</td><td>'+esc(r.turns??'—')+'</td><td>'+esc(r.tool_calls??'—')+'</td><td>'+n(r.e2e_s)+'</td><td>'+(r.trace?'<a href="'+esc(r.trace)+'">開啟</a>':'—')+'</td></tr>').join('');}
document.querySelectorAll('button[data-mode]').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;document.querySelectorAll('button[data-mode]').forEach(x=>x.classList.toggle('active',x===b));render()});target.onchange=render;document.getElementById('search').oninput=render;document.getElementById('metadata').textContent=JSON.stringify(data.meta,null,2);render();
</script></html>'''


def report_rows(root, rows):
    """Recover timings from older traces without changing original run records."""
    enriched = []
    root = root.resolve()
    for row in rows:
        metrics = [dict(m) for m in row.get("turn_metrics", [])]
        if any("prefill_s" not in m or "prefill_tps" not in m for m in metrics) and row.get("trace"):
            try:
                trace_path = (root / row["trace"]).resolve()
                trace_path.relative_to(root)
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                for metric, turn in zip(metrics, trace.get("turns", [])):
                    for key, value in server_metrics(turn.get("reply", {}).get("native_timings") or {}).items():
                        metric.setdefault(key, value)
            except (OSError, ValueError):
                # Missing/unreadable old traces must not prevent report output.
                pass
        for metric in metrics:
            metric.setdefault("prefill_s", None)
            metric.setdefault("prefill_tps", None)
            metric.setdefault("generation_tps", metric.get("server_decode_tps"))
        enriched.append({**row, "turn_metrics": metrics})
    return enriched


def write_report(run_dir, rows=None):
    root = Path(run_dir)
    if rows is None:
        rows = [json.loads(line) for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = report_rows(root, rows)
    metadata = json.loads((root / "run.json").read_text(encoding="utf-8"))
    metadata["report_version"] = __version__
    summary = summarize(rows)
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_export(root / "results.csv", rows)
    csv_export(root / "summary.csv", summary)
    payload = json.dumps({"meta": metadata, "rows": rows, "summary": summary}, ensure_ascii=False).replace("<", "\\u003c")
    (root / "report.html").write_text(PAGE.replace("__DATA__", payload), encoding="utf-8")
    return root / "report.html"
