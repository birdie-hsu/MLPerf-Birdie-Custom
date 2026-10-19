"""Explicitly synthetic offline smoke-test fixtures; never performance results."""
import json
from pathlib import Path

from .transport import Reply
from .workloads import SWE_CORRECT, make_data

ANALYSIS = '''import csv, json
rows = list(csv.DictReader(open("sales.csv", newline="", encoding="utf-8")))
seen, valid, invalid, duplicates = set(), [], 0, 0
for row in rows:
    key = tuple(row.items())
    if key in seen:
        duplicates += 1
        continue
    seen.add(key)
    try:
        r, w = float(row["retail"]), float(row["warehouse"])
        if r < 0 or w < 0: raise ValueError()
    except ValueError:
        invalid += 1
        continue
    valid.append((row, r + w))
suppliers, months = {}, {}
for row, total in valid:
    s, m = row["supplier"], row["month"]
    suppliers[s] = suppliers.get(s, 0) + total
    months[m] = months.get(m, 0) + total
report = dict(valid_rows=len(valid), duplicate_rows=duplicates, invalid_rows=invalid,
              totals_by_supplier=suppliers, monthly_total=months,
              top_supplier=max(suppliers, key=suppliers.get))
json.dump(report, open("report.json", "w"))
print(json.dumps(report))
'''


class MockAdapter:
    def __init__(self, target, case):
        self.target, self.case, self.index = target, case, 0

    def call(self, messages, tools=None, cap=512, timeout=None):
        reply = Reply(synthetic=True, resolved_model="MOCK/" + self.target["model"],
                      finish="completed", duration_s=0.02, first_event_s=0.001)
        if self.case["mode"] == "autonomous" and tools:
            kind = self.case["kind"]
            filename = "bootstrap_mean_ci.py" if kind == "swe" else "sales.csv"
            program = "bootstrap_mean_ci.py" if kind == "swe" else "analysis.py"
            steps = [("read", {"path": filename}),
                     ("write", {"path": program, "content": SWE_CORRECT if kind == "swe" else ANALYSIS}),
                     ("execute", {"command": "python " + program})]
            if self.index < len(steps):
                name, args = steps[self.index]
                reply.calls = [{"id": f"mock_{self.index}", "name": name, "arguments": json.dumps(args)}]
                reply.first_tool_s = 0.005
            else:
                reply.text = "Synthetic agent completed its scripted workflow."
            self.index += 1
        elif "data_csv2json" in self.case["id"]:
            import csv, io
            rows = []
            for row in csv.DictReader(io.StringIO(self.case["messages"][-1]["content"])):
                obj = {k.strip(): v.strip() for k, v in row.items()}
                for key in obj:
                    if key == "id": obj[key] = int(obj[key])
                    elif "TTFT" in key or "TPS" in key: obj[key] = float(obj[key])
                rows.append(obj)
            reply.text = json.dumps(rows)
        elif "ics2xml" in self.case["id"]:
            reply.text = "<events><event>Synthetic syntax fixture</event></events>"
        else:
            reply.text = "Synthetic fixture response. This does not measure model quality or speed."
        if reply.text:
            reply.first_text_s, reply.last_text_s, reply.text_delta_count = 0.005, 0.015, 2
        reply.usage = {}  # No invented token counts.
        reply.history_items = ([{"type": "function_call", "call_id": c["id"], "name": c["name"],
                                 "arguments": c["arguments"]} for c in reply.calls]
                               or [{"role": "assistant", "content": reply.text}])
        return reply


class MockDocker:
    def __init__(self, case):
        self.case = case
        self.image_id = "MOCK/no-container-execution"

    def preflight(self):
        return self.image_id

    def execute(self, root, command, timeout=30, tests=None):
        # Never execute authored code on the host. Script outputs are canned and
        # marked synthetic throughout the run and report.
        if self.case.get("kind") == "data":
            _, expected = make_data(self.case["seed"])
            (Path(root) / "report.json").write_text(json.dumps(expected), encoding="utf-8")
        return {"ok": True, "exit_code": 0, "stdout": "MOCK execution fixture",
                "stderr": "", "duration_s": 0.001, "synthetic": True}
