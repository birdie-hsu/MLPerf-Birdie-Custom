import csv, json
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
