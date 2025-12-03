import csv
from collections import defaultdict

data = []
with open("results.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

counts = defaultdict(lambda: {"NEUTRAL": 0, "DETECTED": 0, "IDENTIFIED": 0})

for row in data:
    filepath, status = row

    filename = filepath.split("/")[-1]  # Get filename
    parts = filename.replace(".mp4", "").split("_")

    if "needle" in filename:
        if "scrambled" in filename:
            symbol_type = "needle_scrambled"
        else:
            symbol_type = "needle"
    elif "spider" in filename:
        if "scrambled" in filename:
            symbol_type = "spider_scrambled"
        elif "rectilinear" in filename:
            symbol_type = "spider_rectilinear"
        else:
            symbol_type = "spider"

    counts[symbol_type][status] += 1

print("Symbol Type Analysis")
print("=" * 60)
for symbol_type in sorted(counts.keys()):
    print(f"\n{symbol_type}:")
    print(f"  NEUTRAL:    {counts[symbol_type]['NEUTRAL']}")
    print(f"  DETECTED:   {counts[symbol_type]['DETECTED']}")
    print(f"  IDENTIFIED: {counts[symbol_type]['IDENTIFIED']}")
    total = sum(counts[symbol_type].values())
    print(f"  Total:      {total}")

import matplotlib.pyplot as plt

labels = sorted(counts.keys())
neutral = [counts[l]["NEUTRAL"] for l in labels]
detected = [counts[l]["DETECTED"] for l in labels]
identified = [counts[l]["IDENTIFIED"] for l in labels]

x = range(len(labels))
width = 0.6

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x, neutral, width, label="NEUTRAL", color="#8da0cb")
ax.bar(x, detected, width, bottom=neutral, label="DETECTED", color="#fc8d62")
bottom = [n + d for n, d in zip(neutral, detected)]
ax.bar(x, identified, width, bottom=bottom, label="IDENTIFIED", color="#66c2a5")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("Count")
ax.set_title("Fear Response per Symbol")
ax.legend()
plt.tight_layout()
plt.savefig("summarised_results.png", dpi=150)
plt.show()
