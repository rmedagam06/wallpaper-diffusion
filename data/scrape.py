import requests
import time
import json
import urllib3
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = "https://wallhaven.cc/api/v1/search"
QUERIES = [
    "nature", "abstract", "cityscape", "space", "minimal",
    "neon", "architecture", "forest", "ocean", "mountains"
]
OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)
metadata = []

for query in QUERIES:
    print(f"Scraping category: {query}")
    for page in range(1, 201):
        try:
            r = requests.get(BASE, params={
                "q": query,
                "page": page,
                "atleast": "1920x1080",
                "sorting": "relevance",
                "purity": "100",
            }, timeout=10, verify=False).json()
        except Exception as e:
            print(f"  Request failed on page {page}: {e}")
            break

        if not r.get("data"):
            break

        for wall in r["data"]:
            img_url = wall["path"]
            fname = OUT_DIR / (wall["id"] + ".jpg")
            if not fname.exists():
                try:
                    img = requests.get(img_url, timeout=20, verify=False).content
                    fname.write_bytes(img)
                except Exception as e:
                    print(f"  Failed to download {img_url}: {e}")
                    continue
            tags = ", ".join(t["name"] for t in wall.get("tags", []))
            metadata.append({
                "image": str(fname),
                "caption": tags if tags else query
            })

        print(f"  Page {page} done. Total collected: {len(metadata)}")
        time.sleep(1.4)

with open("data/metadata.jsonl", "w") as f:
    for row in metadata:
        f.write(json.dumps(row) + "\n")

print(f"\nDone. {len(metadata)} images saved to data/raw/")
print(f"Metadata saved to data/metadata.jsonl")
