from PIL import Image
from pathlib import Path
import imagehash
import json
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed/512x512")
OUT_DIR.mkdir(parents=True, exist_ok=True)

seen_hashes = set()
kept = []

with open("data/metadata.jsonl") as f:
    rows = [json.loads(line) for line in f if line.strip()]

print(f"Processing {len(rows)} images...")

for row in tqdm(rows, desc="Resizing & deduplicating"):
    src = Path(row["image"])
    if not src.exists():
        continue
    try:
        img = Image.open(src).convert("RGB")
        phash = str(imagehash.phash(img))
        if phash in seen_hashes:
            continue
        seen_hashes.add(phash)

        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top  = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((512, 512), Image.LANCZOS)

        dst = OUT_DIR / src.name
        img.save(dst, quality=95)
        kept.append({"image": str(dst), "caption": row["caption"]})
    except Exception as e:
        print(f"Skipped {src.name}: {e}")

with open("data/metadata_processed.jsonl", "w") as f:
    for row in kept:
        f.write(json.dumps(row) + "\n")

print(f"\nDone. Kept {len(kept)} unique images after deduplication.")
print(f"Saved to data/processed/512x512/")
print(f"Metadata saved to data/metadata_processed.jsonl")
