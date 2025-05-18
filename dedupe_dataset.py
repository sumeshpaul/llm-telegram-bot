import json

input_path = "train_dataset.json"
output_path = "train_dataset_v2.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

seen = set()
unique_data = []
for item in data:
    key = (item.get("instruction", ""), item.get("input", ""))
    if key not in seen:
        seen.add(key)
        unique_data.append(item)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(unique_data, f, indent=2, ensure_ascii=False)

print(f"✅ Saved deduplicated dataset to {output_path} ({len(unique_data)} entries)")
