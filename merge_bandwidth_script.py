import os
import json
import csv
from collections import defaultdict

BASE_DIR = "data/pi-to-pi"
OUTPUT_CSV = "data/lookup_table/pi_to_pi_lookup_results.csv"

rows_by_group = defaultdict(list)
group_order = []  # preserve insertion order

for bandwidth_folder in os.listdir(BASE_DIR):
    bandwidth_path = os.path.join(BASE_DIR, bandwidth_folder)
    if not os.path.isdir(bandwidth_path):
        continue

    bandwidth_mbps = int(bandwidth_folder.replace("mbps", ""))

    for model_folder in os.listdir(bandwidth_path):
        model_path = os.path.join(bandwidth_path, model_folder)
        if not os.path.isdir(model_path):
            continue

        group_key = (bandwidth_mbps, model_folder)
        if group_key not in rows_by_group:
            group_order.append(group_key)

        for file in os.listdir(model_path):
            if file.endswith(".json"):
                file_path = os.path.join(model_path, file)

                with open(file_path, "r") as f:
                    data = json.load(f)

                metrics = data.get("average_metrics_per_batch", {})

                raw_model_name = data.get("l_name", model_folder)
                clean_model_name = raw_model_name.split("_")[0]

                row = {
                    "bandwidth_mbps": bandwidth_mbps,
                    "model_name": clean_model_name,
                    "split_index": data.get("split_index"),
                    "static_network_delay_ms": data.get("static_network_delay_ms"),
                    "system_inference_throughput_imgs_per_s": data.get(
                        "system_inference_throughput_imgs_per_s"
                    ),
                    "part1_inference_time_s": metrics.get("part1_inference_time_s"),
                    "part2_inference_time_s": metrics.get("part2_inference_time_s"),
                    "network_time_s": metrics.get("network_time_s"),
                    "end_to_end_latency_s": metrics.get("end_to_end_latency_s"),
                    "intermediate_data_size_bytes": metrics.get(
                        "intermediate_data_size_bytes"
                    ),
                    "network_throughput_mbps": metrics.get(
                        "network_throughput_mbps"
                    ),
                }

                rows_by_group[group_key].append(row)

# 🔹 Assemble final rows (sort only within each group)
final_rows = []

for group_key in group_order:
    group_rows = rows_by_group[group_key]
    group_rows.sort(
        key=lambda x: x["split_index"]
        if x["split_index"] is not None else float("inf")
    )
    final_rows.extend(group_rows)

# Write CSV
if final_rows:
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=final_rows[0].keys())
        writer.writeheader()
        writer.writerows(final_rows)

print(f"CSV written to {OUTPUT_CSV}")
