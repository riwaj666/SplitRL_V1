import os
import json
import csv
from collections import defaultdict

# ================== CONFIG ==================
BASE_DIR = "data/pi-to-pi"
OUTPUT_CSV = "data/lookup_table/pi_to_pi_lookup_results.csv"
# ============================================


def first_available(d, *keys):
    """Return the first non-None value for given keys."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def mbps_from_MBps(value):
    """Convert MB/s → Mb/s if value exists."""
    return value * 8 if value is not None else None


rows_by_group = defaultdict(list)
group_order = []  # preserve insertion order


for bandwidth_folder in sorted(os.listdir(BASE_DIR)):
    bandwidth_path = os.path.join(BASE_DIR, bandwidth_folder)
    if not os.path.isdir(bandwidth_path):
        continue

    if not bandwidth_folder.endswith("mbps"):
        continue

    bandwidth_mbps = int(bandwidth_folder.replace("mbps", ""))

    for model_folder in sorted(os.listdir(bandwidth_path)):
        model_path = os.path.join(bandwidth_path, model_folder)
        if not os.path.isdir(model_path):
            continue

        group_key = (bandwidth_mbps, model_folder)
        if group_key not in rows_by_group:
            group_order.append(group_key)

        for file in sorted(os.listdir(model_path)):
            if not file.endswith(".json"):
                continue

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

                # Times
                "part1_inference_time_s": first_available(
                    metrics, "part1_inference_time_s"
                ),
                "part2_inference_time_s": first_available(
                    metrics, "part2_inference_time_s"
                ),
                "network_time_s": first_available(
                    metrics, "network_time_s"
                ),
                "end_to_end_latency_s": first_available(
                    metrics, "end_to_end_latency_s"
                ),

                # Intermediate data size (old + new schemas)
                "intermediate_data_size_bytes": first_available(
                    metrics,
                    "intermediate_data_size_bytes",  # old
                    "intermediate_fp32_bytes",       # alt
                ),

                # Throughput (old + new schemas)
                "network_throughput_mbps": mbps_from_MBps(
                    first_available(
                        metrics,
                        "network_throughput_mbps",    # old
                        "network_throughput_MBps",    # new
                    )
                ),
            }

            rows_by_group[group_key].append(row)


# ================== SORT WITHIN GROUP ==================
final_rows = []

for group_key in group_order:
    group_rows = rows_by_group[group_key]
    group_rows.sort(
        key=lambda x: x["split_index"]
        if x["split_index"] is not None else float("inf")
    )
    final_rows.extend(group_rows)


# ================== WRITE CSV ==================
if final_rows:
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=final_rows[0].keys())
        writer.writeheader()
        writer.writerows(final_rows)

print(f"✅ CSV written to {OUTPUT_CSV}")
