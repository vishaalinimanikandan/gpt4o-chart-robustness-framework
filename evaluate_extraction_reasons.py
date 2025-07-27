import json
from pathlib import Path
import pandas as pd

# Load required JSON files
base_path = Path("./")  # Adjust this if your files are in a different folder

with open("E:/langchain/Dissertation/data/analysis_cache/chart_generation_summary.json") as f:
    chart_generation_summary = json.load(f)

with open("E:/langchain/Dissertation/data/analysis_cache/extraction_summary.json") as f:
    extraction_summary = json.load(f)

with open("E:/langchain/Dissertation/data/analysis_cache/complete_extraction_results.json") as f:
    complete_results = json.load(f)

# Step 1: Determine which original charts were NOT perturbed
original_total = chart_generation_summary["charts_generated"]  # 200 original charts
original_ids = {f"chart_{str(i).zfill(3)}" for i in range(original_total)}
extracted_original_ids = {k.split("_original")[0] for k in complete_results if "_original" in k}

not_perturbed_ids = sorted(original_ids - extracted_original_ids)
not_perturbed_reasons = []

for chart_id in not_perturbed_ids:
    key = f"{chart_id}_original"
    if key not in complete_results:
        not_perturbed_reasons.append((chart_id, "Missing GPT-4V extraction result"))
    else:
        res = complete_results[key]
        if "chart_type" not in res or "data" not in res or not res["data"]:
            not_perturbed_reasons.append((chart_id, "Invalid JSON: missing 'chart_type' or empty 'data'"))
        else:
            not_perturbed_reasons.append((chart_id, "Unknown: possibly skipped in perturbation loop"))

df_unperturbed = pd.DataFrame(not_perturbed_reasons, columns=["Chart ID", "Exclusion Reason"])
df_unperturbed.to_csv("charts_not_perturbed.csv", index=False)
print(" Saved: charts_not_perturbed.csv")

# Step 2: Determine which perturbations were excluded from evaluation
used_keys = set()
if "evaluation_keys" in extraction_summary:
    used_keys = set(extraction_summary["evaluation_keys"].get("perturbations", []))
else:
    used_perturbations = extraction_summary["extraction_breakdown"]["perturbations"]
    used_keys = set(list(complete_results.keys())[-used_perturbations:])  # fallback approx

excluded_perturbations = []
for key, val in complete_results.items():
    if "_original" in key:
        continue
    if key not in used_keys:
        if not isinstance(val, dict) or "data" not in val or not val["data"] or "chart_type" not in val:
            reason = "Invalid or incomplete JSON (missing 'data' or 'chart_type')"
        else:
            reason = "Valid JSON but excluded from analysis (e.g., unmatched original or filtered)"
        excluded_perturbations.append((key, reason))

df_excluded_perturbs = pd.DataFrame(excluded_perturbations, columns=["Perturbed Key", "Exclusion Reason"])
df_excluded_perturbs.to_csv("excluded_perturbations.csv", index=False)
print(" Saved: excluded_perturbations.csv")
