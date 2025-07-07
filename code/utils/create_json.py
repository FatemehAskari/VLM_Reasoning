import pandas as pd
import numpy as np
import ast
import json
import os
import re

def fix_positions_string(pos_str):
    """Fix a malformed string representing position array into proper list of coordinates."""
    pos_str = re.sub(r'array\((.+)\)', r'\1', pos_str)
    pos_str = re.sub(r'(?<=\d)\.(?=\s)', '.0,', pos_str)
    pos_str = re.sub(r'\.(?=\])', '.0', pos_str)
    pos_str = re.sub(r'\]\s*\[', '], [', pos_str)
    pos_str = pos_str.replace('\n', ' ').replace('  ', ' ')
    return ast.literal_eval(pos_str)

def assign_objects_by_area(positions, features, img_height=556, rows=4, box_size=35):
    """
    Assign each object to a row based on maximum vertical overlap area.
    """
    row_height = img_height / rows
    assignments = {i+1: [] for i in range(rows)}  # keys: 1 to 4

    for i, (x, y) in enumerate(positions):
        top = y
        bottom = y + box_size
        area_per_row = {}

        for r in range(rows):
            row_top = r * row_height
            row_bottom = (r + 1) * row_height
            height_overlap = max(0, min(bottom, row_bottom) - max(top, row_top))
            area = height_overlap * box_size
            area_per_row[r + 1] = area

        best_row = max(area_per_row, key=area_per_row.get)
        assignments[best_row].append(features[i])

    return assignments

# === Paths ===
metadata_path = "/home/mmd/vlm-binding-main/data/vlm/binding2/metadata.csv"   # <-- path to your metadata.csv
output_dir = "/home/mmd/vlm-binding-main/data/vlm/binding2/json"              # <-- directory to save JSON files
os.makedirs(output_dir, exist_ok=True)

# === Load metadata ===
df = pd.read_csv(metadata_path)

# === Process each row
for idx, row in df.iterrows():
    try:
        positions = np.array(fix_positions_string(row["positions"]))
        features = ast.literal_eval(row["features"])
        assignment = assign_objects_by_area(positions, features)

        image_filename = os.path.basename(row["path"])
        output_json = {image_filename: assignment}
        json_path = os.path.join(output_dir, image_filename + ".json")

        with open(json_path, "w") as f:
            json.dump(output_json, f, indent=2)

        print(f"✅ Saved: {json_path}")
    except Exception as e:
        print(f"❌ Error processing row {idx}: {e}")


