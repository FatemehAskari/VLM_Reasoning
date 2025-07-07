import json
import csv
import argparse
from pathlib import Path
from tqdm.auto import tqdm

# Compute precision, recall, F1-score, and Jaccard index
def compute_metrics(gt_items, gen_items):
    gt_set = set(gt_items)
    gen_set = set(gen_items)

    tp = len(gt_set & gen_set)
    fp = len(gen_set - gt_set)
    fn = len(gt_set - gen_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 1.0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0

    return precision, recall, f1, jaccard

# Row-wise evaluation using weighted error and metrics
def evaluate_row(gt_list, gen_list):
    gt_items = [f"{i['color']}|{i['shape']}" for i in gt_list]
    gen_items = [f"{i['color']}|{i['shape']}" for i in gen_list]

    remaining_gt = gt_items.copy()
    remaining_gen = gen_items.copy()
    error = 0

    # Step 1: Exact match (color + shape)
    for item in gt_items:
        if item in remaining_gen and item in remaining_gt:
            remaining_gen.remove(item)
            remaining_gt.remove(item)

    # Step 2: Match by shape only
    temp_gt = remaining_gt.copy()
    temp_gen = remaining_gen.copy()
    for l1 in temp_gt:
        shape1 = l1.split('|')[1]
        for l2 in temp_gen:
            shape2 = l2.split('|')[1]
            if shape1 == shape2:
                if l1 in remaining_gt and l2 in remaining_gen:
                    remaining_gt.remove(l1)
                    remaining_gen.remove(l2)
                    error += 1
                    break

    # Step 3: Match by color only
    temp_gt = remaining_gt.copy()
    temp_gen = remaining_gen.copy()
    for l1 in temp_gt:
        color1 = l1.split('|')[0]
        for l2 in temp_gen:
            color2 = l2.split('|')[0]
            if color1 == color2:
                if l1 in remaining_gt and l2 in remaining_gen:
                    remaining_gt.remove(l1)
                    remaining_gen.remove(l2)
                    error += 1
                    break

    # Step 4: Remaining unmatched ground-truth items = 2 points each
    error += 2 * len(remaining_gt)

    # Compute metrics
    precision, recall, f1, jaccard = compute_metrics(gt_items, gen_items)

    return error, precision, recall, f1, jaccard

# Main evaluation loop
def evaluate_accuracy(truth_dir, row_dir, simple_dir, output_csv):
    truth_dir = Path(truth_dir)
    row_dir = Path(row_dir)
    simple_dir = Path(simple_dir)

    results = []

    for truth_file in tqdm(sorted(truth_dir.glob("*.json")), desc="Evaluating"):
        try:
            with open(truth_file, "r", encoding="utf-8") as f:
                truth_data = json.load(f)

            image_key = list(truth_data.keys())[0]
            base_name = truth_file.name

            row_file = row_dir / base_name
            simple_file = simple_dir / base_name

            if not row_file.exists() or not simple_file.exists():
                print(f"⚠️ Skipping {base_name} — missing in row or simple folder.")
                continue

            with open(row_file, "r", encoding="utf-8") as f:
                row_data = json.load(f)
            with open(simple_file, "r", encoding="utf-8") as f:
                simple_data = json.load(f)

            gt = truth_data[image_key]
            row_pred = row_data.get(image_key, {})
            simple_pred = simple_data.get(image_key, {})

            num_rows = 0
            row_error_total = 0
            simple_error_total = 0
            row_metrics = {"precision": 0, "recall": 0, "f1": 0, "jaccard": 0}
            simple_metrics = {"precision": 0, "recall": 0, "f1": 0, "jaccard": 0}

            for row_id in ["1", "2", "3", "4"]:
                gt_row = gt.get(row_id, [])
                row_row = row_pred.get(row_id, [])
                simple_row = simple_pred.get(row_id, [])

                if not gt_row:
                    continue  # Skip empty rows

                # Evaluate row method
                row_error, p, r, f1, jacc = evaluate_row(gt_row, row_row)
                row_error_total += row_error
                row_metrics["precision"] += p
                row_metrics["recall"] += r
                row_metrics["f1"] += f1
                row_metrics["jaccard"] += jacc

                # Evaluate simple method
                simple_error, p, r, f1, jacc = evaluate_row(gt_row, simple_row)
                simple_error_total += simple_error
                simple_metrics["precision"] += p
                simple_metrics["recall"] += r
                simple_metrics["f1"] += f1
                simple_metrics["jaccard"] += jacc

                num_rows += 1

            total_objects = sum(len(v) for v in gt.values())
            row_avg_error = row_error_total / num_rows if num_rows else 0
            simple_avg_error = simple_error_total / num_rows if num_rows else 0

            results.append({
                "image": image_key,
                "total_objects": total_objects,
                "row_avg_error": round(row_avg_error, 4),
                "simple_avg_error": round(simple_avg_error, 4),
                "row_precision": round(row_metrics["precision"] / num_rows, 4),
                "row_recall": round(row_metrics["recall"] / num_rows, 4),
                "row_f1": round(row_metrics["f1"] / num_rows, 4),
                "row_jaccard": round(row_metrics["jaccard"] / num_rows, 4),
                "simple_precision": round(simple_metrics["precision"] / num_rows, 4),
                "simple_recall": round(simple_metrics["recall"] / num_rows, 4),
                "simple_f1": round(simple_metrics["f1"] / num_rows, 4),
                "simple_jaccard": round(simple_metrics["jaccard"] / num_rows, 4)
            })

        except Exception as e:
            print(f"❌ Error processing {truth_file.name}: {e}")

    # Write final results to CSV
    if results:
        with open(output_csv, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"✅ Results saved to {output_csv}")
    else:
        print("⚠️ No valid data to write.")

# ---------------- CLI Entry ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate simple vs row layout predictions.")
    parser.add_argument("--truth", required=True, help="Path to ground truth JSON folder")
    parser.add_argument("--row", required=True, help="Path to predictions using row prompt")
    parser.add_argument("--simple", required=True, help="Path to predictions using simple prompt")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    args = parser.parse_args()

    evaluate_accuracy(
        truth_dir=args.truth,
        row_dir=args.row,
        simple_dir=args.simple,
        output_csv=args.output
    )
