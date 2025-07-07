import pandas as pd
from pathlib import Path
import argparse

def summarize_csv_metrics(csv_path, output_path):
    """
    Read the evaluation CSV, compute average of all numeric columns, and write to a summary TXT file.
    
    Args:
        csv_path (str or Path): Path to the input CSV file.
        output_path (str or Path): Path to save the output summary .txt file.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    # Load CSV file
    df = pd.read_csv(csv_path)

    # Select numeric columns only
    numeric_df = df.select_dtypes(include='number')

    # Compute column-wise mean and round values
    means = numeric_df.mean().round(4)

    # Write summary to TXT file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Average Metrics Across All Images:\n")
        f.write("=" * 40 + "\n")
        for col, val in means.items():
            f.write(f"{col}: {val}\n")

    print(f"✅ Summary saved to {output_path}")


# ------------------ CLI Entry ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize average metrics from evaluation CSV.")
    parser.add_argument("--csv", required=True, help="Path to input CSV file (e.g., result.csv)")
    parser.add_argument("--output", required=True, help="Path to save summary TXT file (e.g., summary.txt)")

    args = parser.parse_args()

    summarize_csv_metrics(
        csv_path=args.csv,
        output_path=args.output
    )
