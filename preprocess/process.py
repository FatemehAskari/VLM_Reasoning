import shutil
import os
import argparse
from draw import (
    GridDrawer,
    ImageGridProcessor,
    FullGridNumberedDrawer,
    load_font,
    RowOnlyDrawer,
    ColumnOnlyDrawer,
    ColumnOnlyNumberedDrawer,
    GradientDotsDrawer,
    RowOnlyNumberedDrawer,
    SequentialNumberedGridDrawer
)

def process_dataset_directory(
    input_root,
    output_root,
    drawer: GridDrawer,
    rows=4,
    cols=4,
    image_extensions=('.png', '.jpg', '.jpeg')
):
    """
    Recursively process all image files in a nested folder structure.
    Applies a given GridDrawer strategy and saves output with same directory layout.
    Also copies `answer.json` files if found in the image folders.
    """
    processor = ImageGridProcessor(drawer)

    for dirpath, _, filenames in os.walk(input_root):
        # Compute the relative path from the input root
        rel_path = os.path.relpath(dirpath, input_root)
        output_dir = os.path.join(output_root, rel_path)
        os.makedirs(output_dir, exist_ok=True)

        for filename in filenames:
            # Process image files
            if filename.lower().endswith(image_extensions):
                input_path = os.path.join(dirpath, filename)
                output_path = os.path.join(output_dir, filename)
                try:
                    processor.process_image(input_path, output_path, rows, cols)
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

            # Copy answer.json if it exists
            if filename == "answer.json":
                src = os.path.join(dirpath, filename)
                dst = os.path.join(output_dir, filename)
                shutil.copyfile(src, dst)

def select_drawer(mode: str, font) -> GridDrawer:
    """
    Return the appropriate drawer object based on the selected mode.
    """
    mode = mode.lower()
    if mode == "row":
        return RowOnlyDrawer()
    elif mode == "rownum":
        return RowOnlyNumberedDrawer(font)
    elif mode == "col":
        return ColumnOnlyDrawer()
    elif mode == "colnum":
        return ColumnOnlyNumberedDrawer(font)
    elif mode == "gridnum":
        return FullGridNumberedDrawer(font)
    elif mode == "sequential":
        return SequentialNumberedGridDrawer(font)
    elif mode == "dots":
        return GradientDotsDrawer()
    else:
        raise ValueError(f"Unknown mode: {mode}")

def main():
    parser = argparse.ArgumentParser(description="Process a dataset directory with a specified grid drawer.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input image directory")
    parser.add_argument('--output', type=str, required=True, help="Path to save processed images")
    parser.add_argument('--mode', type=str, required=True, choices=['row', 'rownum', 'col', 'colnum', 'gridnum', 'sequential', 'dots'],
                        help="Drawer mode: row, rownum, col, colnum, gridnum, sequential, dots")
    parser.add_argument('--rows', type=int, default=4, help="Number of rows in the grid")
    parser.add_argument('--cols', type=int, default=4, help="Number of columns in the grid")

    args = parser.parse_args()
    font = load_font()

    drawer = select_drawer(args.mode, font)

    process_dataset_directory(
        input_root=args.input,
        output_root=args.output,
        drawer=drawer,
        rows=args.rows,
        cols=args.cols,
        image_extensions=('.png', '.jpg', '.jpeg')
    )

if __name__ == "__main__":
    main()
