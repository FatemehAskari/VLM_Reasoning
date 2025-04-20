import os
from PIL import Image, ImageDraw, ImageFont

def add_horizontal_rulers_with_coordinates(
    input_path,
    output_path,
    step=70,
    font_size=0,
    tick_length=10,
    line_width=2,
    ruler_interval=80  
):
    img = Image.open(input_path)
    width, height = img.size
    ruler_img = img.copy()
    draw = ImageDraw.Draw(ruler_img)

    font_paths = [
        "font/Arial.ttf",  # Use a proper font file path
    ]

    # Try to load a custom font, fallback to default if not found
    font = None
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        except:
            continue
    else:
        font = ImageFont.load_default()
        print("Using default font - size may not be accurate")

    # Draw horizontal rulers and coordinate ticks
    for y_pos in range(0, height, ruler_interval):
        tick_dir = 1 if y_pos < height / 2 else -1

        for x in range(0, width, step):
            # Draw a tick mark
            draw.line(
                [(x, y_pos), (x, y_pos + tick_dir * tick_length)],
                fill="black",
                width=line_width
            )

            # Create coordinate label
            text = f"({x}, {y_pos})"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Position the text based on tick direction
            if tick_dir == 1:
                text_y = y_pos + tick_length + 5
                anchor = "mt"  # middle-top
            else:
                text_y = y_pos - tick_length - 5
                anchor = "mb"  # middle-bottom

            draw.text(
                (x, text_y),
                text,
                fill="black",
                font=font,
                anchor=anchor
            )

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ruler_img.save(output_path)
    print(f"Saved to {output_path}")


# Root directories
input_root = "images_counting_binding_test_samples"               # Input folder containing subfolders and images
output_root = "output_images"  # Output folder to save processed images

# Acceptable image file extensions
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

# Walk through all subdirectories and process each image
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(valid_extensions):
            input_path = os.path.join(root, file)
            # Create a relative path from input_root
            relative_path = os.path.relpath(input_path, input_root)
            # Generate the output path with the same subfolder structure
            output_path = os.path.join(output_root, relative_path)
            # Call the function to process and save the image
            add_horizontal_rulers_with_coordinates(input_path, output_path)
