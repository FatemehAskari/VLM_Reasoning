import os
import re
import json
import base64
import argparse
from pathlib import Path
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

# Set up LLM
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)

# ---------------- UTILS ----------------
def encode_image(image_path):
    """Convert image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def build_prompt(mode: str, dimension: str) -> str:
    """
    Construct prompt for counting shapes depending on mode and dimension.
    """
    shape = "spheres" if dimension == "3d" else "circles"
    scan_line = "Scan the image sequentially based on horizontal lines exists in the image." if mode == "row" else ""
    
    return f"""
**Task:**
How many {shape.capitalize()} are there in this image?
{scan_line}
Finally, provide the Total Number of {shape.capitalize()} at the end with this format:
Total Number of {shape.capitalize()}: x
"""

def extract_count(response, dimension):
    """Extract total shape count from model response."""
    shape = "spheres" if dimension == "3d" else "circles"
    pattern = rf"Total Number of {shape.capitalize()}:\s*(\d+)"
    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
    if match:
        return int(match.group(1))
    return None

def invoke_llm(image_path, ext, mode, dimension):
    """Send prompt and image to LLM."""
    base64_img = encode_image(image_path)
    prompt_text = build_prompt(mode, dimension)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{ext};base64,{base64_img}",
                        "detail": "auto"
                    }
                }
            ]
        }
    ]
    return llm.invoke(messages).content

def process_folder(folder_path, output_path, mode, dimension):
    """Iterate over images and save counts to JSON."""
    folder_path = Path(folder_path)
    output = {}

    for image_path in sorted(folder_path.glob("*.*")):
        if image_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue
        ext = image_path.suffix[1:]
        print(f"🔍 Processing: {image_path.name}")

        try:
            response = invoke_llm(str(image_path), ext, mode, dimension)
            count = extract_count(response, dimension)
            output[str(image_path)] = count
            print(f"✅ Count: {count}")
        except Exception as e:
            print(f"❌ Failed to process {image_path.name}: {e}")
            output[str(image_path)] = None

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"\n📁 Results saved to: {output_path}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count shapes in 2D/3D images using LLM.")
    parser.add_argument("--mode", type=str, choices=["simple", "row"], required=True, help="Prompt mode: 'simple' or 'row'")
    parser.add_argument("--dimension", type=str, choices=["2d", "3d"], required=True, help="Data type: '2d' or '3d'")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing images")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file to save counts")

    args = parser.parse_args()

    process_folder(args.input, args.output, args.mode, args.dimension)
