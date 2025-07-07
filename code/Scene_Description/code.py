import os
import re
import json
import base64
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from langchain_openai import ChatOpenAI

# ---------------- LLM SETUP ----------------
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)

# ---------------- UTILITIES ----------------
def encode_image(image_path):
    """Encode image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_json(response_text):
    """Extract list of dicts from unstructured LLM response."""
    try:
        pattern = r"\[\s*{[\s\S]+?}\s*]"
        match = re.search(pattern, response_text)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print(f"❌ JSON extract error: {e}")
    return None

def build_prompt_text(mode: str, dimension: str) -> str:
    """
    Build the shared prompt text, with variations depending on mode (simple/row)
    and dimension (2d/3d).
    """
    if dimension == "3d":
        shapes = "cube, sphere, cylinder, cone, diamond, hexagon, prism, pyramid, torus, bowl"
        colors = "black, gray, red, blue, green, brown, purple, cyan, yellow, orange"
    else:  # 2D
        shapes = "airplane, triangle, cloud, cross, umbrella, scissors, heart, star, circle, square, infinity, up-arrow, pentagon, left-arrow, flag"
        colors = "red, magenta, salmon, green, lime, olive, blue, teal, yellow, purple, brown, gray, black, cyan, orange"

    base_prompt = f"""
** Task: **
You are presented with an image containing multiple colored objects, each defined by a shape and a color.
Your task is to identify all objects in the image and return a list of dictionaries.
Each dictionary must contain exactly two keys: 'shape' and 'color'.
Only use shapes from this list: {shapes}.
Only use colors from this list: {colors}.
Always respond **only** with a JSON list of the detected objects.
"""
    if mode == "row":
        base_prompt += "Scan sequentially based on horizontal lines and numbers in the image."

    return base_prompt

def invoke_llm(image_path, mode, dimension):
    """Send image and task prompt to LLM and return raw response."""
    image_base64 = encode_image(image_path)
    ext = Path(image_path).suffix[1:]

    prompt_text = build_prompt_text(mode, dimension)

    messages = [
        {"role": "system", "content": "You are a helpful vision assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{ext};base64,{image_base64}",
                        "detail": "auto"
                    }
                }
            ]
        }
    ]
    return llm.invoke(messages).content

# ------------- MAIN PROCESSING -------------
def process_dataset(input_dir, output_dir, mode="simple", dimension="2d"):
    """
    Process all images inside a 2-level directory structure:
    object_count/triplet_number/*.png
    Saves results as individual JSONs per image in matching output folders.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for object_folder in sorted(input_dir.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else float('inf')):
        if not object_folder.is_dir():
            continue

        print(f"📁 Object Count Folder: {object_folder.name}")
        for triplet_folder in tqdm(sorted(object_folder.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else float('inf')), desc=f"Triplet Folders in {object_folder.name}"):
            if not triplet_folder.is_dir():
                continue

            triplet_output_dir = output_dir / object_folder.name / triplet_folder.name
            triplet_output_dir.mkdir(parents=True, exist_ok=True)

            for image_path in sorted(triplet_folder.glob("*.png"), key=lambda x: x.name):
                image_name = image_path.stem
                output_path = triplet_output_dir / f"{image_name}.json"

                if output_path.exists():
                    continue

                success = False
                attempts = 0
                while not success and attempts < 3:
                    attempts += 1
                    try:
                        raw_response = invoke_llm(str(image_path), mode, dimension)
                        raw_response = raw_response.replace("'", '"').replace('```', "").replace('json:', "").replace('json', "")
                        parsed = extract_json(raw_response) if mode == "row" else json.loads(raw_response)
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(parsed, f, indent=4)
                        success = True
                    except Exception as e:
                        print(f"❌ Error on {image_name}, attempt {attempts}: {e}")

# --------------- ENTRY POINT ---------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process 2D/3D images in simple or row mode using LLM.")
    parser.add_argument("--mode", type=str, choices=["simple", "row"], required=True, help="Annotation mode: 'simple' or 'row'")
    parser.add_argument("--dimension", type=str, choices=["2d", "3d"], required=True, help="Dataset type: '2d' or '3d'")
    parser.add_argument("--input", type=str, required=True, help="Root input directory (e.g., base_data or row_data)")
    parser.add_argument("--output", type=str, required=True, help="Root output directory for saving JSON results")

    args = parser.parse_args()

    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        mode=args.mode,
        dimension=args.dimension
    )
