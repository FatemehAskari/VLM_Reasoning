import os
import re
import json
import base64
import argparse
from pathlib import Path
from langchain_openai import ChatOpenAI

# --------------- LLM Setup ---------------
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)
# ------------- Utility Functions -------------
def encode_image(image_path):
    """Convert image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_prompt(dimension: str, mode: str) -> str:
    """
    Build prompt depending on image type (2D or 3D) and scan mode (simple or row).
    """
    if dimension == "2d":
        task_text = """
**Task:** 
You are presented with an image containing a set of letters, specifically the letters 'L' and 'T'.
These letters will appear in either red or green.
Your task is to determine if there are any green 'L's in the image. Follow these steps carefully:
1. Describe each shape in the image, noting their color.
2. Conclude your response by stating [True] if the letter 'L' appears in green, or [False] if there are no green 'L's.
Enclose your final answer in square brackets (**Final Answer:** []).
"""
    else:  # 3d
        task_text = """
**Task:** 
You are presented with an image containing a set of objects, specifically spheres and cubes.
These objects will appear in either red or green.
Your task is to determine if there are any red spheres in the image. Follow these steps carefully:
1. Describe each object in the image, noting their color.
2. Conclude your response by stating [True] if a red sphere is present, or [False] if there are none.
Enclose your final answer in square brackets (**Final Answer:** []).
"""

    # Add scan instruction if mode is row
    if mode == "row":
        task_text += "\nScan the image sequentially based on horizontal lines that exist in the image."

    return task_text.strip()

def extract_final_answer(response: str):
    """Extract answer inside **Final Answer:** [ ... ]"""
    match = re.search(r"\*\*Final Answer:\*\*\s*\[([^\]]*)\]", response, re.IGNORECASE)
    return match.group(1) if match else None

def invoke_llm(image_path: Path, ext: str, dimension: str, mode: str) -> str:
    """Send the image and prompt to the LLM and get response."""
    base64_img = encode_image(image_path)
    prompt = build_prompt(dimension, mode)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
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

# ---------------- Main ----------------
def process_folder(input_dir: str, output_json_path: str, dimension: str, mode: str):
    input_dir = Path(input_dir)
    results = {}

    for image_path in sorted(input_dir.glob("*.*")):
        if image_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue

        ext = image_path.suffix[1:]
        print(f"🔍 Processing: {image_path.name}")

        try:
            response = invoke_llm(image_path, ext, dimension, mode)
            print(f"📩 LLM Response: {response.strip()}")
            answer = extract_final_answer(response)
            results[str(image_path)] = answer
            print(f"✅ Final Answer: {answer}\n")
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
            results[str(image_path)] = None

        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=4)

    print(f"\n📁 All results saved to: {output_json_path}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Search Prompting (2D/3D + simple/row)")
    parser.add_argument("--input", type=str, required=True, help="Path to image folder")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--dimension", type=str, choices=["2d", "3d"], required=True, help="Choose '2d' or '3d'")
    parser.add_argument("--mode", type=str, choices=["simple", "row"], required=True, help="Prompt mode: 'simple' or 'row'")
    args = parser.parse_args()

    process_folder(
        input_dir=args.input,
        output_json_path=args.output,
        dimension=args.dimension,
        mode=args.mode
    )
