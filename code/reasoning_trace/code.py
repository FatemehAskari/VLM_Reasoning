import os
import json
import base64
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# -------------------- LLM Setup --------------------

# Set up LLM
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)
# -------------------- Image Encoding --------------------
def encode_image(image_path):
    """Convert an image to base64 encoding."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# -------------------- Prompt Builder --------------------
def build_prompt(mode: str) -> str:
    """Generate the appropriate prompt text depending on mode ('simple' or 'row')."""
    if mode == "simple":
        return (
            "**Task:**\n"
            "You are given an image that contains multiple colored squares.\n"
            "Your job is to:\n"
            "1. Divide the image into 4 equal horizontal sections from top to bottom. Label them as \"1\", \"2\", \"3\", and \"4\".\n"
            "2. Identify every square in the image and assign it to the correct section it belongs to.\n"
            "- If a square overlaps two sections, assign it to the section where the majority of the square lies.\n"
            "3. Each square must be described using two properties:\n"
            "- shape: always \"square\"\n"
            "- color: one of the following:\n"
            "  red, magenta, salmon, green, lime, olive, blue, teal, yellow, purple, brown, gray, black, cyan, orange\n\n"
            "Return your answer in this JSON format (valid JSON only, no explanations):\n\n"
            "{\n"
            "  \"1\": [ {\"shape\": \"square\", \"color\": \"green\"}, ... ],\n"
            "  \"2\": [ {\"shape\": \"square\", \"color\": \"red\"}, ... ],\n"
            "  \"3\": [ ... ],\n"
            "  \"4\": [ ... ]\n"
            "}"
        )
    else:  # mode == "row"
        return (
            "**Task:**\n"
            "You are given an image that contains multiple colored squares.\n"
            "The image is divided into 4 horizontal rows using horizontal black lines and visible numbers (1 to 4) along the left margin.\n\n"
            "Your job is to:\n"
            "1. Detect every square in the image.\n"
            "2. Based on the horizontal lines and row numbers shown in the image, assign each square to the correct row: \"1\", \"2\", \"3\", or \"4\".\n"
            "- If a square crosses two rows, assign it to the row where the majority of the square is located.\n\n"
            "Each square must be described with:\n"
            "- shape: always \"square\"\n"
            "- color: one of the following:\n"
            "  red, magenta, salmon, green, lime, olive, blue, teal, yellow, purple, brown, gray, black, cyan, orange\n\n"
            "Return your output in the following JSON format:\n\n"
            "{\n"
            "  \"1\": [ {\"shape\": \"square\", \"color\": \"...\"}, ... ],\n"
            "  \"2\": [ ... ],\n"
            "  \"3\": [ ... ],\n"
            "  \"4\": [ ... ]\n"
            "}"
        )

# -------------------- Model Invocation --------------------
def invoke_llm(image_path, prompt):
    """Send image and prompt to LLM and return the response text."""
    image_base64 = encode_image(image_path)
    ext = os.path.splitext(image_path)[1][1:]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful vision assistant."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
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

# -------------------- Clean Response --------------------
def clean_response_json(raw_text):
    """Remove markdown formatting and clean JSON string."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

# -------------------- Batch Processing --------------------
def process_image_directory(input_dir, output_dir, mode):
    """Process all .png images in the input directory and save results as JSON."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(mode)

    for image_file in tqdm(sorted(input_path.glob("*.png")), desc="Processing images"):
        try:
            response = invoke_llm(str(image_file), prompt)
            print(f"📤 Raw response from {image_file.name}:\n{response}\n")

            cleaned_json = clean_response_json(response)
            parsed = json.loads(cleaned_json)

            # Structure result with filename as key
            structured = {image_file.name: parsed}

            json_file_path = output_path / f"{image_file.stem}.json"
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(structured, f, indent=4)

            print(f"✅ Saved: {json_file_path.name}\n")

        except Exception as e:
            print(f"❌ Error processing {image_file.name}: {e}\n")


# -------------------- CLI Entry --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision-based square labeling using GPT-4o.")
    parser.add_argument("--mode", choices=["simple", "row"], required=True, help="Prompt mode: 'simple' or 'row'")
    parser.add_argument("--input", type=str, required=True, help="Path to folder containing input .png images")
    parser.add_argument("--output", type=str, required=True, help="Path to folder for saving output .json results")
    args = parser.parse_args()

    process_image_directory(
        input_dir=args.input,
        output_dir=args.output,
        mode=args.mode
    )
