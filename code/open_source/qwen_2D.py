import os
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ------------------ Load Model ------------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    temperature=0.0,
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


# ------------------ Build Prompt ------------------
def build_prompt_text(mode: str) -> str:
    shapes = (
        "airplane, triangle, cloud, cross, umbrella, scissors, heart, star, "
        "circle, square, infinity, up-arrow, pentagon, left-arrow, flag"
    )
    colors = (
        "red, magenta, salmon, green, lime, olive, blue, teal, yellow, purple, "
        "brown, gray, black, cyan, orange"
    )

    base_prompt = (
        "** Task: **\n"
        "You are presented with an image containing multiple colored objects, "
        "each defined by a shape and a color.\n"
        "Your task is to identify all objects in the image and return a list of dictionaries.\n"
        "Each dictionary must contain exactly two keys: 'shape' and 'color'.\n"
        f"Only use shapes from this list: {shapes}.\n"
        f"Only use colors from this list: {colors}.\n"
        "Always respond **only** with a JSON list of the detected objects.\n"
    )

    if mode == "row":
        base_prompt += "Scan sequentially based on horizontal lines and numbers in the image."

    return base_prompt


# ------------------ Model Inference ------------------
def invoke_llm(image_path: str, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


# ------------------ Response Cleaning ------------------
def clean_and_parse_response(response: str) -> list:
    try:
        cleaned = (
            response.replace("'", '"')
                    .lower()
                    .replace("```", "")
                    .replace("json:", "")
                    .replace("json", "")
                    .strip()
        )
        return json.loads(cleaned)
    except Exception as e:
        print(f"⚠️ Warning: Failed to parse response: {e}")
        return None


# ------------------ Folder Processor ------------------
def process_all_image_folders(input_dir: str, output_dir: str, mode: str) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt_text(mode)

    for object_folder in sorted(input_dir.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else float('inf')):
        if not object_folder.is_dir() or not object_folder.name.isdigit():
            continue

        object_count = object_folder.name
        print(f"📁 Object Count Folder: {object_count}")

        for triplet_folder in tqdm(
            sorted(object_folder.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else float('inf')),
            desc=f"Triplet Folders in {object_count}"
        ):
            if not triplet_folder.is_dir() or not triplet_folder.name.isdigit():
                continue

            triplet_count = triplet_folder.name
            print(f"\n🔍 Processing: {object_count} objects, {triplet_count} triplets")

            triplet_output_dir = output_dir / object_count / triplet_count
            triplet_output_dir.mkdir(parents=True, exist_ok=True)

            output_json_path = triplet_output_dir / "qwen.json"
            results = {}

            image_files = sorted(triplet_folder.glob("*.png"), key=lambda x: x.name)
            if not image_files:
                continue

            for image_path in tqdm(image_files, desc="Processing Images"):
                image_name = image_path.stem
                output_path = triplet_output_dir / f"{image_name}.json"

                if output_path.exists():
                    continue

                success = False
                attempt = 0

                while not success and attempt < 3:
                    response = invoke_llm(str(image_path), prompt)
                    parsed = clean_and_parse_response(response)

                    if parsed is not None:
                        results[image_name] = parsed
                        success = True
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(parsed, f, indent=4)
                    else:
                        print(f"⚠️ Error for {image_name}, retrying ({attempt+1}/3)")
                        attempt += 1

            try:
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4)
            except Exception as e:
                print(f"❌ Failed to write JSON file: {output_json_path} | Error: {e}")


# ------------------ CLI Entry ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process 2D images in simple or row mode using Qwen2.5-VL model.")
    parser.add_argument("--mode", type=str, choices=["simple", "row"], required=True,
                        help="Annotation mode: 'simple' or 'row'")
    parser.add_argument("--input", type=str, required=True,
                        help="Root input directory (e.g., base_data_20 or base_data_20_another_row)")
    parser.add_argument("--output", type=str, required=True,
                        help="Root output directory for saving JSON results")

    args = parser.parse_args()

    process_all_image_folders(
        input_dir=args.input,
        output_dir=args.output,
        mode=args.mode
    )
