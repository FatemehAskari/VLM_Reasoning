from langchain_openai import ChatOpenAI
import base64
import os
import re
import json
from pathlib import Path

base_url = "https://api.avalai.ir/v1"
api_key = "aa-IbxIACL4oknjL1lneG03Cgum5IrWc0PGV5KyH8JwU3At7yj3"
model_name = "gpt-4o"

llm = ChatOpenAI(
    base_url=base_url,
    model=model_name,
    api_key=api_key,
    temperature=0,
)

EXAMPLE_IMAGE_PATH = "/home/mmd/vlm-binding-main/data/vlm/counting_colored/images_13objects_random_samples_with_grid/trial-13_10.png" 
TEST_IMAGE_FOLDER = "/home/mmd/vlm-binding-main/data/vlm/counting_colored/images_13objects_random_samples_with_grid"    
OUTPUT_JSON_PATH = "results_counting_gpt4o_13objects_ours.json"    

def encode_image(image_path):
    """Encodes an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def invoke_llm(test_image_path, test_image_ext):
    
    example_base64_image = encode_image(EXAMPLE_IMAGE_PATH)
    test_base64_image = encode_image(test_image_path)
    example_image_ext = os.path.splitext(EXAMPLE_IMAGE_PATH)[1][1:]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": 
                        """
                            **Task:** 
                            Count the number of circles in a 4x4 grid. Each cell is labeled as "Row-Column" (e.g., "3-2" = Row 3, Column 2) and separated by black lines. 

                            **Rules:**  
                                Majority Rule:  
                                    Count a circle only in the cell where most of its area lies.
                                    If a circle overlaps multiple cells, assign it to the cell containing the majority of its area.
                                    Do not count the same circle more than once. 
                                    
                                Multiple Circles:  
                                    A single cell can contain multiple circles. Count all circles whose majority area lies within the cell.
                                    
                                Sequential Scanning:  
                                    Process the cells sequentially, starting from 1-1  → 1-2  → ... → 4-4. 

                            **Steps:**
                                Scan each cell: Begin at 1-1  and proceed row by row until 4-4.
                                For each cell:
                                    Identify all circles where most of their area is inside the cell.
                                    Count each valid circle separately.
                                    Ignore circles where the majority of their area lies in another cell.
                                    
                                Verify no duplication:  Ensure that each circle is counted only once, based on the cell where its majority area is located.
                                Summarize the counts:  After processing all cells, sum the counts from the Count column to determine the total number of circles.
                            
                            **Output Format:**
                            Provide the results in a table format like this:

                            | Cell | Count | Notes (Optional)  |
                            |------|-------|-------------------|
                            | 1-1  |   2   |     Two circles   |
                            | 1-2  |   1   |     One circle    |
                            | 1-3  |   0   |     No circles    |
                            | ...  |  ...  |     ...           |

                            Finally, provide the **Total Number of Circles** at the end.

                            **Example Input and Output:**
                            Below is an example image and its corresponding output to help you understand the task.
                        """
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{example_image_ext};base64,{example_base64_image}",
                        "detail": "auto",
                    },
                },
                {
                    "type": "text",
                    "text": 
                    """
                        **Example Output for the Above Image:**

                        | Cell | Count | Notes (Optional)  |
                        |------|-------|-------------------|
                        | 1-1  | 1     | One circle        |
                        | 1-2  | 0     | No circles        |
                        | 1-3  | 1     | One circle        |
                        | 1-4  | 1     | One circle        |
                        | 2-1  | 0     | No circles        |
                        | 2-2  | 1     | One circle        |
                        | 2-3  | 1     | One circle        |
                        | 2-4  | 0     | No circles        |
                        | 3-1  | 3     | Three circles     |
                        | 3-2  | 2     | Two circles       |
                        | 3-3  | 1     | One circle        |
                        | 3-4  | 0     | No circles        |
                        | 4-1  | 0     | No circles        |
                        | 4-2  | 1     | One circle        |
                        | 4-3  | 0     | No circles        |
                        | 4-4  | 1     | One circle        |

                        **Total Number of Circles:** 1 + 0 + 1 + 1 + 0 + 1 + 1 + 0 + 3 + 2 + 1 + 0 + 0 + 1 + 0 + 1 = 13

                        Now, process the following test image and provide the output in the same format.
                    """
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{test_image_ext};base64,{test_base64_image}",
                        "detail": "auto",
                    },
                },
                {
                    "type": "text",
                    "text": "The above image is the test input. Perform the task on this image and provide the output in the specified format."
                },
            ],
        },
    ]

    ai_message = llm.invoke(messages)
    return ai_message.content

def extract_total_circles(response):
    total_circles_pattern = r"\*\*Total Number of Circles:\*\*\s*(?:\d+\s*\+?\s*)*=\s*(\d+)|\*\*Total Number of Circles:\*\*\s*(\d+)"
    match = re.search(total_circles_pattern, response)
    if match:
        return int(match.group(1) or match.group(2))
    return None

def process_folder(folder_path, output_json_path):
    output_data = {}

    for image_path in Path(folder_path).glob("*.*"):
        if image_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue 

        print(f"Processing image: {image_path}")

        test_image_ext = image_path.suffix[1:]

        llm_response = invoke_llm(image_path, test_image_ext)
        print(llm_response)
        # exit()

        total_circles = extract_total_circles(llm_response)

        output_data[str(image_path)] = total_circles
        print(f"Processed {image_path}, Total Circles: {total_circles}")

        with open(output_json_path, "w") as json_file:
            json.dump(output_data, json_file, indent=4)

    print(f"All images processed. Final results saved to {output_json_path}.")


process_folder(TEST_IMAGE_FOLDER, OUTPUT_JSON_PATH)