# Visual Structures Help Visual Reasoning
Despite recent advancements in Vision-Language Models (VLMs), these models often struggle with tasks that require associating visual features—like shape and color—with their correct positions. This issue, known as the binding problem, leads to failures in tasks like counting, visual search,scene description and understanding spatial relationships.

Our work introduces a simple but powerful fix: we add lightweight visual structures (e.g., horizontal lines) to the images and pair them with task-specific prompts that guide the model to reason in a spatially grounded, sequential manner. As shown in the example below, this structured setup helps models like GPT-4o reason more accurately—achieving significant improvements across multiple tasks.

<p align="center"> <img src="assets/final_image.jpg" width="600" alt="Visual Structuring Example"> </p>