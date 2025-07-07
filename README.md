# Visual Structures Helps Visual Reasoning: Addressing the Binding Problem in VLMs
\[[Read our arXiv Paper](https://arxiv.org/pdf/2506.22146)\] 

Amirmohammad Izadi,Mohammad Ali Banayeeanzade, Fatemeh Askari,Ali Rahimiakbar,Mohammad Mahdi Vahedi, Hosein Hasani,and Mahdieh Soleymani Baghshah


### Introduction
<div style="text-align: justify">

Despite recent advancements in **Vision-Language Models (VLMs)**, these models often struggle with tasks that require associating visual features—like shape and color—with their correct positions. This issue, known as the *binding problem*, leads to failures in tasks such as **counting**, **visual search**, **scene description**, and understanding **spatial relationships**.

Our work introduces a simple but powerful fix: we add lightweight **visual structures** (e.g., horizontal lines) to the images and pair them with task-specific prompts that guide the model to reason in a **spatially grounded**, sequential manner. As shown in the example below, this structured setup helps models like **GPT-4o** reason more accurately—achieving significant improvements across multiple tasks.

</div>
<p align="center"> <img src="assets/final_image.jpg" width="800" alt="Visual Structuring Example"> </p>

## Installation

To set up the environment and install the required dependencies for this project, follow the steps below.

### 1. Clone the repository

```bash
git clone https://github.com/FatemehAskari/VLM_Reasoning.git
cd code
pip install --upgrade pip 
pip install -r requirements.txt
