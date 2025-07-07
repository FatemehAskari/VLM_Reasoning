# Visual Structures Helps Visual Reasoning: Addressing the Binding Problem in VLMs
\[[Read our arXiv Paper](https://arxiv.org/pdf/2506.22146)\] 

Amirmohammad Izadi,Mohammad Ali Banayeeanzade, Fatemeh Askari,Ali Rahimiakbar,Mohammad Mahdi Vahedi, Hosein Hasani,Mahdieh Soleymani Baghshah


### Introduction
<div style="text-align: justify">

Despite recent advancements in **Vision-Language Models (VLMs)**, these models often struggle with tasks that require associating visual features—like shape and color—with their correct positions. This issue, known as the *binding problem*, leads to failures in tasks such as **counting**, **visual search**, **scene description**, and understanding **spatial relationships**.

Our work introduces a simple but powerful fix: we add lightweight **visual structures** (e.g., horizontal lines) to the images and pair them with task-specific prompts that guide the model to reason in a **spatially grounded**, sequential manner. As shown in the example below, this structured setup helps models like **GPT-4o** reason more accurately—achieving significant improvements across multiple tasks.

</div>
<p align="center"> <img src="assets/final_image.jpg" width="800" alt="Visual Structuring Example"> </p>

## Installation

To set up the environment and install the required dependencies for this project, follow the steps below.

```bash
git clone https://github.com/FatemehAskari/VLM_Reasoning.git
cd code
pip install --upgrade pip 
pip install -r requirements.txt
```

## Dataset

To generate the datasets used in our experiments, navigate to the `data/task_scripts` directory. Each dataset is generated using a Python script with configurable arguments, and supports both 2D and 3D visual settings (for Visual Search, Counting, and Scene Description).

---

### 🔧 Dataset Generation Commands

```bash
# Navigate to script directory
cd data/task_scripts

# Generate Visual Search Datasets
python task_scripts/gen_vlm_search.py --n_objects 5 10 15 20 25 30 35 40 45 50 \
                                      --n_trials=100 \
                                      --size=22 \
                                      --use_letters=True \
                                      --colors green red \
                                      --output_dir=data/vlm/search

                                      
python generate_visual_search.py --dimension 3d --output_dir ../visual_search/3d --num_samples 1000

# Generate Counting Datasets
python generate_counting.py --dimension 2d --output_dir ../counting/2d --num_samples 1000
python generate_counting.py --dimension 3d --output_dir ../counting/3d --num_samples 1000

# Generate Scene Description Datasets
python generate_scene_description.py --dimension 2d --output_dir ../scene_description/2d --num_samples 500
python generate_scene_description.py --dimension 3d --output_dir ../scene_description/3d --num_samples 500

# Generate Reasoning Dataset
python generate_reasoning.py --mode hard --output_dir ../reasoning --num_samples 200