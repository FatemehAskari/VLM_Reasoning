import argparse
import os
from itertools import product
from typing import List
from tqdm import tqdm
import sys
import random

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# Add the parent directory to the sys.path list so we can import utils.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *


def paste_shape(shape: np.ndarray, 
                positions: np.ndarray, 
                canvas_img: Image.Image, 
                i: int, 
                img_size: int = 90, 
                canvas_size: int = 556, 
                max_attempts: int = 100):
    """
    Paste a shape onto a canvas image at a random position.

    Parameters:
    shape (np.ndarray): The shape to be pasted.
    positions (np.ndarray): The positions of the shapes on the canvas.
    canvas_img (Image.Image): The canvas image.
    i (int): The index of the current shape.
    img_size (int): The size of the shape. Default is 40.
    canvas_size (int): The size of the canvas. Default is 300.
    max_attempts (int): Maximum number of attempts to find a free spot. Default is 100.

    Returns:
    np.ndarray: The updated positions of the shapes on the canvas.

    Raises:
    RuntimeError: If no suitable position is found after max_attempts.
    """
    img = Image.fromarray(np.transpose(shape, (1, 2, 0)))
    
    attempt = 0
    # Adjust position range to account for the shape size and canvas size
    while attempt < max_attempts:
        # Randomly choose a position, but make sure the whole shape fits inside the canvas
        position = np.random.randint(0, canvas_size - img_size, size=(1, 2))
        
        # Ensure the shape doesn't overlap with previous shapes
        if not np.any(np.linalg.norm(positions - position, axis=1) < img_size):
            # Good position found
            canvas_img.paste(img, tuple(position.squeeze()))
            positions[i] = position
            
            return positions
        attempt += 1
    
    # If no position found after max_attempts
    raise RuntimeError(f"Could not find a non-overlapping position after {max_attempts} attempts.")




def place_shapes(shape_imgs, img_size=72):
    # Define the canvas to draw images on, font, and drawing tool.
    canvas = np.ones((3, 556, 556), dtype=np.uint8) * 255
    canvas = np.transpose(canvas, (1, 2, 0))  # Transpose to (556x556x3) for PIL compatibility.
    canvas_img = Image.fromarray(canvas)
    # Add the shapes to the canvas.
    n_shapes = len(shape_imgs)
    positions = np.zeros([n_shapes, 2])
    for i, img in enumerate(shape_imgs):
        positions = paste_shape(img, positions, canvas_img, i, img_size=img_size)
    return canvas_img,positions


def make_binding_trial(shapes: np.ndarray, 
                       colors: np.ndarray,
                       shape_names: List[str], 
                       n_objects: int = 5, 
                       n_shapes: int = 5, 
                       n_colors: int = 5, 
                       img_size: int = 28):
    """
    Generate one binding trial with specified numbers of objects, shapes, and colors.
    Handles sampling safely to avoid ValueError from np.random.choice.
    """

    # ✅ Adjust n_shapes and n_colors if they exceed available pool
    n_shapes = min(n_shapes, len(shapes))
    n_colors = min(n_colors, len(colors))

    # Sample unique shape indices
    unique_shape_inds = np.random.choice(len(shapes), n_shapes, replace=False)
    shape_inds = np.concatenate([
        unique_shape_inds,
        np.random.choice(unique_shape_inds, n_objects - n_shapes, replace=True)
    ])

    # Sample unique color indices
    unique_color_inds = np.random.choice(len(colors), random.randint(1, 10), replace=True)
    print(unique_color_inds,len(colors),n_colors)
    color_inds = np.concatenate([
        unique_color_inds,
        np.random.choice(unique_color_inds, n_objects - n_colors, replace=True)
    ])

    # Prepare shapes and colors
    shape_imgs = shapes[shape_inds]
    selected_colors = np.array(colors)[color_inds]

    object_features = [
        {'shape': shape_names[shape], 'color': color}
        for shape, color in zip(shape_inds, selected_colors)
    ]

    # Convert color names to RGB
    rgb_codes = np.array([mcolors.to_rgba(color)[:-1] for color in selected_colors])
    colored_imgs = [
        color_shape(img.astype(np.float32), rgb)
        for img, rgb in zip(shape_imgs, rgb_codes)
    ]
    resized_imgs = [resize(img, img_size=img_size) for img in colored_imgs]

    # Create final scene image
    counting_trial,positions = place_shapes(resized_imgs, img_size=img_size + 10)

    return counting_trial, object_features,positions


def parse_args() -> argparse.Namespace:
	'''
	Parse command line arguments.

	Returns:
	argparse.Namespace: The parsed command line arguments.
	'''
	parser = argparse.ArgumentParser(description='Generate feature binding trials.')
	parser.add_argument('--n_objects', type=int, nargs='+', default=[15], help='Number of stimuli to present.')
	parser.add_argument('--n_trials', type=int, default=100, help='Number of trials to generate per n_shapes condition.')
	parser.add_argument('--size', type=int, default=40, help='Size of the shapes to paste in the image.')
	parser.add_argument('--color_names', type=str, nargs='+', default=['red', 'green', 'blue', 'gold', 'purple', 'orange', 'saddlebrown', 'pink', 'gray', 'black'], help='Colors to use for the shapes.')
	parser.add_argument('--shape_names', type=str, nargs='+', default=['square'], help='Names of the shapes to use in the trials.')
	parser.add_argument('--shape_inds', type=int, nargs='+', default=[101], help='Indices of the shapes to include in the trials.')
	parser.add_argument('--output_dir', type=str, default='data/vlm/binding_15', help='Directory to save the generated trials.')
	return parser.parse_args()

def main():
    # Fix the random seed for reproducibility.
    np.random.seed(88)
    nn = 0
    bb = False

    # Load shape images and trial configuration.
    args = parse_args()
    basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    imgs = np.load(os.path.join(basepath, 'imgs.npy'))
    imgs = imgs[args.shape_inds]
    assert len(args.shape_names) == len(args.shape_inds)

    # Create directory for binding task.
    os.makedirs(os.path.join(basepath, args.output_dir, 'images'), exist_ok=True)

    # Initialize DataFrame for storing task metadata later.
    metadata_df = pd.DataFrame(columns=[
        'path', 'n_objects', 'n_shapes', 'n_colors',
        'features', 'shapes_names', 'color_names', 'positions'
    ], dtype=object)

    # Generate the trials.
    for n in tqdm(args.n_objects):
        if bb:
            break

        # Task conditions to generate
        task_conditions = list(product(range(1, n + 1), range(1, n + 1)))
        condition_feature_counts = np.vstack(task_conditions).sum(axis=1)
        counts, count_freq = np.unique(condition_feature_counts, return_counts=True)

        for n_features, (n_shapes, n_colors) in zip(condition_feature_counts, task_conditions):
            if bb:
                break
            n_trials = int(np.ceil(args.n_trials / count_freq[counts == n_features][0]))

            for i in range(n_trials):
                nn += 1
                if nn > 100:
                    bb = True
                    break

                trial, features, positions = make_binding_trial(
                    imgs, np.array(args.color_names),
                    shape_names=args.shape_names,
                    n_objects=n,
                    n_shapes=n_shapes,
                    n_colors=n_colors,
                    img_size=args.size
                )

                fname = f'nObjects={n}_nShapes={n_shapes}_nColors={n_colors}_{i}.png'
                trial_path = os.path.join(basepath, args.output_dir, 'images', fname)
                trial.save(trial_path)

                row = {
                    'path': trial_path,
                    'n_objects': n,
                    'n_shapes': n_shapes,
                    'n_colors': n_colors,
                    'features': features,
                    'shapes_names': args.shape_names,
                    'color_names': args.color_names,
                    'positions': positions
                }
                metadata_df = metadata_df._append(row, ignore_index=True)

    # Save to CSV
    metadata_df.to_csv(os.path.join(basepath, args.output_dir, 'metadata.csv'), index=False)

if __name__ == '__main__':
    main()

