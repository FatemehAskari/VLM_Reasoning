import argparse
import os
from itertools import product
from typing import List
from tqdm import tqdm
import sys

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from PIL import Image

# Add the parent directory to the sys.path list so we can import utils.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

def paste_shape(shape: np.ndarray, 
                positions: np.ndarray, 
                canvas_img: Image.Image, 
                i: int, 
                img_size: int, 
                canvas_size: int, 
                max_attempts: int = 100):
    """
    Paste a shape onto a canvas image at a random position.

    Parameters:
    shape (np.ndarray): The shape to be pasted.
    positions (np.ndarray): The positions of the shapes on the canvas.
    canvas_img (Image.Image): The canvas image.
    i (int): The index of the current shape.
    img_size (int): The size of the shape.
    canvas_size (int): The size of the canvas.
    max_attempts (int): Maximum number of attempts to find a free spot. Default is 100.

    Returns:
    np.ndarray: The updated positions of the shapes on the canvas.

    Raises:
    RuntimeError: If no suitable position is found after max_attempts.
    """
    img = Image.fromarray(np.transpose(shape, (1, 2, 0)))
    
    attempt = 0
    while attempt < max_attempts:
        position = np.random.randint(0, canvas_size - img_size, size=(1, 2))
        if not np.any(np.linalg.norm(positions - position, axis=1) < img_size):
            canvas_img.paste(img, tuple(position.squeeze()))
            positions[i] = position
            return positions
        attempt += 1
    
    raise RuntimeError(f"Could not find a non-overlapping position after {max_attempts} attempts.")

def place_shapes(shape_imgs, img_size, canvas_size):
    canvas = np.ones((3, canvas_size, canvas_size), dtype=np.uint8) * 255
    canvas = np.transpose(canvas, (1, 2, 0))
    canvas_img = Image.fromarray(canvas)
    n_shapes = len(shape_imgs)
    positions = np.zeros([n_shapes, 2])
    for i, img in enumerate(shape_imgs):
        positions = paste_shape(img, positions, canvas_img, i, img_size=img_size, canvas_size=canvas_size)
    return canvas_img

def make_binding_trial(
    shapes: np.ndarray, 
    colors: np.ndarray,
    shape_names: List[str], 
    n_objects: int = 5, 
    n_shapes: int = 5, 
    n_colors: int = 5, 
    img_size: int = 28,
    num: str = "5"
):
    unique_shape_inds = np.random.choice(len(shapes), n_shapes, replace=False)
    shape_inds = np.concatenate([
        unique_shape_inds, 
        np.random.choice(unique_shape_inds, n_objects - n_shapes, replace=True)
    ])
    unique_color_inds = np.random.choice(len(colors), n_colors, replace=False)
    color_inds = np.concatenate([
        unique_color_inds, 
        np.random.choice(unique_color_inds, n_objects - n_colors, replace=True)
    ])
    shape_imgs = shapes[shape_inds]
    colors = colors[color_inds]
    object_features = [
        {'shape': shape_names[shape], 'color': color} 
        for shape, color in zip(shape_inds, colors)
    ]
    count_triplet(object_features, n_objects, num)
    rgb_codes = np.array([
        mcolors.to_rgba(color)[:-1] 
        for color in colors
    ])
    colored_imgs = [
        color_shape(img.astype(np.float32), rgb) 
        for img, rgb in zip(shape_imgs, rgb_codes)
    ]
    resized_imgs = [
        resize(img, img_size=img_size) 
        for img in colored_imgs
    ]
    counting_trial = place_shapes(resized_imgs, img_size=img_size + 10, canvas_size=img_size + 100)
    return counting_trial, object_features

def make_binding_triplet(
    shapes: np.ndarray, 
    colors: np.ndarray,
    num_triplet,
    num: str,
    shape_names: List[str], 
    n_objects: int = 5, 
    img_size: int = 28,
):
    object_features = []
    shape_inds = []
    color_inds = []
    app = []
    attempts = 0
    max_attempts = 1000
    count2 = 0
    while (count2 != num_triplet or len(object_features) != n_objects):
        if len(object_features) > n_objects:
            count2 = 0
            shape_inds = []
            color_inds = []
            object_features = []
        shape_ind = np.random.choice(len(shapes))
        color_ind = np.random.choice(len(colors))
        obj = {
            'shape': shape_names[shape_ind],
            'color': colors[color_ind]
        }
        object_features = object_features + [obj]
        count1, count2, app = count_triplet(object_features, len(object_features), num, n_objects)
        if count2 <= num_triplet:
            shape_inds.append(shape_ind)
            color_inds.append(color_ind)
        else:
            object_features.pop()
        if count2 > num_triplet:
            count2 = 0
            shape_inds = []
            color_inds = []
            object_features = []
        attempts += 1
        if attempts >= max_attempts:
            raise ValueError("Couldn't generate valid objects with the required number of triplets.")
    shape_imgs = shapes[shape_inds]
    rgb_codes = np.array([
        mcolors.to_rgba(colors[i])[:-1] 
        for i in color_inds
    ])
    colored_imgs = [
        color_shape(img.astype(np.float32), rgb) 
        for img, rgb in zip(shape_imgs, rgb_codes)
    ]
    resized_imgs = [
        resize(img, img_size=img_size) 
        for img in colored_imgs
    ]
    counting_trial = place_shapes(resized_imgs, img_size=img_size + 10, canvas_size=img_size + 100)
    return counting_trial, object_features, app

def count_triplet(object_features, n_objects, num, n):
    count1 = 0
    count2 = 0
    app = []
    for i in range(len(object_features)):
        for j in range(i + 1, len(object_features)):
            for k in range(j + 1, len(object_features)):
                if object_features[i]['shape'] == object_features[j]['shape'] and (
                    object_features[i]['color'] == object_features[k]['color'] or
                    object_features[j]['color'] == object_features[k]['color']
                ):
                    count1 += 1
                if check(object_features[i], object_features[j], object_features[k]) or \
                   check(object_features[i], object_features[k], object_features[j]) or \
                   check(object_features[j], object_features[k], object_features[i]):
                    app.append([object_features[i], object_features[j], object_features[k]])
                    count2 += 1
    return count1, count2, app

def check(obj1, obj2, obj3):
    if obj1['shape'] == obj2['shape'] and obj1['color'] != obj2['color'] and (
        (obj1['color'] == obj3['color'] and obj1['shape'] != obj3['shape']) or
        (obj2['color'] == obj3['color'] and obj2['shape'] != obj3['shape'])
    ):
        return True
    return False

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate feature binding trials.')
    parser.add_argument('--n_objects', type=int, nargs='+', default=[10, 15, 20], 
                       help='Number of stimuli to present. (default: [10, 15, 20])')
    parser.add_argument('--n_trials', type=int, default=15, 
                       help='Number of trials to generate per n_shapes condition. (default: 15)')
    parser.add_argument('--size', type=int, default=42, 
                       help='Size of the shapes (overridden: 356 for n_objects=10,15; 656 for n_objects=20)')
    parser.add_argument('--color_names', type=str, nargs='+', 
                       default=['red', 'magenta', 'salmon', 'green', 'lime', 'olive', 'blue', 'teal', 
                                'yellow', 'purple', 'brown', 'gray', 'black', 'cyan', 'orange'], 
                       help='Colors to use for the shapes.')
    parser.add_argument('--shape_names', type=str, nargs='+', 
                       default=['airplane', 'triangle', 'cloud', 'cross', 'umbrella', 'scissors', 
                                'heart', 'star', 'circle', 'square', 'infinity', 'up-arrow', 
                                'pentagon', 'left-arrow', 'flag'], 
                       help='Names of the shapes to use in the trials.')
    parser.add_argument('--shape_inds', type=int, nargs='+', 
                       default=[6, 9, 21, 24, 34, 60, 96, 98, 100, 101, 5, 22, 59, 13, 35], 
                       help='Indices of the shapes to include in the trials.')
    parser.add_argument('--output_dir', type=str, default='data_triplet_20', 
                       help='Directory to save the generated trials.')
    return parser.parse_args()

def main():
    np.random.seed(88)
    args = parse_args()
    basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    imgs = np.load(os.path.join(basepath, 'imgs.npy'))
    imgs = imgs[args.shape_inds]
    
    assert len(args.shape_names) == len(args.shape_inds)
    assert len(args.color_names) == len(args.shape_names)

    os.makedirs(os.path.join(basepath, args.output_dir, 'images'), exist_ok=True)

    for n in tqdm(args.n_objects):
        img_size = 356 if n in [10, 15] else 656
        num_triplet2 = {10: 21, 15: 51, 20: 71}.get(n, 21)
        for i in range(5, num_triplet2, 5):
            metadata_rows = []
            for j in range(20):
                fname = f'nObjects={n}_triplet={i}_{j}.png'
                trial, features, app = make_binding_triplet(
                    imgs, 
                    np.array(args.color_names), 
                    num_triplet=i,
                    num=fname, 
                    n_objects=n, 
                    img_size=img_size, 
                    shape_names=args.shape_names, 
                )
                trial_path = os.path.join(basepath, args.output_dir, str(n), str(i), fname)
                os.makedirs(os.path.dirname(trial_path), exist_ok=True)
                trial.save(trial_path)
                metadata_rows.append({
                    'path': trial_path,
                    'n_objects': n,
                    'features': features,
                    'shapes_names': args.shape_names,
                    'color_names': args.color_names,
                    'num_triplet': i,
                    'triplet': app
                })
            metadata_df = pd.DataFrame(metadata_rows)
            metadata_df.to_csv(
                os.path.join(basepath, args.output_dir, str(n), str(i), 'metadata.csv'), 
                index=False
            )

if __name__ == '__main__':
    main()