import argparse
import os
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import sys

# Add the parent directory to the sys.path list so we can import utils.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *


def make_counting_trial(imgs, n_shapes=10, n_unique=5, size=32, uniform=False, sigma=0.1, all_black=False):
	# sample the shapes to include in the trial.
	unique_inds = np.random.choice(len(imgs), n_unique, replace=False)
	shape_inds = np.random.choice(unique_inds, n_shapes, replace=True)
	shape_imgs = imgs[shape_inds]
	# color the shapes
	mu = np.random.uniform(0,1)
	colors = generate_isoluminant_colors(n_shapes, mu=mu, sigma=sigma, uniform=uniform)
	colored_imgs = [color_shape(img.astype(np.float32), rgb, all_black=all_black) for img, rgb in zip(shape_imgs, colors)]
	resized_imgs = [resize(img, img_size=size) for img in colored_imgs]
	counting_trial = place_shapes(resized_imgs, img_size=size)
	return counting_trial

def generate_isoluminant_colors(num_colors, saturation=1, lightness=0.8, mu=0.5, sigma=0.1, uniform=False):
	if uniform:
		hues = np.linspace(0, 1, num_colors, endpoint=False)
	else:
		hues = np.random.normal(loc=mu, scale=sigma, size=num_colors) % 1.0
	hsl_colors = [(hue, saturation, lightness) for hue in hues]
	rgb_colors = [mcolors.hsv_to_rgb(color) for color in hsl_colors]
	return rgb_colors

def parse_args() -> argparse.Namespace:
	"""
	Parse command line arguments.

	Returns:
	argparse.Namespace: The parsed command line arguments.
	"""
	parser = argparse.ArgumentParser(description='Generate counting trials.')
	parser.add_argument('--n_shapes', type=int, nargs='+', default=[2,4,6,8,10], help='Number of stimuli to present.')
	parser.add_argument('--n_trials', type=int, default=100, help='Number of trials to generate per n_shapes condition.')
	parser.add_argument('--size', type=int, default=40, help='Size of the shapes to paste in the image.')
	parser.add_argument('--object_inds', type=int, nargs='+', default=[37], help='Indices of the objects to include in the trials.')
	parser.add_argument('--n_unique', type=int, default=1, help='Number of unique object shapes to include on each trial.')
	parser.add_argument('--uniform', type=bool, default=False, help='Whether to use uniform colors (i.e. maximally distinct)')
	parser.add_argument('--sigma', type=float, default=0.1, help='Standard deviation of the hue distribution.')
	parser.add_argument('--all_black', type=bool, default=False, help='Whether to color all shapes black.')
	parser.add_argument('--output_dir', type=str, default='data/vlm/counting', help='Directory to save the generated trials.')
	return parser.parse_args()

def main():
	# Fix the random seed for reproducibility.
	np.random.seed(88)
	
	# Parse command line arguments.
	args = parse_args()
	basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	assert args.n_unique <= len(args.object_inds), 'Number of unique objects must be less than or equal to the number of objects.'
	imgs = np.load(os.path.join(basepath, 'imgs.npy'))
	imgs = imgs[np.array(args.object_inds)] # sample only the shapes that we want to include in the trials.

	# Create directory for serial search exists.
	os.makedirs(os.path.join(basepath, args.output_dir, 'images'), exist_ok=True)

	# Initialize DataFrame for storing task metadata_df.
	metadata_df = pd.DataFrame(columns=['path', 'n_objects'])

	# Generate the trials.
	for n in tqdm(args.n_shapes):
		for i in range(args.n_trials):
			# Save the trials and their metadata.
			trial = make_counting_trial(imgs, n_shapes=n, n_unique=args.n_unique, size=args.size, uniform=args.uniform, sigma=args.sigma, all_black=args.all_black)
			trial_path = os.path.join(basepath, args.output_dir, 'images', f'trial-{n}_{i}.png')
			trial.save(trial_path)
			metadata_df = metadata_df._append({'path': trial_path, 'n_objects': n}, ignore_index=True)

	# Save results DataFrame to CSV
	metadata_df.to_csv(os.path.join(basepath, args.output_dir, 'metadata.csv'), index=False)

if __name__ == '__main__':
	main()