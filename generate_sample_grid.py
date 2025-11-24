#!/usr/bin/env python3
"""Generate a 3x4 grid of random font samples for reporting"""

import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def generate_sample_grid(data_dir="data", output_path="sample_grid.png", rows=4, cols=3):
    """Create a grid of random font samples"""

    data_path = Path(data_dir)

    # Get all font directories
    font_dirs = [d for d in data_path.iterdir() if d.is_dir()]

    if len(font_dirs) < rows * cols:
        print(f"Warning: Only {len(font_dirs)} fonts available, need {rows * cols}")
        rows = len(font_dirs) // cols

    # Randomly select fonts
    selected_fonts = random.sample(font_dirs, rows * cols)

    # Create figure with gridspec for better control
    fig, axes = plt.subplots(rows, cols, figsize=(12, 16))

    # Flatten axes for easier iteration
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (font_dir, ax) in enumerate(zip(selected_fonts, axes)):
        # Get all images in this font directory
        images = list(font_dir.glob("*.png"))

        if not images:
            print(f"Warning: No images found in {font_dir.name}")
            ax.axis('off')
            continue

        # Randomly select one image
        img_path = random.choice(images)

        # Load and display image
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(font_dir.name.replace('_', ' '), fontsize=10, pad=5)

        # Add black border around each cell
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        # Hide ticks but keep borders
        ax.set_xticks([])
        ax.set_yticks([])

    # Adjust spacing to show grid lines clearly
    plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.02, right=0.98, top=0.97, bottom=0.02)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Sample grid saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate a grid of random font samples')
    parser.add_argument('--data-dir', default='data', help='Path to data directory')
    parser.add_argument('--output', default='sample_grid.png', help='Output image path')
    parser.add_argument('--rows', type=int, default=4, help='Number of rows')
    parser.add_argument('--cols', type=int, default=3, help='Number of columns')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    generate_sample_grid(args.data_dir, args.output, args.rows, args.cols)
