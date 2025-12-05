#!/usr/bin/env python3
"""Benchmark frontier VLMs on font classification task"""

import random
import os
import base64
from pathlib import Path
from collections import Counter
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

from litellm import completion
from fonts import FONTS

from dotenv import load_dotenv
load_dotenv()

# Thread-safe print lock
print_lock = Lock()

def encode_image_base64(image_path):
    """Encode image to base64 string"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def get_random_samples(data_dir="data", num_samples=10, seed=42):
    """Get random samples from dataset"""
    random.seed(seed)

    data_path = Path(data_dir)
    font_dirs = [d for d in data_path.iterdir() if d.is_dir()]

    # Randomly select fonts
    selected_fonts = random.sample(font_dirs, min(num_samples, len(font_dirs)))

    samples = []
    for font_dir in selected_fonts:
        # Get all images in this font directory
        images = list(font_dir.glob("*.png"))
        if images:
            # Randomly select one image
            img_path = random.choice(images)
            samples.append({
                'path': img_path,
                'true_font': font_dir.name.replace('_', ' '),
                'true_font_underscored': font_dir.name
            })

    return samples

def create_system_prompt(fonts):
    """Create system prompt with font list"""
    fonts_list = ', '.join(sorted(fonts))

    prompt = f"""You are a font classification expert. Your task is to identify which font is used in the provided image.

The image contains text rendered in one of the following {len(fonts)} fonts:

{fonts_list}

CRITICAL: Respond with ONLY the exact font name from the list above.
- DO NOT include any explanation
- DO NOT include reasoning
- DO NOT include confidence scores
- DO NOT include any other text
- ONLY output the font name, nothing else

Example valid responses:
Arial
Times New Roman
Montserrat

Invalid responses:
"The font is Arial"
"Arial (high confidence)"
"I believe this is Arial"""

    return prompt

def predict_font(image_path, model, system_prompt):
    """Predict font using vision model"""

    # Encode image
    image_b64 = encode_image_base64(image_path)

    # Combine system prompt and question into user message
    # (some models don't support system role with vision)
    user_text = f"{system_prompt}\n\nWhat font is used in this image?"

    # Get API key based on model
    api_key = None
    if "gemini" in model:
        api_key = os.getenv("GEMINI_API_KEY")
    elif "gpt" in model:
        api_key = os.getenv("OPENAI_API_KEY")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                }
            ]
        }
    ]

    try:
        # gpt-5 requires temperature=1, other models can use 0
        temp = 1 if "gpt-5" in model else 0

        response = completion(
            model=model,
            messages=messages,
            api_key=api_key,
            max_tokens=4096,
            temperature=temp,
            timeout=120
        )

        content = response.choices[0].message.content
        if content is None:
            print(f"  Warning: Model returned None content")
            print(f"  Response object: {response}")
            return None

        prediction = content.strip()
        return prediction

    except Exception as e:
        with print_lock:
            print(f"  Error with {model}: {e}")
        return None

def normalize_font_name(name):
    """Normalize font name for comparison"""
    return name.lower().replace(' ', '').replace('_', '').replace('-', '')

def calculate_accuracy(results):
    """Calculate accuracy metrics"""
    correct = 0
    total = 0

    for result in results:
        if result['prediction'] is not None:
            total += 1

            # Normalize both names for comparison
            pred_norm = normalize_font_name(result['prediction'])
            true_norm = normalize_font_name(result['true_font'])

            if pred_norm == true_norm:
                correct += 1
                result['correct'] = True
            else:
                result['correct'] = False

    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, correct, total

def process_sample(args):
    """Worker function to process a single sample"""
    sample, model_id, system_prompt = args

    prediction = predict_font(sample['path'], model_id, system_prompt)

    result = {
        'image': str(sample['path']),
        'true_font': sample['true_font'],
        'prediction': prediction,
    }

    return result

def benchmark_models(data_dir="data", num_samples=100, seed=42, max_workers=8):
    """Benchmark both models with parallel requests"""

    print(f"Loading {num_samples} random samples from {data_dir}...")
    samples = get_random_samples(data_dir, num_samples, seed)
    print(f"Selected {len(samples)} samples")

    # Create system prompt with all fonts
    system_prompt = create_system_prompt(FONTS)
    print(f"\nSystem prompt created with {len(FONTS)} fonts")

    models = {
        "gemini-2.5-pro": "gemini/gemini-2.5-pro",
        "gpt-5": "gpt-5"
    }

    results = {}

    for model_name, model_id in models.items():
        print(f"\n{'='*60}")
        print(f"Testing {model_name} with {max_workers} parallel threads")
        print('='*60)

        # Prepare tasks for all samples
        tasks = [(sample, model_id, system_prompt) for sample in samples]

        model_results = []

        # Process samples in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_sample, task) for task in tasks]

            # Use tqdm to show progress
            for future in tqdm(as_completed(futures), total=len(futures),
                             desc=f"{model_name}", unit="sample"):
                result = future.result()
                model_results.append(result)

        # Calculate accuracy
        accuracy, correct, total = calculate_accuracy(model_results)

        results[model_name] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': model_results
        }

        print(f"\n{model_name} Accuracy: {correct}/{total} = {accuracy:.1f}%")

    return results

def print_summary(results):
    """Print summary comparison"""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    for model_name, data in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {data['correct']}/{data['total']} = {data['accuracy']:.1f}%")

        # Show mistakes
        mistakes = [r for r in data['results'] if not r.get('correct', False)]
        if mistakes:
            print(f"  Mistakes:")
            for m in mistakes:
                print(f"    - True: {m['true_font']} | Predicted: {m['prediction']}")

def save_results(results, output_path="benchmark_results.json"):
    """Save results to JSON"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

def plot_results(results, output_path="benchmark_plot.png"):
    """Create visualization of benchmark results"""
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    model_names = list(results.keys())

    # 1. Overall Accuracy Comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    accuracies = [results[m]['accuracy'] for m in model_names]
    colors = ['#4285F4', '#EA4335'][:len(model_names)]
    bars = ax1.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 2. Correct/Incorrect breakdown (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(model_names))
    width = 0.35

    correct_counts = [results[m]['correct'] for m in model_names]
    incorrect_counts = [results[m]['total'] - results[m]['correct'] for m in model_names]

    bars1 = ax2.bar(x, correct_counts, width, label='Correct', color='#34A853', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x, incorrect_counts, width, bottom=correct_counts, label='Incorrect',
                   color='#EA4335', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax2.set_title('Correct vs Incorrect Predictions', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # 3. Per-font accuracy heatmap (middle row, spans both columns)
    ax3 = fig.add_subplot(gs[1, :])

    # Get per-font accuracy for each model
    font_accuracies = defaultdict(lambda: {})
    all_fonts = set()

    for model_name in model_names:
        font_results = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in results[model_name]['results']:
            if r['prediction'] is not None:
                font = r['true_font']
                all_fonts.add(font)
                font_results[font]['total'] += 1
                if r.get('correct', False):
                    font_results[font]['correct'] += 1

        for font, counts in font_results.items():
            acc = (counts['correct'] / counts['total'] * 100) if counts['total'] > 0 else 0
            font_accuracies[font][model_name] = acc

    # Sort fonts by average accuracy (hardest first)
    fonts_sorted = sorted(all_fonts,
                         key=lambda f: np.mean([font_accuracies[f].get(m, 0) for m in model_names]))

    # Limit to top 20 hardest + top 20 easiest fonts (or all if fewer)
    if len(fonts_sorted) > 40:
        fonts_to_show = fonts_sorted[:20] + fonts_sorted[-20:]
    else:
        fonts_to_show = fonts_sorted

    # Create heatmap data
    heatmap_data = np.array([[font_accuracies[f].get(m, 0) for m in model_names]
                             for f in fonts_to_show])

    im = ax3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax3.set_xticks(np.arange(len(model_names)))
    ax3.set_yticks(np.arange(len(fonts_to_show)))
    ax3.set_xticklabels(model_names, fontsize=11)
    ax3.set_yticklabels(fonts_to_show, fontsize=8)
    ax3.set_title('Per-Font Accuracy (Hardest 20 + Easiest 20)', fontsize=14, fontweight='bold', pad=15)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=20, fontweight='bold')

    # 4. Top 10 Most Confused Fonts (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])

    # Aggregate mistakes across models
    mistake_counts = defaultdict(int)
    for model_name in model_names:
        for r in results[model_name]['results']:
            if not r.get('correct', True) and r['prediction'] is not None:
                mistake_counts[r['true_font']] += 1

    top_mistakes = sorted(mistake_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    if top_mistakes:
        fonts_mistake, counts_mistake = zip(*top_mistakes)
        y_pos = np.arange(len(fonts_mistake))
        ax4.barh(y_pos, counts_mistake, color='#EA4335', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(fonts_mistake, fontsize=9)
        ax4.invert_yaxis()
        ax4.set_xlabel('Number of Mistakes', fontsize=11, fontweight='bold')
        ax4.set_title('Top 10 Most Confused Fonts', fontsize=13, fontweight='bold', pad=15)
        ax4.grid(axis='x', alpha=0.3, linestyle='--')

    # 5. Sample Predictions (bottom right) - show some example mistakes
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    # Collect interesting mistakes (different predictions between models)
    example_text = "Example Mistakes:\n\n"
    examples_shown = 0

    for model_name in model_names:
        mistakes = [r for r in results[model_name]['results']
                   if not r.get('correct', True) and r['prediction'] is not None]

        for mistake in mistakes[:3]:  # Show up to 3 per model
            example_text += f"{model_name}:\n"
            example_text += f"  True: {mistake['true_font']}\n"
            example_text += f"  Predicted: {mistake['prediction']}\n\n"
            examples_shown += 1
            if examples_shown >= 8:
                break
        if examples_shown >= 8:
            break

    ax5.text(0.05, 0.95, example_text, transform=ax5.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Font Classification Benchmark Results',
                fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark vision models on font classification')
    parser.add_argument('--data-dir', default='data', help='Path to data directory')
    parser.add_argument('--num-samples', type=int, default=40, help='Number of samples to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--workers', type=int, default=16, help='Max parallel threads')
    parser.add_argument('--output', default='benchmark_results.json', help='Output JSON path')

    args = parser.parse_args()

    # Run benchmark
    results = benchmark_models(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        max_workers=args.workers
    )

    # Print summary
    print_summary(results)

    # Save results
    save_results(results, args.output)

    # Plot results
    plot_output = args.output.replace('.json', '.png')
    plot_results(results, plot_output)
