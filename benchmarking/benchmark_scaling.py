#!/usr/bin/env python3
"""Benchmark how model accuracy scales with number of font classes"""

import random
import os
import base64
from pathlib import Path
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

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

def get_samples_for_fonts(data_dir, font_list, samples_per_font=1, seed=42):
    """Get samples for specific fonts"""
    random.seed(seed)

    data_path = Path(data_dir)
    samples = []

    for font in font_list:
        font_dir = data_path / font.replace(' ', '_')
        if not font_dir.exists():
            continue

        images = list(font_dir.glob("*.png"))
        if images:
            # Get samples_per_font random images
            selected_images = random.sample(images, min(samples_per_font, len(images)))
            for img_path in selected_images:
                samples.append({
                    'path': img_path,
                    'true_font': font,
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

def process_sample(args):
    """Worker function to process a single sample"""
    sample, model_id, system_prompt = args
    prediction = predict_font(sample['path'], model_id, system_prompt)

    result = {
        'true_font': sample['true_font'],
        'prediction': prediction,
    }

    return result

def calculate_accuracy(results):
    """Calculate accuracy metrics"""
    correct = 0
    total = 0

    for result in results:
        if result['prediction'] is not None:
            total += 1
            pred_norm = normalize_font_name(result['prediction'])
            true_norm = normalize_font_name(result['true_font'])
            if pred_norm == true_norm:
                correct += 1

    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, correct, total

def benchmark_at_scale(data_dir, num_classes, samples_per_font, model_name, model_id, max_workers, seed):
    """Benchmark a model at a specific number of classes"""

    # Select random fonts for this scale
    random.seed(seed)
    selected_fonts = random.sample(FONTS, min(num_classes, len(FONTS)))

    # Get samples for these fonts
    samples = get_samples_for_fonts(data_dir, selected_fonts, samples_per_font, seed)

    # Create system prompt with only these fonts
    system_prompt = create_system_prompt(selected_fonts)

    # Prepare tasks
    tasks = [(sample, model_id, system_prompt) for sample in samples]

    # Process samples in parallel
    model_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sample, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures),
                         desc=f"  {model_name} @ {num_classes} classes",
                         unit="sample", leave=False):
            result = future.result()
            model_results.append(result)

    # Calculate accuracy
    accuracy, correct, total = calculate_accuracy(model_results)

    return {
        'num_classes': num_classes,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }

def run_scaling_benchmark(data_dir="data", samples_per_font=3, max_workers=8, seed=42):
    """Run benchmark across different numbers of classes"""

    # Class counts to test
    class_counts = [2, 4, 8, 16]

    models = {
        "gemini-2.5-pro": "gemini/gemini-2.5-pro",
        "gpt-5": "gpt-5"
    }

    results = {model_name: [] for model_name in models.keys()}

    print(f"Running scaling benchmark with {samples_per_font} samples per font\n")

    for num_classes in class_counts:
        print(f"\n{'='*60}")
        print(f"Testing with {num_classes} classes")
        print('='*60)

        for model_name, model_id in models.items():
            result = benchmark_at_scale(
                data_dir=data_dir,
                num_classes=num_classes,
                samples_per_font=samples_per_font,
                model_name=model_name,
                model_id=model_id,
                max_workers=max_workers,
                seed=seed
            )

            results[model_name].append(result)
            print(f"  {model_name}: {result['correct']}/{result['total']} = {result['accuracy']:.1f}%")

    return results, class_counts

def plot_scaling_results(results, class_counts, output_path="benchmark_scaling.png"):
    """Plot scaling results"""

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        'gemini-2.5-pro': '#4285F4',
        'gpt-5': '#EA4335'
    }

    markers = {
        'gemini-2.5-pro': 'o',
        'gpt-5': 's'
    }

    for model_name, model_results in results.items():
        accuracies = [r['accuracy'] for r in model_results]

        ax.plot(class_counts, accuracies,
               marker=markers[model_name],
               linewidth=2.5,
               markersize=10,
               label=model_name,
               color=colors[model_name],
               alpha=0.8)

        # Add value labels on points
        for x, y in zip(class_counts, accuracies):
            ax.annotate(f'{y:.1f}%',
                       xy=(x, y),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=9,
                       fontweight='bold')

    ax.set_xlabel('Number of Font Classes', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Model Accuracy vs Number of Font Classes', fontsize=15, fontweight='bold', pad=20)

    # Set x-axis to log scale for better visualization
    ax.set_xscale('log')
    ax.set_xticks(class_counts)
    ax.set_xticklabels(class_counts)

    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, frameon=True, shadow=True, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nScaling plot saved to {output_path}")
    plt.close()

def save_scaling_results(results, output_path="benchmark_scaling_results.json"):
    """Save scaling results to JSON and CSV"""
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Scaling results saved to {output_path}")

    # Save CSV
    csv_path = output_path.replace('.json', '.csv')
    rows = []
    for model_name, model_results in results.items():
        for result in model_results:
            rows.append({
                'model': model_name,
                'num_classes': result['num_classes'],
                'accuracy': result['accuracy'],
                'correct': result['correct'],
                'total': result['total']
            })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Scaling results saved to {csv_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark model scaling with number of classes')
    parser.add_argument('--data-dir', default='data', help='Path to data directory')
    parser.add_argument('--samples-per-font', type=int, default=1, help='Samples per font')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--workers', type=int, default=32, help='Max parallel threads')
    parser.add_argument('--output', default='benchmark_scaling_results.json', help='Output JSON path')

    args = parser.parse_args()

    # Run scaling benchmark
    results, class_counts = run_scaling_benchmark(
        data_dir=args.data_dir,
        samples_per_font=args.samples_per_font,
        max_workers=args.workers,
        seed=args.seed
    )

    # Save results
    save_scaling_results(results, args.output)

    # Plot results
    plot_output = args.output.replace('.json', '.png')
    plot_scaling_results(results, class_counts, plot_output)

    print("\n" + "="*60)
    print("SCALING BENCHMARK COMPLETE")
    print("="*60)
