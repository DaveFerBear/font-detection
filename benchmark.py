#!/usr/bin/env python3
"""Benchmark frontier VLMs on font classification task"""

import random
import os
import base64
from pathlib import Path
from collections import Counter
import json

from litellm import completion
from fonts import FONTS

from dotenv import load_dotenv
load_dotenv()

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

Analyze the image carefully and respond with ONLY the exact font name from the list above. Do not include any explanation, reasoning, or additional text - just the font name."""

    return prompt

def predict_font(image_path, model, system_prompt):
    """Predict font using vision model"""

    # Encode image
    image_b64 = encode_image_base64(image_path)

    # Combine system prompt and question into user message
    # (some models don't support system role with vision)
    user_text = f"{system_prompt}\n\nWhat font is used in this image?"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                },
                {
                    "type": "text",
                    "text": user_text
                }
            ]
        }
    ]

    try:
        response = completion(
            model=model,
            messages=messages,
            max_tokens=50,
            temperature=0,
            drop_params=True
        )

        content = response.choices[0].message.content
        if content is None:
            print(f"  Warning: Model returned None content")
            print(f"  Response object: {response}")
            return None

        prediction = content.strip()
        return prediction

    except Exception as e:
        print(f"  Error with {model}: {e}")
        import traceback
        traceback.print_exc()
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

def benchmark_models(data_dir="data", num_samples=10, seed=42):
    """Benchmark both models"""

    print(f"Loading {num_samples} random samples from {data_dir}...")
    samples = get_random_samples(data_dir, num_samples, seed)
    print(f"Selected {len(samples)} samples")

    # Create system prompt with all fonts
    system_prompt = create_system_prompt(FONTS)
    print(f"\nSystem prompt created with {len(FONTS)} fonts")

    models = {
        "gemini-2.0-flash-exp": "gemini/gemini-2.0-flash-exp",
        "gpt-4o": "gpt-4o"
    }

    results = {}

    for model_name, model_id in models.items():
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print('='*60)

        model_results = []

        for i, sample in enumerate(samples, 1):
            print(f"\n[{i}/{len(samples)}] Testing {sample['true_font']}...")
            print(f"  Image: {sample['path'].name}")

            prediction = predict_font(sample['path'], model_id, system_prompt)

            result = {
                'image': str(sample['path']),
                'true_font': sample['true_font'],
                'prediction': prediction,
            }

            model_results.append(result)

            print(f"  Predicted: {prediction}")
            print(f"  Actual: {sample['true_font']}")

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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark vision models on font classification')
    parser.add_argument('--data-dir', default='data', help='Path to data directory')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', default='benchmark_results.json', help='Output JSON path')

    args = parser.parse_args()

    # Run benchmark
    results = benchmark_models(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        seed=args.seed
    )

    # Print summary
    print_summary(results)

    # Save results
    save_results(results, args.output)
