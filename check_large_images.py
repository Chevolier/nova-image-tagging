#!/usr/bin/env python3
import os
from PIL import Image

img_dir = "imgs"
large_files = []
all_images = []

for filename in os.listdir(img_dir):
    filepath = os.path.join(img_dir, filename)
    if os.path.isfile(filepath) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        try:
            with Image.open(filepath) as img:
                width, height = img.size
                pixels = width * height
                all_images.append((filename, width, height, pixels, size_mb))
                
                if size_mb > 1:
                    large_files.append((filename, f"{size_mb:.2f}MB"))
        except:
            continue

# Sort by pixel count (dimensions)
top_dimensions = sorted(all_images, key=lambda x: x[3], reverse=True)[:20]

print(f"Found {len(large_files)} images larger than 1MB:")
for filename, size in large_files:
    print(f"{filename}: {size}")

print(f"\nTop 20 images by dimensions:")
for filename, width, height, pixels, size_mb in top_dimensions:
    print(f"{filename}: {width}x{height} ({pixels:,} pixels, {size_mb:.2f}MB)")
