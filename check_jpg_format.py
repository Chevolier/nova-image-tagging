#!/usr/bin/env python3
import os
from PIL import Image

def check_jpg_format_mismatch(imgs_dir="imgs"):
    mismatched = []
    
    for filename in os.listdir(imgs_dir):
        if filename.lower().endswith('.jpg'):
            filepath = os.path.join(imgs_dir, filename)
            try:
                with Image.open(filepath) as img:
                    if img.format != 'JPEG':
                        mismatched.append((filename, img.format))
            except Exception as e:
                mismatched.append((filename, f"Error: {e}"))
    
    return mismatched

if __name__ == "__main__":
    mismatched_files = check_jpg_format_mismatch()
    
    if mismatched_files:
        print(f"Found {len(mismatched_files)} files with .jpg extension but different format:")
        for filename, actual_format in mismatched_files:
            print(f"  {filename} -> {actual_format}")
    else:
        print("All .jpg files have correct JPEG format")
