# Image Augmentation Script - README

## Overview

This script applies various image augmentations to images stored in a directory. It supports multiple augmentation techniques, including geometric transformations, color adjustments, noise addition, and environmental simulations.

## Requirements

Ensure you have the following dependencies installed before running the script:

```bash
pip install icrawler opencv-python numpy pillow tensorflow
```

## Features

The script provides the following augmentations:

- **Geometric Transformations:** Rotation, Resizing, Flipping, Random Cropping, Shear, Perspective
- **Color & Intensity Adjustments:** Brightness, Contrast, Saturation, Black & White, Hue Shift, Solarization, Posterization, Equalization
- **Noise Additions:** Gaussian Noise, Salt & Pepper Noise
- **Filtering Effects:** Edge Enhancement, Sharpening, Embossing, Median Blur
- **Occlusion Techniques:** Random Erase, Cutout
- **Weather Simulations:** Fog, Rain
- **Resolution Modifications:** JPEG Compression, Pixelation

## Usage Instructions

### 1. Download Images (Optional)

The script allows downloading images from Google using `icrawler`. To download images, initialize the crawler and specify your query:

```python
from icrawler.builtin import GoogleImageCrawler

crawler = GoogleImageCrawler(storage={'root_dir': 'images'})
crawler.crawl(keyword='cat', max_num=10)
```

### 2. Run the Augmentation Script

To apply augmentations, specify the image directory and desired augmentations in a dictionary format:

```python
import os

image_dir = 'images/'
save_dir = 'augmented_images/'

# Define augmentations to apply
augment_options = {
    "blur": True,
    "brightness": True,
    "rotation": True,
    "flip_horizontal": True,
    "contrast": True,
    "salt_pepper_noise": True
}

# Create output directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Apply augmentations
for img_file in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_file)
    apply_augmentation(img_path, save_dir, augment_options)
```

### 3. Processed Images

All augmented images will be saved in the `augmented_images/` directory with filenames indicating the applied augmentation.

### 4. Adding Custom Augmentations

To add new augmentations, modify the `augmentation_functions` dictionary inside `apply_augmentation()` and define a new function.

## Notes

- Some augmentations may not work on certain images due to format or color mode restrictions.
- The script automatically skips unreadable images.
- You can customize the augmentation intensity by modifying the random parameters inside each augmentation function.

## License

This script is licensed to Joel Santosh Pawar
