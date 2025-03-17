from icrawler.builtin import GoogleImageCrawler
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import random
import math
from tensorflow.keras.preprocessing.image import random_rotation, random_shift, random_zoom, random_shear

# Function to apply selected augmentations
def apply_augmentation(image_path, save_dir, augment_options):
    # Load image using OpenCV (cv2)
    image = cv2.imread(image_path)

    # 🛑 Skip if image is not loaded properly
    if image is None:
        print(f"⚠️ Skipping {image_path} (Failed to read)")
        return

    img_name = os.path.basename(image_path)
    
    # Convert OpenCV image to PIL (for brightness & saturation adjustments)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Convert to numpy array for Keras augmentations
    image_np = np.array(image_pil)

    # Dictionary of augmentation functions
    augmentation_functions = {
        # Original augmentations
        "blur": lambda img: cv2.GaussianBlur(img, (18, 18), 0),  # Apply Gaussian Blur
        "saturation": lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.5, 2.5)),  # Adjust Saturation
        "brightness": lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(1.5, 4.5)),  # Adjust Brightness
        "rotation": lambda img: img.rotate(random.choice([90, 180, 270])),  # Random Rotation
        "resizing": lambda img: img.resize((random.randint(50, 300), random.randint(50, 300))),  # Random Resize
        "black_white": lambda img: img.convert("L"),  # Convert to Black & White
        
        # Additional geometric transformations
        "flip_horizontal": lambda img: ImageOps.mirror(img),  # Horizontal flip
        "flip_vertical": lambda img: ImageOps.flip(img),  # Vertical flip
        "random_crop": lambda img: img.crop((
            random.randint(0, img.width // 4),
            random.randint(0, img.height // 4),
            random.randint(img.width * 3 // 4, img.width),
            random.randint(img.height * 3 // 4, img.height)
        )),
        "shear": lambda img: np.array(img).astype(np.float32) if isinstance(img, Image.Image) else img.astype(np.float32),  # Prepare for shear via Keras
        "perspective": lambda img: img.transform(
            img.size, 
            Image.PERSPECTIVE, 
            (
                random.uniform(0, 0.1), random.uniform(0, 0.1),
                random.uniform(0, 0.1), random.uniform(0.9, 1.0),
                random.uniform(0.9, 1.0), random.uniform(0, 0.1),
                random.uniform(0.9, 1.0), random.uniform(0.9, 1.0)
            ),
            resample=Image.BICUBIC
        ),
        
        # Additional color/intensity transformations
        "contrast": lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 2.0)),  # Adjust Contrast
        "hue": lambda img: img.convert("HSV").point(lambda i: (i + random.randint(-20, 20)) % 256 if i <= 255 else i),  # Hue shift
        "solarize": lambda img: ImageOps.solarize(img, threshold=random.randint(0, 255)),  # Solarize effect
        "posterize": lambda img: ImageOps.posterize(img, bits=random.randint(1, 7)),  # Posterize effect
        "equalize": lambda img: ImageOps.equalize(img),  # Histogram equalization
        
        # Noise additions
        "gaussian_noise": lambda img: cv2.add(
            cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) if isinstance(img, Image.Image) else img,
            np.random.normal(0, random.uniform(5, 25), img.shape if isinstance(img, np.ndarray) else np.array(img).shape).astype(np.uint8)
        ),
        "salt_pepper_noise": lambda img: add_salt_pepper_noise(
            cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) if isinstance(img, Image.Image) else img
        ),
        
        # Filtering operations
        "edge_enhance": lambda img: img.filter(ImageFilter.EDGE_ENHANCE),  # Edge enhancement
        "sharpen": lambda img: img.filter(ImageFilter.SHARPEN),  # Sharpen
        "emboss": lambda img: img.filter(ImageFilter.EMBOSS),  # Emboss
        "median_blur": lambda img: cv2.medianBlur(
            cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) if isinstance(img, Image.Image) else img, 
            ksize=random.choice([3, 5, 7])
        ),
        
        # Occlusion techniques
        "random_erase": lambda img: random_erase(np.array(img) if isinstance(img, Image.Image) else img),
        "cutout": lambda img: cutout(np.array(img) if isinstance(img, Image.Image) else img),
        
        # Weather/environmental simulations
        "fog": lambda img: add_fog(np.array(img) if isinstance(img, Image.Image) else img),
        "rain": lambda img: add_rain(np.array(img) if isinstance(img, Image.Image) else img),
        
        # Resolution modifications
        "jpeg_compression": lambda img: compress_jpeg(img),
        "pixelate": lambda img: img.resize(
            (img.width // random.randint(5, 10), img.height // random.randint(5, 10)), 
            Image.NEAREST
        ).resize((img.width, img.height), Image.NEAREST),
    }

    # Apply only selected augmentations
    for aug, enabled in augment_options.items():
        if enabled:  # Check if user enabled the augmentation
            try:
                func = augmentation_functions[aug]  # Get the function
                
                # Handle special cases
                if aug == "blur" or aug == "gaussian_noise" or aug == "salt_pepper_noise" or aug == "median_blur":
                    aug_img = func(image)  # Apply OpenCV augmentation
                    aug_save_path = os.path.join(save_dir, f"{aug}_{img_name}")
                    cv2.imwrite(aug_save_path, aug_img)
                
                elif aug == "random_erase" or aug == "cutout" or aug == "fog" or aug == "rain":
                    aug_img = func(image)  # Apply NumPy augmentation
                    aug_save_path = os.path.join(save_dir, f"{aug}_{img_name}")
                    cv2.imwrite(aug_save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR) if aug_img.shape[-1] == 3 else aug_img)
                
                elif aug == "shear":
                    # Prepare image for shear with Keras
                    img_array = func(image_pil)
                    img_array = img_array.reshape((1,) + img_array.shape)
                    # Apply shear using Keras
                    aug_img = random_shear(img_array, intensity=random.uniform(0.1, 0.3), row_axis=0, col_axis=1, channel_axis=2)[0]
                    aug_save_path = os.path.join(save_dir, f"{aug}_{img_name}")
                    cv2.imwrite(aug_save_path, cv2.cvtColor(aug_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
                
                else:
                    aug_img = func(image_pil)  # Apply PIL augmentation
                    aug_save_path = os.path.join(save_dir, f"{aug}_{img_name}")
                    
                    # Handle saving for different image types
                    if aug_img.mode == "L":  # Black and white images
                        aug_img.save(aug_save_path)
                    else:
                        aug_img.convert("RGB").save(aug_save_path)

                print(f"✅ Applied {aug} to {img_name}")

            except Exception as e:
                print(f"⚠️ Error applying {aug} to {img_name}: {e}")

# Helper functions for advanced augmentations
def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = np.copy(image)
    # Add salt noise
    salt_mask = np.random.random(image.shape[:2]) < salt_prob
    noisy_image[salt_mask] = 255
    # Add pepper noise
    pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
    noisy_image[pepper_mask] = 0
    return noisy_image

def random_erase(image, erase_ratio=0.2):
    img_h, img_w = image.shape[:2]
    area = img_h * img_w
    
    target_area = random.uniform(0.02, erase_ratio) * area
    aspect_ratio = random.uniform(0.3, 1/0.3)
    
    h = int(round(math.sqrt(target_area * aspect_ratio)))
    w = int(round(math.sqrt(target_area / aspect_ratio)))
    
    if h < img_h and w < img_w:
        x1 = random.randint(0, img_w - w)
        y1 = random.randint(0, img_h - h)
        
        if len(image.shape) == 3:
            image[y1:y1+h, x1:x1+w, :] = random.randint(0, 255)
        else:
            image[y1:y1+h, x1:x1+w] = random.randint(0, 255)
    
    return image

def cutout(image, n_holes=1, length=50):
    h, w = image.shape[:2]
    mask = np.ones((h, w), np.uint8)
    
    for _ in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        mask[y1:y2, x1:x2] = 0
    
    if len(image.shape) == 3:
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, 3, axis=2)
    
    masked_img = image * mask
    return masked_img

def add_fog(image, fog_coeff=0.3):
    fog = np.zeros_like(image, dtype=np.uint8)
    fog[:] = 255
    
    if len(image.shape) == 3:
        fog_applied = cv2.addWeighted(image, 1 - fog_coeff, fog, fog_coeff, 0)
    else:
        fog_applied = cv2.addWeighted(image, 1 - fog_coeff, fog, fog_coeff, 0)
    
    return fog_applied

def add_rain(image, rain_drops=500, slant=20, drop_length=20, drop_width=2, drop_color=(200, 200, 200)):
    imshape = image.copy()
    image_rainy = image.copy()
    
    for i in range(rain_drops):
        x = np.random.randint(0, imshape.shape[1]-slant)
        y = np.random.randint(0, imshape.shape[0]-drop_length)
        
        for j in range(drop_length):
            x_shift = int(j * slant / drop_length)
            if x + x_shift < imshape.shape[1] and y + j < imshape.shape[0]:
                if len(imshape.shape) == 3:
                    image_rainy[y+j, x+x_shift, :] = drop_color
                else:
                    image_rainy[y+j, x+x_shift] = drop_color[0]
    
    return image_rainy

def compress_jpeg(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Create a BytesIO object to hold the compressed image
    import io
    buffer = io.BytesIO()
    
    # Save image with JPEG compression (quality 10-30)
    quality = random.randint(10, 30)
    image.save(buffer, format="JPEG", quality=quality)
    
    # Load the compressed image back
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    
    return compressed_image

# Function to download images
def download_images(search_term, num_images, folder_name, augment_options):
    save_dir = os.path.join(folder_name, search_term.replace(" ", "_"))
    os.makedirs(save_dir, exist_ok=True)

    crawler = GoogleImageCrawler(storage={"root_dir": save_dir})
    crawler.crawl(keyword=search_term, max_num=num_images)

    print(f"✅ Downloaded {num_images} images of '{search_term}' in '{save_dir}'.")

    # Apply augmentations
    for img_file in os.listdir(save_dir):
        img_path = os.path.join(save_dir, img_file)
        apply_augmentation(img_path, save_dir, augment_options)

# Main function
def main():
    folder_name = input("📂 Enter the name for the main download folder: ")
    os.makedirs(folder_name, exist_ok=True)

    search_term = input("📸 Enter the image name you want: ")
    num_images = int(input(f"🔢 How many images of '{search_term}'? "))

    # Full list of augmentation options
    augment_options = {
        # Original augmentations
        "blur": input("🔄 Augmentation Blur (Y/N): ").strip().lower() == "y",
        "saturation": input("🎨 Augmentation Saturation (Y/N): ").strip().lower() == "y",
        "brightness": input("💡 Augmentation Brightness (Y/N): ").strip().lower() == "y",
        "rotation": input("🔄 Augmentation Rotation (Y/N): ").strip().lower() == "y",
        "resizing": input("📏 Augmentation Resizing (Y/N): ").strip().lower() == "y",
        "black_white": input("⚫⚪ Augmentation Black & White (Y/N): ").strip().lower() == "y",
        
        # Geometric transformations
        "flip_horizontal": input("↔️ Augmentation Horizontal Flip (Y/N): ").strip().lower() == "y",
        "flip_vertical": input("↕️ Augmentation Vertical Flip (Y/N): ").strip().lower() == "y",
        "random_crop": input("✂️ Augmentation Random Crop (Y/N): ").strip().lower() == "y",
        "shear": input("↗️ Augmentation Shear (Y/N): ").strip().lower() == "y",
        "perspective": input("🔲 Augmentation Perspective Transform (Y/N): ").strip().lower() == "y",
        
        # Color transformations
        "contrast": input("⚖️ Augmentation Contrast (Y/N): ").strip().lower() == "y",
        "hue": input("🌈 Augmentation Hue Shift (Y/N): ").strip().lower() == "y",
        "solarize": input("🌞 Augmentation Solarize (Y/N): ").strip().lower() == "y",
        "posterize": input("🖼️ Augmentation Posterize (Y/N): ").strip().lower() == "y",
        "equalize": input("📊 Augmentation Histogram Equalization (Y/N): ").strip().lower() == "y",
        
        # Noise additions
        "gaussian_noise": input("🔍 Augmentation Gaussian Noise (Y/N): ").strip().lower() == "y",
        "salt_pepper_noise": input("🧂 Augmentation Salt & Pepper Noise (Y/N): ").strip().lower() == "y",
        
        # Filtering operations
        "edge_enhance": input("📏 Augmentation Edge Enhancement (Y/N): ").strip().lower() == "y",
        "sharpen": input("✨ Augmentation Sharpen (Y/N): ").strip().lower() == "y",
        "emboss": input("🔖 Augmentation Emboss (Y/N): ").strip().lower() == "y",
        "median_blur": input("🌫️ Augmentation Median Blur (Y/N): ").strip().lower() == "y",
        
        # Occlusion techniques
        "random_erase": input("❌ Augmentation Random Erase (Y/N): ").strip().lower() == "y",
        "cutout": input("✂️ Augmentation Cutout (Y/N): ").strip().lower() == "y",
        
        # Weather/environmental simulations
        "fog": input("🌫️ Augmentation Fog Effect (Y/N): ").strip().lower() == "y",
        "rain": input("🌧️ Augmentation Rain Effect (Y/N): ").strip().lower() == "y",
        
        # Resolution modifications
        "jpeg_compression": input("📉 Augmentation JPEG Compression (Y/N): ").strip().lower() == "y",
        "pixelate": input("🔍 Augmentation Pixelate (Y/N): ").strip().lower() == "y",
    }

    download_images(search_term, num_images, folder_name, augment_options)

if __name__ == "__main__":
    main()
        