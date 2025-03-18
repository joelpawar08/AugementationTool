import streamlit as st
import os
import random
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import cv2
from tensorflow.keras.preprocessing.image import random_rotation, random_shift, random_zoom, random_shear
from icrawler.builtin import GoogleImageCrawler

# Function to apply selected augmentations
def apply_augmentation(image_path, save_dir, augment_options):
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Skipping {image_path} (Failed to read)")
        return

    img_name = os.path.basename(image_path)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_np = np.array(image_pil)

    # Dictionary of augmentation functions
    augmentation_functions = {
        "blur": lambda img: cv2.GaussianBlur(img, (35, 35), 0),
        "saturation": lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.5, 2.5)),
        "brightness": lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(1.5, 4.5)),
        "rotation": lambda img: img.rotate(random.choice([90, 180, 270])),
        "resizing": lambda img: img.resize((random.randint(50, 300), random.randint(50, 300))),
        "black_white": lambda img: img.convert("L"),
        "flip_horizontal": lambda img: ImageOps.mirror(img),
        "flip_vertical": lambda img: ImageOps.flip(img),
        "random_crop": lambda img: img.crop((
            random.randint(0, img.width // 4),
            random.randint(0, img.height // 4),
            random.randint(img.width * 3 // 4, img.width),
            random.randint(img.height * 3 // 4, img.height)
        )),
        "shear": lambda img: np.array(img).astype(np.float32),
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
        "contrast": lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 2.0)),
        "hue": lambda img: img.convert("HSV").point(lambda i: (i + random.randint(-20, 20)) % 256 if i <= 255 else i),
        "solarize": lambda img: ImageOps.solarize(img, threshold=random.randint(0, 255)),
        "posterize": lambda img: ImageOps.posterize(img, bits=random.randint(1, 7)),
        "equalize": lambda img: ImageOps.equalize(img),
        "gaussian_noise": lambda img: cv2.add(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), np.random.normal(0, random.uniform(15, 35), img.shape).astype(np.uint8)),
        "salt_pepper_noise": lambda img: add_salt_pepper_noise(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)),
        "edge_enhance": lambda img: img.filter(ImageFilter.EDGE_ENHANCE),
        "sharpen": lambda img: img.filter(ImageFilter.SHARPEN),
        "emboss": lambda img: img.filter(ImageFilter.EMBOSS),
        "median_blur": lambda img: cv2.medianBlur(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), ksize=random.choice([9, 14, 18])),
        "random_erase": lambda img: random_erase(np.array(img)),
        "cutout": lambda img: cutout(np.array(img)),
        "fog": lambda img: add_fog(np.array(img)),
        "rain": lambda img: add_rain(np.array(img)),
        "jpeg_compression": lambda img: compress_jpeg(img),
        "pixelate": lambda img: img.resize(
            (img.width // random.randint(5, 10), img.height // random.randint(5, 10)),
            Image.NEAREST
        ).resize((img.width, img.height), Image.NEAREST),
    }

    # Apply selected augmentations
    for aug, enabled in augment_options.items():
        if enabled:
            try:
                func = augmentation_functions[aug]
                aug_img = func(image_pil) if aug != "gaussian_noise" and aug != "salt_pepper_noise" else func(image)
                aug_save_path = os.path.join(save_dir, f"{aug}_{img_name}")
                if aug_img.mode == "L":
                    aug_img.save(aug_save_path)
                else:
                    aug_img.convert("RGB").save(aug_save_path)
                print(f"✅ Applied {aug} to {img_name}")
            except Exception as e:
                print(f"⚠️ Error applying {aug} to {img_name}: {e}")

# Helper functions
def add_salt_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    noisy_image = np.copy(image)
    salt_mask = np.random.random(image.shape[:2]) < salt_prob
    noisy_image[salt_mask] = 255
    pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
    noisy_image[pepper_mask] = 0
    return noisy_image

def random_erase(image, erase_ratio=0.2):
    img_h, img_w = image.shape[:2]
    target_area = random.uniform(0.02, erase_ratio) * img_h * img_w
    aspect_ratio = random.uniform(0.3, 1/0.3)
    h = int(round(math.sqrt(target_area * aspect_ratio)))
    w = int(round(math.sqrt(target_area / aspect_ratio)))
    if h < img_h and w < img_w:
        x1 = random.randint(0, img_w - w)
        y1 = random.randint(0, img_h - h)
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
    return image * np.expand_dims(mask, axis=2)

def add_fog(image, fog_coeff=0.3):
    fog = np.zeros_like(image, dtype=np.uint8)
    fog[:] = 255
    return cv2.addWeighted(image, 1 - fog_coeff, fog, fog_coeff, 0)

def add_rain(image, rain_drops=500, slant=20, drop_length=20, drop_width=2, drop_color=(200, 200, 200)):
    for _ in range(rain_drops):
        x = np.random.randint(0, image.shape[1] - slant)
        y = np.random.randint(0, image.shape[0] - drop_length)
        for j in range(drop_length):
            x_shift = int(j * slant / drop_length)
            if x + x_shift < image.shape[1] and y + j < image.shape[0]:
                image[y + j, x + x_shift, :] = drop_color
    return image

def compress_jpeg(image):
    buffer = io.BytesIO()
    quality = random.randint(10, 30)
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

# Image downloader function
def download_images(search_term, num_images, folder_name, augment_options):
    save_dir = os.path.join(folder_name, search_term.replace(" ", "_"))
    os.makedirs(save_dir, exist_ok=True)
    crawler = GoogleImageCrawler(storage={"root_dir": save_dir})
    crawler.crawl(keyword=search_term, max_num=num_images)
    for img_file in os.listdir(save_dir):
        img_path = os.path.join(save_dir, img_file)
        apply_augmentation(img_path, save_dir, augment_options)

# Streamlit UI
def main():
    st.set_page_config(layout="wide")
    st.title('Image Augmentation with Streamlit')
    folder_name = st.text_input("📂 Folder Name", "Images")
    search_term = st.text_input("📸 Image Search Term", "cat")
    num_images = st.number_input("🔢 Number of Images", min_value=1, max_value=100, value=10)

    st.write("### Augmentation Options")
    col1, col2, col3, col4, col5 = st.columns(5)
    augment_options = {
        "blur": col1.checkbox("Blur", value=True),
        "saturation": col2.checkbox("Saturation", value=True),
        "brightness": col3.checkbox("Brightness", value=True),
        "rotation": col4.checkbox("Rotation", value=True),
        "resizing": col5.checkbox("Resizing", value=True),
        "black_white": col1.checkbox("Black & White", value=True),
        "flip_horizontal": col2.checkbox("Horizontal Flip", value=True),
        "flip_vertical": col3.checkbox("Vertical Flip", value=True),
        "random_crop": col4.checkbox("Random Crop", value=True),
        "shear": col5.checkbox("Shear", value=True),
        "perspective": col1.checkbox("Perspective", value=True),
        "contrast": col2.checkbox("Contrast", value=True),
        "hue": col3.checkbox("Hue", value=True),
        "solarize": col4.checkbox("Solarize", value=True),
        "posterize": col5.checkbox("Posterize", value=True),
        "equalize": col1.checkbox("Equalize", value=True),
        "gaussian_noise": col2.checkbox("Gaussian Noise", value=True),
        "salt_pepper_noise": col3.checkbox("Salt & Pepper Noise", value=True),
        "edge_enhance": col4.checkbox("Edge Enhance", value=True),
        "sharpen": col5.checkbox("Sharpen", value=True),
        "emboss": col1.checkbox("Emboss", value=True),
        "median_blur": col2.checkbox("Median Blur", value=True),
        "random_erase": col3.checkbox("Random Erase", value=True),
        "cutout": col4.checkbox("Cutout", value=True),
        "fog": col5.checkbox("Fog", value=True),
        "rain": col1.checkbox("Rain", value=True),
        "sunflare": col2.checkbox("Sunflare", value=True),
        "snow": col3.checkbox("Snow", value=True),
        "jpeg_compression": col4.checkbox("JPEG Compression", value=True),
        "pixelate": col5.checkbox("Pixelate", value=True),
        "mixed": col1.checkbox("Mixed Augmentation", value=True)
    }

    if st.button("Download and Apply Augmentations"):
        download_images(search_term, num_images, folder_name, augment_options)
        st.success("✅ Images Downloaded and Augmentations Applied!")

if __name__ == "__main__":
    main()
