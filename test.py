import streamlit as st
import os
from icrawler.builtin import GoogleImageCrawler
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
        "blur": lambda img: cv2.GaussianBlur(img, (35, 35), 0),  # Apply Gaussian Blur
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
            np.random.normal(0, random.uniform(15, 35), img.shape if isinstance(img, np.ndarray) else np.array(img).shape).astype(np.uint8)
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
            ksize=random.choice([9, 14, 18])
        ),
        
        # Occlusion techniques
        "random_erase": lambda img: random_erase(np.array(img) if isinstance(img, Image.Image) else img),
        "cutout": lambda img: cutout(np.array(img) if isinstance(img, Image.Image) else img),
        
        # Weather/environmental simulations
        "fog": lambda img: add_fog(np.array(img) if isinstance(img, Image.Image) else img),
        "rain": lambda img: add_rain(np.array(img) if isinstance(img, Image.Image) else img),
        #
        
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
def add_salt_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
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

def add_sunflare(image):
    # Simulate a sunflare effect by adding random spots of brightness
    flare_intensity = random.uniform(0.5, 1.5)
    width, height = image.size
    img_array = np.array(image)
    
    for _ in range(random.randint(3, 5)):
        x = random.randint(0, width)
        y = random.randint(0, height)
        radius = random.randint(20, 50)
        
        for i in range(x-radius, x+radius):
            for j in range(y-radius, y+radius):
                if 0 <= i < width and 0 <= j < height:
                    dist = np.sqrt((i-x)**2 + (j-y)**2)
                    if dist < radius:
                        img_array[j, i] = np.clip(img_array[j, i] + random.randint(0, int(flare_intensity * 255)), 0, 255)
    
    return Image.fromarray(img_array)

def add_snow(image):
    # Simulate snow by adding random white spots
    width, height = image.size
    img_array = np.array(image)
    
    for _ in range(random.randint(1000, 3000)):
        x = random.randint(0, width)
        y = random.randint(0, height)
        img_array[y, x] = [255, 255, 255]
    
    return Image.fromarray(img_array)

def apply_mixed_augmentation(img):
    # Apply all augmentations in sequence for a mixed augmentation
    augmentations = [
        "blur", "saturation", "brightness", "rotation", "resizing", 
        "black_white", "flip_horizontal", "flip_vertical", 
        "random_crop", "shear", "perspective", "contrast", 
        "hue", "solarize", "posterize", "equalize", 
        "gaussian_noise", "salt_pepper_noise", "edge_enhance", "sharpen", 
        "emboss", "median_blur", "random_erase", "cutout", 
        "fog", "rain", "sunflare", "snow", "jpeg_compression", "pixelate"
    ]
    for aug in augmentations:
        img = apply_augmentation_single(img, aug)
    return img

def apply_augmentation_single(img, aug):
    # Apply the individual augmentation to an image
    # This method would simply call the corresponding augmentation function
    pass


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


# Streamlit UI
def main():
    st.set_page_config(layout="wide")
    st.title('Image Augmentation with Streamlit')

    folder_name = st.text_input("📂 Enter the name for the main download folder:", "Images")
    search_term = st.text_input("📸 Enter the image name you want:", "cat")
    num_images = st.number_input(f"🔢 How many images of '{search_term}'?", min_value=1, max_value=100, value=10)

    st.write("### Augmentation Options")

    # Layout for checkboxes in rows of 5 columns
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

        "perspective": col1.checkbox("Perspective Transform", value=True),
        "contrast": col2.checkbox("Contrast", value=True),
        "hue": col3.checkbox("Hue Shift", value=True),
        "solarize": col4.checkbox("Solarize", value=True),
        "posterize": col5.checkbox("Posterize", value=True),

        "equalize": col1.checkbox("Histogram Equalization", value=True),
        "gaussian_noise": col2.checkbox("Gaussian Noise", value=True),
        "salt_pepper_noise": col3.checkbox("Salt & Pepper Noise", value=True),
        "edge_enhance": col4.checkbox("Edge Enhancement", value=True),
        "sharpen": col5.checkbox("Sharpen", value=True),

        "emboss": col1.checkbox("Emboss", value=True),
        "median_blur": col2.checkbox("Median Blur", value=True),
        "random_erase": col3.checkbox("Random Erase", value=True),
        "cutout": col4.checkbox("Cutout", value=True),
        "fog": col5.checkbox("Fog Effect", value=True),

        "rain": col1.checkbox("Rain Effect", value=True),
        "sunflare": col2.checkbox("Sunflare Effect", value=True),
        "snow": col3.checkbox("Snow Effect", value=True),
        "jpeg_compression": col4.checkbox("JPEG Compression", value=True),
        "pixelate": col5.checkbox("Pixelate", value=True),

        "mixed": col1.checkbox("Mixed Augmentation", value=True)
    }

    if st.button("Download and Apply Augmentations"):
        download_images(search_term, num_images, folder_name, augment_options)
        st.success("✅ Images Downloaded and Augmentations Applied!")


if __name__ == "__main__":
    main()
