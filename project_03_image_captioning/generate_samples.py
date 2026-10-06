"""
Utility script to generate benchmark sample images for testing the Image Captioning pipeline.
Creates sample images with shapes and annotations representing classic benchmark scenes:
1. dog_ball_park.jpg -> "A dog playing with a ball in a park"
2. cat_on_sofa.jpg   -> "A cat sleeping on a comfortable sofa"
3. beach_sunset.jpg  -> "A beautiful sunset over the ocean beach"
"""

import os
from PIL import Image, ImageDraw, ImageFont

def generate_sample_images(output_dir="sample_images"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Dog playing with a ball in a park
    img1 = Image.new("RGB", (400, 300), color=(100, 190, 90)) # Green park grass background
    draw1 = ImageDraw.Draw(img1)
    # Sky
    draw1.rectangle([0, 0, 400, 100], fill=(135, 206, 235))
    # Sun
    draw1.ellipse([320, 20, 370, 70], fill=(255, 220, 0))
    # Trees
    draw1.rectangle([40, 50, 60, 120], fill=(100, 60, 30))
    draw1.ellipse([20, 20, 80, 80], fill=(30, 120, 40))
    # Dog (brown body, head, legs, tail)
    draw1.ellipse([140, 170, 220, 220], fill=(160, 82, 45)) # Body
    draw1.ellipse([200, 140, 245, 185], fill=(160, 82, 45)) # Head
    draw1.ellipse([235, 155, 250, 170], fill=(0, 0, 0))    # Nose
    draw1.rectangle([150, 210, 165, 250], fill=(130, 65, 30)) # Legs
    draw1.rectangle([190, 210, 205, 250], fill=(130, 65, 30))
    # Red ball
    draw1.ellipse([260, 200, 290, 230], fill=(230, 40, 40))
    
    img1.save(os.path.join(output_dir, "dog_ball_park.jpg"))

    # 2. Cat sleeping on sofa
    img2 = Image.new("RGB", (400, 300), color=(220, 210, 200)) # Warm living room wall
    draw2 = ImageDraw.Draw(img2)
    # Sofa (blue cushion)
    draw2.rectangle([50, 140, 350, 260], fill=(70, 130, 180))
    draw2.rectangle([30, 100, 70, 260], fill=(50, 100, 150))  # Armrest left
    draw2.rectangle([330, 100, 370, 260], fill=(50, 100, 150)) # Armrest right
    # Cat (orange sleeping circle on sofa)
    draw2.ellipse([160, 160, 240, 210], fill=(245, 140, 50))
    draw2.ellipse([145, 170, 175, 200], fill=(245, 140, 50)) # Head
    img2.save(os.path.join(output_dir, "cat_on_sofa.jpg"))

    # 3. Sunset over ocean beach
    img3 = Image.new("RGB", (400, 300), color=(255, 120, 80)) # Orange sunset sky
    draw3 = ImageDraw.Draw(img3)
    # Setting Sun
    draw3.ellipse([160, 80, 240, 160], fill=(255, 230, 100))
    # Sea Water
    draw3.rectangle([0, 150, 400, 240], fill=(30, 90, 160))
    # Sandy Beach
    draw3.rectangle([0, 240, 400, 300], fill=(238, 214, 175))
    img3.save(os.path.join(output_dir, "beach_sunset.jpg"))

    print(f"Sample benchmark images successfully generated in '{output_dir}/'!")

if __name__ == "__main__":
    generate_sample_images()
