import os
from PIL import Image, ImageDraw, ImageFont
import os
import cv2


# Path to folders
normal_folder = '/workspace/prismer/datasets/cityscapes/leftImg8bit/val/lindau/'
depth_folder = '/workspace/prismer/test/output/depth/val/lindau'
normal_surface_folder = '/workspace/prismer/test/output/normal/val/lindau'

# Path to output folder
output_folder = '/workspace/prismer/test/output/experts_join/'

# Font and text size for watermark
# font = ImageFont.load_default().font_variant(size=30)
font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
font = ImageFont.truetype(font_path, size=64)
text_color = (255, 255, 255)  # white

# Iterate over images in normal folder
for filename in os.listdir(normal_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load normal image
        normal_image = Image.open(os.path.join(normal_folder, filename))
        
        # Check if depth image exists
        depth_filename = os.path.join(depth_folder, filename)
        depth_filename = os.path.splitext(depth_filename)[0] + '.png'  # Change extension to png
        if not os.path.exists(depth_filename):
            print(f"Depth image does not exist for {filename}. Skipping...")
            continue
        depth_image = Image.open(depth_filename)
        
        # Check if surface normal image exists
        normal_surface_filename = os.path.join(normal_surface_folder, filename)
        normal_surface_filename = os.path.splitext(normal_surface_filename)[0] + '.png'  # Change extension to png
        if not os.path.exists(normal_surface_filename):
            print(f"Surface normal image does not exist for {filename}. Skipping...")
            continue
        normal_surface_image = Image.open(normal_surface_filename)
        
        # Combine images horizontally
        combined_image = Image.new('RGB', (normal_image.width * 3, normal_image.height))
        combined_image.paste(normal_image, (0, 0))
        combined_image.paste(depth_image, (normal_image.width, 0))
        combined_image.paste(normal_surface_image, (normal_image.width * 2, 0))
        
        # Add watermark text
        draw = ImageDraw.Draw(combined_image)
        draw.text((10, 10), 'RGB', font=font, fill=text_color)
        draw.text((normal_image.width + 10, 10), 'Depth', font=font, fill=text_color)
        draw.text((normal_image.width * 2 + 10, 10), 'Surface', font=font, fill=text_color)
        
        # Save combined image
        output_filename = os.path.join(output_folder, filename)
        combined_image.save(output_filename)