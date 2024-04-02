import os
from PIL import Image
import torch
from utils.custom_transforms import Cutout
import torchvision.transforms as transforms

def apply_cutout(img_path, cutout_transform, output_folder): # ./WIDER_train/0--Parade/0_Parade_ ... .jpg, cutout, ./AFTER_wider/0--Parade
    
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return
    
    print(img)
    img_transformed = cutout_transform(img=img)
    
    # Extract the relative path from WIDER_train folder
    relative_path = os.path.relpath(img_path, "WIDER_train") # 0--Parade/0_Parade_ ... .jpg
    desired_part = relative_path.split('/')[1]
    # Construct the path in the AFTER_wider folder
    output_path = os.path.join(output_folder, desired_part) # ./AFTER_wider/0--Parade/0_Parade_ ... .jpg

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    topilimage = transforms.ToPILImage()
    img_pil = topilimage(img_transformed)
    img_pil.save(output_path)

def process_folder(folder_path, cutout_transform, output_folder): # ./WIDER_train/0--Parade, cutout, ./AFTER_wider/0--Parade
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file) # ./WIDER_train/0--Parade/0_Parade_ ... .jpg
                apply_cutout(file_path, cutout_transform, output_folder)

def augmentation(before, gt):
    cutout_transform = Cutout(n_holes=1, probability=1.0)

    # Check if gt folder exists, if not create it
    if not os.path.exists(gt):
        os.makedirs(gt)

    # Loop through all subdirectories in before folder
    for folder in os.listdir(before):
        folder_path_before = os.path.join(before, folder) # ./WIDER_train/0--Parade
        folder_path_gt = os.path.join(gt, folder) # ./AFTER_wider/0--Parade

        # Check if it's a directory
        if os.path.isdir(folder_path_before):
            # Create corresponding directory in gt folder
            if not os.path.exists(folder_path_gt):
                os.makedirs(folder_path_gt)
            # Process all images in the current subdirectory
            process_folder(folder_path_before, cutout_transform, folder_path_gt)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--before', default="./WIDER_train/")  # /home/ijieun/yolov5-face/widerface_evaluate/widerface_txt
    parser.add_argument('-g', '--gt', default='./AFTER_h_4/')
    args = parser.parse_args()

    augmentation(args.before, args.gt)
