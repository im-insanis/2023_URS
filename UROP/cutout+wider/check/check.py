from utils.custom_transforms import Cutout
from PIL import Image
import torchvision.transforms as transforms
import torch

# 이미지를 PIL Image 객체로 열기

cutout_transform = Cutout(n_holes=1, length=100, probability=1.0)

def apply_cutout(cutout_transform):
    
    img = Image.open("0_Parade_marchingband_1_5.jpg").convert('RGB')
    print(img)
    img_transformed = cutout_transform(img=img)
    topilimage = transforms.ToPILImage()
    img_pil = topilimage(img_transformed)
    img_pil.save("chang.jpg")

if __name__ == '__main__':
    img=apply_cutout(cutout_transform)