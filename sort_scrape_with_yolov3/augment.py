import imageio
import imgaug as ia
import os
import cv2
from imgaug import augmenters as iaa
import numpy as np
from termcolor import colored

def im_aug(num_im,angle,blur,color_var,path='data/sorted',outfile='data/aug_data'):
    print(colored(f'Generating {num_im} Augmented Images for Every Item Scraped','blue'))
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-angle, angle)),
        iaa.AdditiveGaussianNoise(scale=(0, blur)),
        iaa.Crop(percent=(0, 0.05)),
        iaa.Multiply((1-color_var, 1+color_var), per_channel=0.5)
    ])

    if not os.path.exists(outfile):
        os.makedirs(outfile)
    for file in os.listdir(path):
        image = imageio.imread(f'{path}/{file}')
        images = [image]*num_im
        images_aug = seq(images=images)
        i=0
        name = file.replace('.jpg','')
        for aug in images_aug:
            i += 1
            cv2.imwrite(f'{outfile}/{name}-aug-{i}.jpg', cv2.cvtColor(aug, cv2.COLOR_BGR2RGB))
            
            
# if you have labeled images uses this code to transpose the labels onto the new image            
def im_aug_transpose_labels(num_im,angle,blur,color_var,path='data/sorted',outfile='data/aug_data'):
    print(colored(f'Generating {num_im} Augmented Images for Every Item Scraped','blue'))
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-angle, angle)),
        iaa.AdditiveGaussianNoise(scale=(0, blur)),
        iaa.Crop(percent=(0, 0.0)),
        iaa.Multiply((1-color_var, 1+color_var), per_channel=0.5)
    ])

    if not os.path.exists(outfile):
        os.makedirs(outfile)
        
    for file in glob.glob(f"{path}/*.jpg"):
        for idxer in range(num_im):
            keypoints_on_images = []
            image = imageio.imread(f'{file}')
            images = [image]*num_im
            h,w,c = image.shape
            seq_det = seq.to_deterministic() #locks the randomness
            images_aug = seq_det(images=images)
            txt_file = file.replace('.jpg','.txt')
            text = open(txt_file, "r").read().split()
            print(text)
            points = [ia.Keypoint(x=(float(text[1])*w), y=(float(text[2])*h))]
            keypoints_on_images.append(ia.KeypointsOnImage(points, shape=image.shape))
            keypoints_aug = seq_det.augment_keypoints(keypoints_on_images)
            name = file.replace(path,outfile)
            name = name.replace('.jpg','')
            for i, (aug, aug_k_pt, key_pt) in enumerate(zip(images_aug, keypoints_aug, keypoints_on_images)):
                for kp_idx, keypoint in enumerate(aug_k_pt.keypoints):
                    x_new, y_new = keypoint.x, keypoint.y    
            aug_points_yolo_format = [ f'{round((keypoint.x)/w,6)} {round((keypoint.y)/h,6)} ' for keypoint in aug_k_pt.keypoints]
            aug_points_yolo_format = ''.join([f'{text[0]} ']+aug_points_yolo_format+[f' {text[3]} {text[4]}'])
            print(aug_points_yolo_format)
            cv2.imwrite(f'{name}-aug-{idxer}.jpg', cv2.cvtColor(aug, cv2.COLOR_BGR2RGB))
            with open(f'{name}-aug-{idxer}.txt', "w") as text_file:
                text_file.write(aug_points_yolo_format)
