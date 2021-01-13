import imageio
import imgaug as ia
import os
import cv2

def im_aug(num_im,angle,blur,color_var,path='data/sorted',outfile='data/aug_data'):
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
        print("Augmented:")
        ia.imshow(np.hstack(images_aug))
        i=0
        name = file.replace('.jpg','')
        for aug in images_aug:
            i += 1
            cv2.imwrite(f'{outfile}/{name}-aug-{i}.jpg', cv2.cvtColor(aug, cv2.COLOR_BGR2RGB))