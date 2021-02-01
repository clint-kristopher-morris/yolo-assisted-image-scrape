# yolo-assisted-image-scrape
Bing image scraper curated by deep learning. I created this repo for mass data collection for deep learning models. Particually, I created this repo for mass data collection for deep learning models. In particular, I have been testing how [SCAN (Learning to Classify Images without Labels)]( https://arxiv.org/pdf/2005.12320.pdf) compares to traditional methods (YOLOv4).

# Goal: Collect mass data that can be used to assign freight trucks to their respective FHAW classification.

Class Chart for this project:

![](https://i.ibb.co/xzHJKqV/classes2small.jpg)

Due to the reduced effort required to train a SCAN model it is advantageous to collect mass data for model training. Web-scraping is a useful tool when generating large data sets. However, search results drift continuously from relevancy. Inducing laborious effort to remove extraneous images not fit for labeling.

Example of images not useful for model training:

![](https://i.ibb.co/3pKtD88/badexamplessmall.png)

Advantageously, the tractor or front portion of the truck is common among most FHAW classes. I created a simple yolo model trained on only 300 images to act as a discriminator to filter out non truck images from the scraped data set. Scraping search results also enables me to target uncommon classes such as “Tractor B-train Double Trailers”. After collecting and sorting, I then made a model to remove duplicates and pad data by augmenting each image across 3 factors (rotation, noise, color).

# 1) Create yoloV3/V4 Darknet Weights for Discriminator Model

Label Images with this tool: https://github.com/alexismailov2/yolo-labeling-tool

![](https://media3.giphy.com/media/apWnL996NJojmc0ROd/giphy.gif)

Link to Darknet: https://github.com/AlexeyAB/darknet

I recommend [“the AI Guy”]( https://github.com/theAIGuysCode?tab=repositories) for the fastest way to learn how to train a custom darknet YOLO model. I also Implemented his method of converting YOLO weights to TensorFlow.

# 2) Run BingScrape.py to Collect Data

Example of how to scrape results for "freight trucks" and "semi/trucks": 

```
python BingScrape.py --searches freight-trucks*semi-trucks --aug 5
```

This process will:
 - Use the discriminator to remove all images that fall below a threshold. Set this value high!
 - Remove duplicate images.
 - Generate image augmentations for additional data.

# Results
Example of removed images:

![](https://i.ibb.co/HtKM7qd/Removedsm.png)

Example of selected images:

![](https://i.ibb.co/b698w21/sortedsm.png)

Example of augmentations of a single image:

![](https://i.ibb.co/Tct17xw/augsmall.png)

On my test set of 100k images, 60k were removed by the discriminator.
The image on the left shows a representative sample of class types passing this station. The disproportionality of this dataset is apparent which, creates issues when trying to develop a model that can accurately predict rare classes. The figure on the left shows how targeted scraping can be employed to collect scarce class types on a large scale.  

![](https://i.ibb.co/TrgcY7r/nat-dis.png) 
![](https://i.ibb.co/LSskgP0/targeted.png)

# Additional Yolo Relate Features  

### Transpose Labels Across Augmentations

To reduces monotonous work, I have also created a method of generating both images and YOLO formatted label simultaneously. Select your labeled image set and using the function below, to generate additional augmented data and labels.

```
from augment import im_aug_transpose_labels

im_aug_transpose_labels(num_im,angle,blur,color_var,path='data/sorted',outfile='data/aug_data')
```

![](https://media1.giphy.com/media/gTKVJJZTNta6vHatvB/giphy.gif)


