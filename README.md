# yolo-assisted-image-scrape
Bing image scraper curated by deep learning. I created this repo for mass data collection for deep learning models. Particually, I created this repo for mass data collection for deep learning models. In particular, I have been testing how [SCAN (Learning to Classify Images without Labels)]( https://arxiv.org/pdf/2005.12320.pdf) compares to traditional methods (YOLOv4).

# Goal: Collect mass data that can be used to assign freight trucks to their respective FHAW classification.

Class Chart for this project:

![](https://i.ibb.co/xzHJKqV/classes2small.jpg)

Due to the reduced effort required to train a SCAN model it is advantageous to collect mass data for model training. Web-scraping is a useful tool when generating large data sets. However, search results drift continuously from relevancy. Inducing laborious effort to remove extraneous images not fit for labeling.

Example of images not useful for model training:

![](https://i.ibb.co/3pKtD88/badexamplessmall.png)

Advantageously, the tractor or front portion of the truck is common among most FHAW classes. I created a simple yolo model trained on only 300 images to act as a discriminator to filter out non truck images from the scraped data set. Scraping search results also enables me to target uncommon classes such as “Tractor B-train Double Trailers”. After collecting and sorting, I then made a model to remove duplicates and pad data by augmenting each image across 3 factors (rotation, noise, color).


# 1) Create yoloV3/V4 Darknet Weights 
Link to Darknet: https://github.com/AlexeyAB/darknet

Label Images with this tool: https://github.com/alexismailov2/yolo-labeling-tool

![](https://media3.giphy.com/media/apWnL996NJojmc0ROd/giphy.gif)


