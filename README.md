# yolo-assisted-image-scrape
Bing image scraper curated by deep learning. I created this repo for mass data collection for deep learning models. Particually, I created this repo for mass data collection for deep learning models. In particular, I have been testing how [SCAN (Learning to Classify Images without Labels)]( https://arxiv.org/pdf/2005.12320.pdf) compares to traditional methods (YOLOv4).

# Goal: Collect mass data that can be used to assign freight trucks to their respective FHAW classification.


![Class](https://i.ibb.co/xzHJKqV/classes2small.jpg)

Due to the reduced effort required to train a SCAN model it is advantageous to collect mass data for model training. This repo uses a simple yolo model trained to 600 images to curate data collection. 


# 1) Create yoloV3/V4 Darknet Weights 
Link to Darknet: https://github.com/AlexeyAB/darknet

Label Images with this tool: https://github.com/alexismailov2/yolo-labeling-tool

![](https://media3.giphy.com/media/apWnL996NJojmc0ROd/giphy.gif)


