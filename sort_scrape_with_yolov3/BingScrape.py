import os
from absl import flags, app
from absl.flags import FLAGS
from selenium import webdriver
import urllib.request
import time
from bs4 import BeautifulSoup
import urllib.request
from thresh_sort import yolo_sort
from remove_dups import drop_duplicates
from augment import im_aug

flags.DEFINE_string('searches', 'trucks * semi trucks', 'list of searches separated by *')
flags.DEFINE_string('aug', True, 'augment_images')

def main(_argv):
    words = FLAGS.searches
    words = words.replace('-',' ')
    words.split('*')
    # Finds images from scrolling Bing
    def find_af_scroll(all_images):
        #Turns driver into soup
        html = driver.page_source
        soup = BeautifulSoup(html)
        for url in soup.find_all('img'):
            if url.get('alt') == f'Image result for {newword}':
                try:
                    if url.get('src')[:7] == "http://":
                        all_images.append(url.get('src'))
                except TypeError:
                    pass
        try:
            content = driver.find_element_by_css_selector('.btn_seemore')
            content.click()
        except:
            pass
        return all_images

    def collect_images_from_word(word):
        url = f'http://www.bing.com/images/search?q={word}&form=HDRSC2&first=1&tsc=ImageBasicHover'
        driver.get(url)
        all_images = []
        passes = 0
        while passes <= 14:
            passes += 1
            print(passes)
            time.sleep(2)
            all_images = find_af_scroll(all_images)
            #scoll window
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        all_images = list(set(all_images))
        return all_images

    # Start Driver
    driver = webdriver.Firefox()
    #downlaod images here I also appended other terms to the search
    total_images = []
    for word in words:
        for wordplus in ['',' on highway',' on freeway']:
            for company in ['xpo logistics ', 'DHL ','UPS ','FEDEX ', 'J.B. HUNT ', 'Ryder ']:
                newword = company + str(word) + wordplus
                all_images = collect_images_from_word(newword)
                total_images = total_images + all_images
                print(f'len: {len(total_images)}')

    #Download links extracted from Bing
    if not os.path.exists('data/raw_im'):
        os.makedirs('data/raw_im')
    t=31
    for x in total_images:
        time.sleep(0.001)
        urllib.request.urlretrieve(x, f"data/raw_im/{t}.jpg")
        t+=1

    #sort raw Bing image data with yolov3 custom weights
    yolo_sort('./data/labels/obj.names', './weights/yolov3-custom.tf', 1, tiny=False, thresh=0.85)
    #remove duplicate images
    drop_duplicates()
    #pad data with augment
    if FLAGS.searches:
        im_aug(4, 25, 20, 3, path='data/sorted', outfile='data/aug_data')

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
