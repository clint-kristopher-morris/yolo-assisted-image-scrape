import numpy as np
from absl import logging
import cv2

import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import os
import shutil

def yolo_sort(obj_path,tf_weights,num_classes,tiny=False,thresh=0.85):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    
    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    #load weights
    if tiny:
        yolo = YoloV3Tiny(classes=num_classes)
    else:
        yolo = YoloV3(classes=num_classes)
    yolo.load_weights(tf_weights)
    logging.info('weights loaded')
    class_names = [c.strip() for c in open(obj_path).readlines()]
    logging.info('classes loaded')

    non_sorted = 'data/raw_im'
    sorted_path = 'data/sorted'
    for img_lo in os.listdir(non_sorted):
        try:
            img_in = cv2.imread(f'{non_sorted}/{img_lo}')
        except FileNotFoundError:
            continue
        # img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, 416)

        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img_in, boxes[0])
        features = encoder(img_in, converted_boxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        for detection in detections:
            print(f'im: {img_lo}  confidence: {detection.confidence}')
            if detection.confidence > thresh:
                try:
                    shutil.move(f'{non_sorted}/{img_lo}', f'{sorted_path}/{img_lo}')
                except FileNotFoundError:
                    continue


