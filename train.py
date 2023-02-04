import os, sys
from imageai.Detection import ObjectDetection
from imageai.Detection.Custom import DetectionModelTrainer


train = DetectionModelTrainer()
train.setModelTypeAsTinyYOLOv3()
train.setDataDirectory(data_directory=r"C:\Users\Matt\OneDrive\GitHub\SmartMirror\SmartMirror\pics")
train.setTrainConfig(
    object_names_array=['matt'],
    batch_size=4,
    train_from_pretrained_model='./models/tiny-yolov3.pt',
    num_experiments=185,
    
    )

train.trainModel()