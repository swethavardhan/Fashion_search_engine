#This script uses ResNet50 to extract feature vectors from fashion images, normalizes them
#and saves the features and file paths into pickle files for later use in image similarity 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = Sequential([model, GlobalMaxPooling2D()])


def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    result_normlized = flatten_result / norm(flatten_result)

    return result_normlized

img_files = []
for fashion_images in os.listdir('images'):
    images_path = os.path.join('images', fashion_images)
    img_files.append(images_path)

image_features = []
for files in tqdm(img_files):
    features_list = extract_features(files, model)
    image_features.append(features_list)
pickle.dump(image_features, open("img_features.pkl", "wb"))
pickle.dump(img_files, open("img_files.pkl", "wb"))
