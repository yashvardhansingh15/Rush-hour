import tensorflow
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet import ResNet50, preprocess_input
import numpy as np
from keras.utils import load_img, img_to_array
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
import cv2

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


def extract_features(img_path, model):
    img1 = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img1)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

for file in indices[0]:
    temp_img = cv2.imread(filenames[file])
    resized_img = cv2.resize(temp_img, (300, 300))
    cv2.imshow('Resized Image', resized_img)
    cv2.waitKey(0)
