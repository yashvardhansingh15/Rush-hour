import streamlit as st
import tensorflow
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet import ResNet50, preprocess_input
import numpy as np
from keras.utils import load_img, img_to_array
from numpy.linalg import norm
import os
from PIL import Image
from tqdm import tqdm
import pickle
from sklearn.neighbors import NearestNeighbors
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
st.title('Rush Hour')
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0;

def feature_extraction(img_path, model):
    img1 = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img1)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices





uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
  if save_uploaded_file(uploaded_file):
      display_image=Image.open(uploaded_file)
      st.image(display_image)
      features = feature_extraction(os.path.join("uploads", uploaded_file.name),model)
      st.text(features)
      #recommend
      indices=recommend(features, feature_list)
      col1, col2, col3, col4, col5 = st.beta_columns(5)

      with col1:
          st.image(filenames[indices[0][0]])
      with col2:
          st.image(filenames[indices[0][1]])
      with col3:
          st.image(filenames[indices[0][2]])
      with col4:
          st.image(filenames[indices[0][3]])
      with col5:
          st.image(filenames[indices[0][4]])



  else:
      st.header("Some error occured in file upload")


