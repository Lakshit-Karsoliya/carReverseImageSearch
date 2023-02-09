import streamlit as st

import numpy as np
import os
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50 , preprocess_input
from keras.applications.resnet import ResNet
from numpy.linalg import norm
from tqdm import tqdm
import pickle
from sklearn.neighbors import NearestNeighbors
# import cv2

st.title('car reverse image search')
def extract_features(file_path,model):
    img = image.load_img(file_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expand_img_array = np.expand_dims(img_array,axis=0)
    process_image = preprocess_input(expand_img_array)
    result = model.predict(process_image).flatten()
    normalise_result = result/norm(result)
    return normalise_result
def save_file(file):
    try:
        with open(os.path.join('uploads',file.name),'wb') as f:
            f.write(file.getbuffer())
        return 1
    except:
        return 0

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

file_name = pickle.load(open('file_names.pkl','rb'))
feature_list = np.array(pickle.load(open('feature_list.pkl','rb')))


neighbors = NearestNeighbors(n_neighbors=4,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)


uploaded_file = st.file_uploader('choose an image')
if uploaded_file is not None:
    if save_file(uploaded_file):
        st.image(uploaded_file)
        img_features = extract_features(os.path.join('uploads',uploaded_file.name),model)
        dis,idx = neighbors.kneighbors([img_features])
        # for i in idx[0]:
        #     st.write(file_name[i])
        st.subheader("Recommended images are")   
        col1,col2=st.columns(2)
        col3,col4=st.columns(2)
        with col1:
            st.image(file_name[idx[0][0]])
        with col2:
            st.image(file_name[idx[0][1]])
        with col3:
            st.image(file_name[idx[0][2]])
        with col4:
            st.image(file_name[idx[0][3]])
    else:
        st.write('error occour')


