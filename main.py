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

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([model,GlobalMaxPooling2D()])

def extract_features(file_path,model):
    img = image.load_img(file_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expand_img_array = np.expand_dims(img_array,axis=0)
    process_image = preprocess_input(expand_img_array)
    result = model.predict(process_image).flatten()
    normalise_result = result/norm(result)
    return normalise_result

file_names = []
for file in os.listdir('archive/cars_test/cars_test/'):
    file_names.append(os.path.join('archive/cars_test/cars_test/',file))
for file in os.listdir('archive/cars_train/cars_train/'):
    file_names.append(os.path.join('archive/cars_train/cars_train/',file))

feature_list=[]
for file in tqdm(file_names):
    feature_list.append(extract_features(file,model))

pickle.dump(file_names,open('file_names.pkl','wb'))
pickle.dump(feature_list,open('feature_list.pkl','wb'))