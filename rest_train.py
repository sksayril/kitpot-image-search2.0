import os
import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

root_folder = 'kitpotproduct'
feature_list = []
filenames = []

for root, dirs, files in os.walk(root_folder):
    for file in tqdm(files):
        if file.lower().endswith(('.png', '.jpg', '.jpeg','PNG','JPG','JPEG')):
            img_path = os.path.join(root, file)
            try:
                img = image.load_img(img_path, target_size=(224, 224))
                filenames.append(img_path)
                feature_list.append(extract_features(img_path, model))
            except (OSError, IOError, ValueError, Exception) as e:
                print(f"Error processing file: {img_path}")
                print(f"Error message: {str(e)}")
                continue
with open('res_vector_embeddings.pkl', 'wb') as f:
    pickle.dump(feature_list, f)

with open('res_filenames.pkl', 'wb') as f:
    pickle.dump(filenames, f)