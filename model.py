import os
import pickle
import sys
import keras
import numpy as np
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

model = None
feature_list = None
filenames = None

def load_model():
    global model, feature_list, filenames
    if model is None:
        model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model.trainable = False
        model = keras.Sequential([
            model,
            GlobalMaxPooling2D()
        ])
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_path = os.path.join(script_dir, 'res_vector_embeddings.pkl')
    filenames_path = os.path.join(script_dir, 'res_filenames.pkl')

    try:
        with open(embeddings_path, 'rb') as emb_file, open(filenames_path, 'rb') as name_file:
            feature_list = pickle.load(emb_file)
            filenames = pickle.load(name_file)
    except FileNotFoundError as e:
        print(f"Error: {e}.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading pickle files: {e}")
        sys.exit(1)

def find_similar_images(image_path):
    if model is None or feature_list is None or filenames is None:
        load_model()

    try:
        query_img = image.load_img(image_path, target_size=(224, 224))
        query_img_array = image.img_to_array(query_img)
        expanded_query_img_array = np.expand_dims(query_img_array, axis=0)
        preprocessed_query_img = preprocess_input(expanded_query_img_array)
        query_result = model.predict(preprocessed_query_img).flatten()
        normalized_query_result = query_result / norm(query_result)

        neighbors = NearestNeighbors(n_neighbors=100, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)

        distances, indices = neighbors.kneighbors([normalized_query_result])

        similar_image_paths = [filenames[idx] for idx in indices[0][1:]]
        return similar_image_paths
    except FileNotFoundError as e:
        print(f"Error: {e}. Check if the specified image file exists.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# def convert_file_paths(array_data):
#     base_path = "uploads/catalog/product/"
#     transformed_paths = [base_path + path.replace("\\", "/") for path in array_data]
    
#     # Remove the "kitpotproduct/" prefix from each path
#     transformed_paths = [path.replace("uploads/catalog/product/kitpotproduct/", "uploads/catalog/product/") for path in transformed_paths]
    
#     return transformed_paths
def extract_filenames(paths):
    return [path.split("\\")[-1] for path in paths]
