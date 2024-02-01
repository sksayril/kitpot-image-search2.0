from flask import Flask, request, jsonify
from model import load_model, find_similar_images,extract_filenames
import os
import difflib
import pandas as pd
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)
data = pd.read_csv('kitpot.categoriesNew.csv', encoding='ISO-8859-1')
titles = data['DATA'].tolist()

lowercase_titles = [str(title).lower() if isinstance(title, str) else '' for title in titles]
data = list(filter(None, lowercase_titles))

def find_closest_match(input_word, data):
    if input_word in data:
        return [input_word]

    closest_match = difflib.get_close_matches(input_word, data)
    return closest_match

def process_search_term(search_term, data):
    words = search_term.split(' ')
    processed_words = []

    for word in words:
        
        if word.lower() in data:
            processed_words.append(word)
        else:
            closest_match = find_closest_match(word.lower(), data)
            processed_words.append(closest_match[0] if closest_match else word)

    processed_sentence = ' '.join(processed_words)
    return processed_sentence

@app.route('/get_closest_match', methods=['GET'])
def get_closest_match():
    search_term = request.args.get('search_key', default='', type=str)
    
    if not search_term:
        response = {
            'message': 'Search term is empty.',
        }
        return jsonify(response)

    processed_sentence = process_search_term(search_term, data)
    
    if processed_sentence:
        response = {
            'result': processed_sentence,
        }
    else:
        response = {
            'message': 'No similar words found.',
        }

    return jsonify(response)


@app.route('/similar_images', methods=['POST'])
def get_similar_images():
    if 'imageSearch' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['imageSearch']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = file.filename
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        similar_images = find_similar_images(filepath)
        actualFormatedData=extract_filenames(similar_images)
        # actualFormatedData = convert_file_paths(similar_images)
        os.remove(filepath)
        
        return jsonify({'result': actualFormatedData})
    else:
        return jsonify({'error': 'File format not supported'})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)

