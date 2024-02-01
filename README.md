# Image Retrieval System using ResNet50 and Nearest Neighbors

This repository contains code for an image retrieval system built using ResNet50, a pre-trained convolutional neural network, and Nearest Neighbors algorithm to find similar images based on feature embeddings.

## Overview

The system leverages ResNet50, a powerful deep learning model pre-trained on ImageNet, for extracting image features. These features are stored as embeddings in a pickle file, along with associated filenames.

### Files Included

- `app.py`: Python script implementing the image retrieval system.
- `res_vector_embeddings`: Pickle file containing feature embeddings of images.
- `filenames.pkl`: Pickle file storing filenames corresponding to the image embeddings.

## Getting Started

### Prerequisites

- Python 3.10
- Dependencies: Keras, NumPy, scikit-learn

You can install the dependencies via:

Requirements installation
```bash
pip install -r requirements.txt
```
Run the Model

```bash
python app.py
```


how to run the flask app in Contunuos in aws


```bash
nohub python3 app.py
```
