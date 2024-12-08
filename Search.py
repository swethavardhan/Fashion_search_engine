#This code enables searching for similar fashion items using either an uploaded image 
# or a text description by extracting features and finding similar matches
import streamlit as st
import tensorflow
import pandas as pd
from PIL import Image, ImageEnhance
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

features_list = pickle.load(open("img_features.pkl", "rb"))
img_files = pickle.load(open("img_files.pkl", "rb"))
with open('text_embeddings.pkl', 'rb') as f:
    text_embeddings = pickle.load(f)

img_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
img_model.trainable = False
img_model = Sequential([img_model, GlobalMaxPooling2D()])
text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def save_file(uploaded_file):
    try:
        with open(os.path.join("uploader", uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0        

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    result_norml = flatten_result/norm(flatten_result)
    return result_norml  

def recommend(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    distance, indices = neighbors.kneighbors([features])
    return indices

def enhance_image(image_path):
    img = Image.open(image_path)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)  
    return img

def search_similar_clothing(query, top_n=5):
    query_embedding = text_model.encode(query)
    similarities = {}
    for cloth_id, text_embedding in text_embeddings.items():
        similarity = cosine_similarity([query_embedding], [text_embedding])[0][0]
        similarities[cloth_id] = similarity

    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    top_results = sorted_similarities[:top_n]
    
    top_cloth_ids = [result[0] for result in top_results]
    return top_cloth_ids

# Streamlit Interface
st.title("Fashion Search")
input_type = st.radio("Choose your input type", ('Image Search', 'Text Search'))

if input_type == 'Image Search':
    uploaded_file = st.file_uploader("Choose an image")
    if uploaded_file is not None:
        if save_file(uploaded_file):
            show_images = Image.open(uploaded_file)
            size = (400, 400)
            resized_im = show_images.resize(size)
            st.image(resized_im)
            features = extract_features(os.path.join("uploader", uploaded_file.name), img_model)
            img_indices = recommend(features, features_list)
            st.write("Top matches:")
            c1, c2, c3, c4, c5 = st.columns(5)
            for i, col in enumerate([c1, c2, c3, c4, c5]):
                with col:
                    st.header(f"Match {i+1}")
                    st.image(img_files[img_indices[0][i]])

elif input_type == 'Text Search':
    query = st.text_input("Enter a description (e.g., 'red floral dress')")
    if query:
        top_cloth_ids = search_similar_clothing(query)
        st.write(f"Top matches for your query: {query}")
        cols = st.columns(5)  
        for idx, cloth_id in enumerate(top_cloth_ids):
            image_path = os.path.join("images", f"{cloth_id}.jpg")
            if os.path.exists(image_path):
                img = enhance_image(image_path)
                with cols[idx]:  
                    st.image(img, caption=f"Cloth ID: {cloth_id}", use_column_width=True)
            else:
                st.write(f"Image not found for Cloth ID: {cloth_id}")