#This file combines various features like gender, category, color, season, etc., 
#into one description column to make it easier to search
import pandas as pd
import numpy as np
import re
import tensorflow_hub as hub
import pickle
df = pd.read_csv('styles.csv')

def preprocess_text(text):
    text = str(text).lower()  
    text = re.sub(r'\b(unknown|nan)\b', '', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    return text.strip()

def remove_duplicates(text):
    words = text.split()
    unique_words = list(dict.fromkeys(words))  
    return ' '.join(unique_words)

df['combined_features'] = (df['gender'] + " " +
                            df['masterCategory'] + " " +
                            df['subCategory'] + " " +
                            df['articleType'] + " " +
                            df['baseColour'] + " " +
                            df['season'] + " " +
                            df['year'].astype(str) + " " +
                            df['usage'] + " " +
                            df['productDisplayName'])
df['combined_features'] = df['combined_features'].apply(preprocess_text)
df['combined_features'] = df['combined_features'].apply(remove_duplicates)

df_filtered = df[['id', 'combined_features']]
df_filtered.to_csv('descriptions.csv', index=False)