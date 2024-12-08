#This code generates text embeddings for each clothing item's description using the 'paraphrase-MiniLM-L6-v2' model 
#and stores the embeddings in a pickle file
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
df = pd.read_csv('descriptions.csv')
df['combined_features'] = df['combined_features'].fillna('').astype(str)
text_embeddings = {
    row['id']: model.encode(row['combined_features'])
    for _, row in df.iterrows()
}
with open('text_embeddings.pkl', 'wb') as f:
    pickle.dump(text_embeddings, f)
