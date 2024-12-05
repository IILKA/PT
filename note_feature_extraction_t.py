import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import torch
import os

#cleaning the text
def preprocess_text(text):
    # Remove placeholders
    text = re.sub(r'___', ' ', text)
    # Segment sections based on headers
    text = re.sub(r'\n+', '\n', text)  # Normalize newline characters
    # Convert to lowercase
    text = text.lower()
    # Remove numbers and punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    stopwords = set(['the', 'and', 'is', 'in', 'to', 'a', 'for', 'of', 'on', 'with', 'at', 'by', 'as', 'an', 'it'])
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text.strip()


data_path = "./notes.csv"
data = pd.read_csv(data_path)
print("The data has been loaded successfully.")

# Create output folder
output_folder = './notes_embedding_t' # this may need some adjustment for your own use case
os.makedirs(output_folder, exist_ok=True)

# Preprocess text column
data['cleaned_text'] = data['text'].fillna('').apply(preprocess_text)

# Extract IDs and cleaned text
texts = data['cleaned_text']
ids = data['id']

# TF-IDF Feature Extraction
print("Extracting TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(max_features=1024)
tfidf_features = tfidf_vectorizer.fit_transform(texts)

# Save features as .pt files
print("Saving features to .pt files...")
for i, doc_id in tqdm(enumerate(ids), desc="Processing notes", total=len(ids)):
    try:
        # Extract feature vector
        feature_vector = tfidf_features[i].toarray()[0]
        feature_tensor = torch.tensor(feature_vector, dtype=torch.float32)
        
        # Save to .pt file
        output_path = os.path.join(output_folder, f'{doc_id}.pt')
        torch.save(feature_tensor, output_path)
    except Exception as e:
        print(f"Error saving features for document ID {doc_id}: {e}")

print(f"Feature files have been successfully generated in '{output_folder}'.")
