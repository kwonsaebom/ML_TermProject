import pandas as pd
import argparse
import os
import pickle
import joblib
from sklearn.neighbors import NearestNeighbors
import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer

# def preprocess_data(movies_df):
#     # One-hot encode the genres
#     cv = CountVectorizer(tokenizer=lambda x: x.split('|'))
#     genres_one_hot = cv.fit_transform(movies_df['genres'])
#     return genres_one_hot

def model_fn(model_dir):
        recommender = joblib.load(os.path.join(model_dir, "model.joblib"))
        return recommender

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker's default arguments
    parser.add_argument('--model-dir', type=str, default=os.getenv('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.getenv('SM_CHANNEL_TRAINING', './data'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    
    # SageMaker's default arguments
    args = parser.parse_args()

    # Load and preprocess data
#     movies_df = pd.read_csv('s3://sagemaker-gacheon-003/data/preprocessed_movies.csv')
#     movies_df = pd.read_csv(os.path.joint(arg.train,'preprocessed_movies.csv'))
    # Correcting the line in train.py
    movies_df = pd.read_csv(os.path.join(args.train, 'preprocessed_movies.csv'))

#     genres_one_hot = preprocess_data(movies_df)

    # Train the NearestNeighbors
    recommender = NearestNeighbors(n_neighbors=2, metric='cosine')
    recommender = recommender.fit(movies_df)

    

    # Save the trained model
#     if not os.path.exists(args.model_dir):
#         os.makedirs(args.model_dir)
    with open(os.path.join(args.model_dir, 'movie_recommender_model.pkl'), 'wb') as f:
        pickle.dump(recommender, f)