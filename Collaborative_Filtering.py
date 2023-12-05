import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import f1_score

# Load ratings and movies data
ratings = pd.read_csv("second/ml-latest-small/ratings.csv")
movies = pd.read_csv("second/ml-latest-small/movies.csv")

# Rename the 'id' column to 'movieId' and convert it to int64
movies = movies.rename(columns={"id":"movieId"})
movies["movieId"] = movies["movieId"].astype("int64")

# Display movie information
print("Movies:")
print(movies[['movieId', 'title']])

# Create a user-item matrix based on ratings
def create_user_item_matrix(ratings) -> pd.DataFrame:
    mat = ratings.pivot(index="userId", columns="movieId", values="rating")
    # Treat movies rated by users as 'seen' and assign a value of 1
    mat[~mat.isna()] = 1
    # Treat movies not rated by users as 'unseen' and assign a value of 0
    mat.fillna(0, inplace=True)
    return mat

user_item_matrix = create_user_item_matrix(ratings)
print(user_item_matrix)

# Get predictions using Singular Value Decomposition (SVD)
def get_svd_prediction(user_item_matrix, k):
    # Obtain U, sigma, V
    u, s, vh = scipy.sparse.linalg.svds(user_item_matrix.to_numpy(), k=k)
    # Reconstruct the original matrix using these components
    preds = np.dot(np.dot(u, np.diag(s)), vh)

    # Convert the result to a DataFrame and normalize values between 0 and 1
    preds = pd.DataFrame(preds, columns=user_item_matrix.columns, index=user_item_matrix.index)
    preds = (preds - preds.min()) / (preds.max() - preds.min())
    return preds

predictions = get_svd_prediction(user_item_matrix, k=64)
print(predictions)

# Example user_id for recommendations
user_id = 609
user_movie_ids = ratings[ratings.userId == user_id].movieId
user_movies = movies[movies.movieId.isin(user_movie_ids)]
print(user_id, "User's watched movies:")
print(len(user_movies), user_movies)

# Retrieve user predictions and exclude already watched movies
user_predictions = predictions.loc[user_id].sort_values(ascending=False)
predicts = user_predictions.head(10)
user_predictions = user_predictions[~user_predictions.index.isin(user_movie_ids)]
# Get the top 10 movies with the highest predicted values
user_predictions = user_predictions.head(10)
# Retrieve information about the top 10 recommended movies
user_recommendations = movies[movies.movieId.isin(user_predictions.index)]
user_recommendations["recommendation_score"] = user_predictions.values
print("Movies the user hasn't watched and recommended:")
print(user_recommendations)

print(len(user_movies))

print(user_recommendations)

# Singular Value Decomposition (SVD) class
class SVD:
    def __init__(self, ratings, movies, k): 
        user_item_matrix = create_user_item_matrix(ratings)
        self.preds = get_svd_prediction(user_item_matrix, k)
        self.ratings = ratings
        self.movies = movies

    def get_recommendations(self, user_id, top_k=10):
        user_movie_ids = self.ratings[self.ratings.userId == user_id].movieId
        user_movies = self.movies[self.movies.movieId.isin(user_movie_ids)]

        # Retrieve user predictions and exclude already watched movies
        user_predictions = self.preds.loc[user_id].sort_values(ascending=False)
        user_predictions = user_predictions[~user_predictions.index.isin(user_movie_ids)]
        # Get information about the top_k recommended movies
        user_recommendations = self.movies[self.movies.movieId.isin(user_predictions.index)]
        user_recommendations["recommendation_score"] = user_predictions.values
        
        return user_recommendations if top_k is None else user_recommendations.head(top_k)

# Instantiate the SVD class
svd = SVD(ratings, movies, 64)
# Get recommendations for a specific user
print(svd.get_recommendations(user_id))

# Define a function to calculate F1 score
def f1_score(svd, user_id, target_movie_ids, k=10):

    # Get the top-k recommendations for the user
    # 주어진 사용자 ID에 대한 top-k 추천 목록을 생성
    rec = svd.get_recommendations(user_id, k)

    # Calculate the number of true positives
    # 추천 목록에 타겟 영화 세트에 포함된 영화가 있는지 확인
    tp = len(rec[rec.movieId.isin(target_movie_ids)])

    # Calculate the number of false positives
    fp = k - tp
    # Calculate the number of false negatives
    # 타겟 영화 세트에 포함된 영화 중 추천되지 않은 영화의 수를 계산
    fn = len(target_movie_ids) - tp
    # Calculate the precision
    
    precision = tp / (tp + fp)
    # Calculate the recall
    recall = tp / (tp + fn)
    
    # Calculate the F1 score
    f1 = 2 * precision * recall / (precision + recall)

    return f1

# Define a target movie set
target_movie_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
]

# Calculate the F1 score for each user
for user_id in ratings.userId.unique():
    f1 = f1_score(svd, user_id, target_movie_ids)
    print(f"User {user_id}: {f1:.2f}")

# Calculate the average F1 score across all users
average_f1 = (
    sum([f1_score(svd, user_id, target_movie_ids) for user_id in ratings.userId.unique()])
    / len(ratings.userId.unique())
)
print(f"Average F1 score: {average_f1:.2f}")
