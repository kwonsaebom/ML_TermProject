import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# 데이터 불러오기
movies_df = pd.read_csv('second/dataset/preprocessed_dataset/preprocessed_movies.csv')
ratings_df = pd.read_csv('second/dataset/preprocessed_dataset/ratings_small.csv')
print(movies_df.head())
print(ratings_df.head())

# 장르 데이터를 one-hot 벡터로 변환
movies_df['genres'] = movies_df['genres'].apply(lambda x: ' '.join(eval(x)))
cv = CountVectorizer()
genres_one_hot = cv.fit_transform(movies_df['genres'])
genres_df = pd.DataFrame(genres_one_hot.toarray(), columns=cv.get_feature_names_out())
print("Genres One-Hot Encoded")
print(genres_df.head())

# 영화 데이터와 평점 데이터에서 일치하는 영화 ID만 필터링
ratings_movie_ids = ratings_df['movieId'].unique()
movies_movie_ids = movies_df['id'].unique()
common_movie_ids = set(movies_movie_ids).intersection(set(ratings_movie_ids))
filtered_movies_df = movies_df[movies_df['id'].isin(common_movie_ids)]
filtered_ratings_df = ratings_df[ratings_df['movieId'].isin(common_movie_ids)]
print("Filtered Movies and Ratings Data by Common IDs")
print(filtered_movies_df.head())
print(filtered_ratings_df.head())

# 평점 데이터의 평균 평점과 평점 개수 계산
filtered_ratings_mean_count = filtered_ratings_df.groupby('movieId').agg({'rating': ['mean', 'count']})
filtered_ratings_mean_count.columns = ['rating_mean', 'rating_count']
print("Calculated Mean and Count of Ratings")
print(filtered_ratings_mean_count.head())

# 영화 데이터와 평점 데이터 병합
filtered_movies_ratings_merged = pd.merge(filtered_movies_df, filtered_ratings_mean_count, left_on='id', right_index=True, how='left')
print("Merged Movies and Ratings Data")
print(filtered_movies_ratings_merged.head())

# 장르 데이터에 대한 NearestNeighbors 모델 학습
filtered_genres_df = genres_df.loc[filtered_movies_df.index]
nbrs_filtered = NearestNeighbors(n_neighbors=10, metric='cosine').fit(filtered_genres_df)
print("Trained NearestNeighbors Model on Filtered Genre Data")
print(nbrs_filtered)

# "The Dark Knight"의 인덱스 찾기 및 추천 목록 생성
filtered_movies_df_reset = filtered_movies_df.reset_index()
filtered_dark_knight_idx = filtered_movies_df_reset[filtered_movies_df_reset['title'] == 'The Dark Knight'].index[0]
distances, indices = nbrs_filtered.kneighbors([filtered_genres_df.iloc[filtered_dark_knight_idx]])
dark_knight_recommendations = filtered_movies_ratings_merged.iloc[indices[0]][['title', 'id']]
dark_knight_recommendations['distance'] = distances[0]
print("Generated Recommendations for 'The Dark Knight'")
print(dark_knight_recommendations)

# 추천 목록에 평점 정보 추가
dark_knight_recommendations = pd.merge(dark_knight_recommendations, filtered_movies_ratings_merged[['id', 'rating_mean', 'rating_count']], on='id', how='left')
print("Added Rating Information to Recommendations")
print(dark_knight_recommendations)

# Evaluation
# 테스트 데이터 생성
# 사용자별로 평점 4 이상인 영화를 선호 영화로 간주
preferred_movies = ratings_df[ratings_df['rating'] >= 4].groupby('userId')['movieId'].apply(list).to_dict()

# 추천 시스템을 사용하여 각 사용자에 대한 영화 추천 목록을 생성
# 예를 들어, 다음 함수는 각 사용자에 대해 10개의 영화를 추천.
def generate_recommendations(user_id, n_recommendations=10):
    # 추천 로직 구현 (예시)
    # 여기서는 단순히 임의의 영화를 추천하는 것으로 가정합니다.
    return movies_df.sample(n_recommendations)['id'].tolist()


user_recommendations = {user: generate_recommendations(user) for user in preferred_movies.keys()}

# 평가
precisions = []
recalls = []
f1_scores = []

for user, preferred in preferred_movies.items():
    recommended = user_recommendations[user]

    true_positives = len(set(preferred) & set(recommended))
    false_positives = len(set(recommended) - set(preferred))
    false_negatives = len(set(preferred) - set(recommended))

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# 평균 평가 점수
avg_precision = sum(precisions) / len(precisions)
avg_recall = sum(recalls) / len(recalls)
avg_f1_score = sum(f1_scores) / len(f1_scores)

print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Average F1 Score: {avg_f1_score}")
