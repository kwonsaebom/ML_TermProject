import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 파일 경로
movies_path = 'preprocessed_movies.csv'

# 데이터 로딩
movies_df = pd.read_csv(movies_path)

# 장르 데이터 전처리
movies_df['genres'] = movies_df['genres'].apply(lambda x: x.strip('[]').replace('\'', '').split(', '))
movies_df['genres_joined'] = movies_df['genres'].apply(lambda x: ' '.join(x))

# CountVectorizer를 사용하여 장르 데이터를 원-핫 인코딩
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(movies_df['genres_joined'])

# Elbow 방법을 사용하여 적절한 클러스터 수 결정
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Elbow 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# 실제 클러스터링 (4개의 클러스터 사용)
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(X)


# 실루엣 점수 계산
silhouette_avg = silhouette_score(X, clusters)
print(f"Average Silhouette Score: {silhouette_avg}")

# 클러스터 결과를 영화 데이터프레임에 추가
movies_df['cluster'] = clusters

# 각 클러스터에 속하는 영화의 예시 확인
cluster_examples = movies_df.groupby('cluster').apply(lambda x: x.sample(3))[['title', 'genres']]
print(cluster_examples)

# 사용자 선호 클러스터 결정 예시 (여기서는 예시로 클러스터 0을 선호한다고 가정)
preferred_cluster = 0

# 선호 클러스터 내 영화 추천
recommended_movies = movies_df[movies_df['cluster'] == preferred_cluster].sample(5)
print("Recommended Movies:")
print(recommended_movies[['title', 'genres']])

# PCA를 사용하여 차원 축소
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X.toarray())

# 클러스터 결과 시각화
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title('Movie Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()