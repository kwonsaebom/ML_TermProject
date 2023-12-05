<h2>GOAL</h2>
: To build a comprehensive movie recommendation system that offers movie suggestions to users based on their preferences, historical ratings, and similar user behaviors. The system will be adaptive, considering both the content of the movies and collaborative filtering techniques. </br></br>
   
<img width="412" alt="image" src="https://github.com/kwonsaebom/ML_TermProject/assets/94830364/e6474717-60bf-4867-aa19-2684fd5e7c69">
</br></br>
<h2>Dataset</h2>
: 
- Credits.csv : Consists of Cast and Crew Information
- Keywords.csv : Contains the movie plot keywords
- Links.csv : Contains the TMDB and IMDB IDs of all the movies featured
- Movies_metadata.csv : Main Movies Metadata file
- Rating.csv : Ratings from users on movies
==> dataset & ml-latest-small

<h2>Code</h2>
<br>
1. Data Preprocssing -> preprocessing.py
2. Content Based Filtering
   - content_based_recommendation.py
   - content_based_recommendation.ipynb
   - content_based_kmeans.py
   - content_based_knn.py
3. Item Based Filtering
   - Collaborative_Filtering.py
   - Collaborative_Filtering.ipynb
4. Sagemaker
  - encapsulation.ipynb
  - Endpoint.py
  - train.py
  - Recommendation.ipynb 
