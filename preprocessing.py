import pandas as pd
import ast

# Define file paths for the data files
movies_metadata_path = 'movies_metadata.csv'
credits_path = 'credits.csv'
keywords_path = 'keywords.csv'
ratings_path = 'ratings_small.csv'

# Load the data into DataFrames
movies_metadata_df = pd.read_csv(movies_metadata_path)
credits_df = pd.read_csv(credits_path)
keywords_df = pd.read_csv(keywords_path)
ratings_df = pd.read_csv(ratings_path)

# Check the loaded data
print("movies_metadata_df:")
print(movies_metadata_df.head())
print("\ncredits_df:")
print(credits_df.head())
print("\nkeywords_df:")
print(keywords_df.head())
print("\nratings_df:")
print(ratings_df.head())

# Standardize the data type of the 'id' column for data merging
credits_df['id'] = pd.to_numeric(credits_df['id'], errors='coerce')
keywords_df['id'] = pd.to_numeric(keywords_df['id'], errors='coerce')
movies_metadata_df['id'] = pd.to_numeric(movies_metadata_df['id'], errors='coerce')

# Merge the data frames
merged_df = pd.merge(movies_metadata_df, credits_df, on='id', how='left')
merged_df = pd.merge(merged_df, keywords_df, on='id', how='left')
print("\nMerged DataFrame:")
print(merged_df.head())

# Modify data types and handle missing values
merged_df['budget'] = pd.to_numeric(merged_df['budget'], errors='coerce')
merged_df['popularity'] = pd.to_numeric(merged_df['popularity'], errors='coerce')
merged_df['adult'] = merged_df['adult'].astype('bool')
merged_df['video'] = merged_df['video'].astype('bool')
merged_df = merged_df.dropna(subset=['id'])
merged_df['id'] = merged_df['id'].astype('int')

numeric_columns = merged_df.select_dtypes(include=['float64', 'int64']).columns
string_columns = merged_df.select_dtypes(include=['object']).columns

merged_df[numeric_columns] = merged_df[numeric_columns].fillna(merged_df[numeric_columns].median())
merged_df[string_columns] = merged_df[string_columns].fillna('')
print("\nModified DataFrame:")
print(merged_df.head())

# Parse JSON-formatted columns
def safe_parse_json_column(column, key):
    def parse(row):
        try:
            data = ast.literal_eval(row)
            if isinstance(data, list):
                return [i[key] for i in data]
            return []
        except (ValueError, SyntaxError):
            return []
    return column.apply(parse)

merged_df['genres'] = safe_parse_json_column(merged_df['genres'], 'name')
merged_df['cast'] = safe_parse_json_column(merged_df['cast'], 'name')
merged_df['crew'] = safe_parse_json_column(merged_df['crew'], 'name')
merged_df['keywords'] = safe_parse_json_column(merged_df['keywords'], 'name')
print("\nParsed Columns:")
print(merged_df[['genres', 'cast', 'crew', 'keywords']].head())

# Extract top 3 actors
merged_df['top_3_actors'] = merged_df['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)
print("\nExtracted Top 3 Actors:")
print(merged_df[['title', 'top_3_actors']].head())

# Define a function to extract director information from the 'crew' column and apply it
def extract_director_from_crew(crew_json):
    crew_data = ast.literal_eval(crew_json)
    for crew_member in crew_data:
        if crew_member.get('job') == 'Director':
            return crew_member.get('name')
    return None

credits_df['director'] = credits_df['crew'].apply(extract_director_from_crew)

# Add director information to the merged DataFrame
merged_df = pd.merge(merged_df, credits_df[['id', 'director']], on='id', how='left')
print("Merged DataFrame (Director Info added):")
print(merged_df.head())

# Print extracted director information
print("\nExtracted Director Information:")
print(merged_df[['title', 'director']].head())

# Save the processed DataFrames to CSV files
merged_df.to_csv('preprocessed_movies.csv', index=False)
credits_df.to_csv('preprocessed_credits.csv', index=False)
