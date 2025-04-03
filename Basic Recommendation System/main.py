import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


columns = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=columns)
ratings.drop(columns=['timestamp'], inplace=True)  


movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown',
                 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=movie_columns, encoding='ISO-8859-1')
movies = movies[['movie_id', 'title']]  


df = pd.merge(ratings, movies, on='movie_id')


sns.set_style("whitegrid")
plt.figure(figsize=(8, 5))
sns.histplot(df['rating'], bins=5, kde=True, color="blue")
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Distribution of Movie Ratings')
plt.show()


movie_stats = df.groupby('title').agg({'rating': ['mean', 'count']})
movie_stats.columns = ['average_rating', 'rating_count']


min_ratings = 50
popular_movies = movie_stats[movie_stats['rating_count'] >= min_ratings].sort_values('average_rating', ascending=False)
print("\n🎬 **Top 10 Highest Rated Movies with at least 50 Ratings:**")
print(popular_movies.head(10))


valid_movies = movie_stats[movie_stats['rating_count'] >= min_ratings].index 
user_movie_matrix = df.pivot_table(index='user_id', columns='title', values='rating')[valid_movies]
user_movie_matrix.fillna(0, inplace=True)  


movie_name = input("\nEnter a movie title for recommendations: ")  


if movie_name in user_movie_matrix.columns:
    
    filtered_users = user_movie_matrix[user_movie_matrix[movie_name] > 0]
    
   
    similar_movies = user_movie_matrix.corrwith(filtered_users[movie_name])

    
    corr_df = pd.DataFrame(similar_movies, columns=['Correlation'])
    corr_df.dropna(inplace=True)

    
    corr_df = corr_df.join(movie_stats['rating_count'])

    
    corr_df = corr_df[corr_df.index != movie_name]

    
    corr_df['weighted_correlation'] = corr_df['Correlation'] * (corr_df['rating_count'] / corr_df['rating_count'].max())

    
    recommendations = corr_df[corr_df['rating_count'] >= min_ratings].sort_values('weighted_correlation', ascending=False)

    
    print(f"\n🎯 **Top 10 Movie Recommendations Similar to '{movie_name}':**")
    print(recommendations[['weighted_correlation', 'rating_count']].head(10))
else:
    print(f"\n❌ Movie '{movie_name}' not found in dataset.")
