import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Sample perfume data with occasion and age group tag

df = pd.read_csv("data/final_perfume_data.csv", encoding='latin1')

#dummy occasion data
# Occasion_list = ['Casual', 'Evening', 'Daytime']
# df['Occasion'] = [random.choice(Occasion_list) for _ in range(len(df))]
#
# #dummy age data
# Age_range_list = ['Any', '25-50', '18-35']
# df['Age_Group'] = [random.choice(Age_range_list) for _ in range(len(df))]

# Combine description, occasion, and age group into a single text for TF-IDF Vectorizer
df['Combined'] = df['Description']

# Function to get user input
def get_user_input():
    occasion = input("Enter the occasion (Casual/Evening/Daytime): ").strip().capitalize()
    age_group = input("Enter the age group (Any/25-50/18-35): ").strip().capitalize()
    description = input("Enter the perfume description: ").strip().lower()
    return occasion, age_group, description




def get_recommendations(occasion, age_group, description, cosine_sim, df,tfidf_vectorizer, top_n=3):
    # Create a combined query
    query = description + ' ' + occasion + ' ' + age_group

    # Transform the query using the trained TF-IDF Vectorizer
    query_vector = tfidf_vectorizer.transform([query])

    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Combined'])

    # Calculate cosine similarity between query vector and perfume vectors
    sim_scores = cosine_similarity(query_vector, tfidf_matrix)

    # Get indices of top N most similar perfumes
    sim_scores = sim_scores[0]
    sim_scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:top_n]
    perfume_indices = [i[0] for i in sim_scores]

    results =  df[['Name','Image URL']].iloc[perfume_indices].reset_index(drop=True)

    return results.to_json()



