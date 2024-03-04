'''
----------- T21 - Capstone Project - NLP Applications ------------
1. This code performs sentiment analysis on a dataset of product reviews.
    Additionally, the similarity analysis is performed.
2. The analized dataset is:
* name: "1429_1.csv",
* source: https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products,
* renamed as: 'amazon_product_reviews.csv'.
---------------------------- IMPORTANT ----------------------------
The following instalations in cmd are required to run this code:
* python -m spacy download en_core_web_sm
* pip install spacytextblob
* python -m textblob.download_corpora
'''

# For getting computing time of testing
import time
# For natural language processing (NLP)
import spacy
# For sentiment assessment
from spacytextblob.spacytextblob import SpacyTextBlob
# For dataset handling
import pandas as pd

# NLP model for sentiment assessment
nlp_sm = spacy.load('en_core_web_sm')
nlp_sm.add_pipe('spacytextblob')
# NLP model for text similarity comparison, as tested in Task 20
nlp_md = spacy.load('en_core_web_md')

# Function for sentiment analysis of a review
## Input: a review (format: string)
## Output: polarity score on a scale from -1 (most negative) to 1 (most positive)
def predict_sentiment(review):
 
    # Ensuring correct input format
    if not isinstance(review, str):
        raise Exception("Input must be a string!")
        
    # Pre-processing of the input text
    # Removing capitalization and leading spaces
    review = review.lower().strip()
    # Processing review text using spaCy ‘en_core_web_sm’
    doc = nlp_sm(review)
    # Filtering out stop words with exceptions
    # Stop words to be preserved to convey the real sentiment of a review
    exceptions = {
        "not", "no", "only","more", "really", "so", "much", "well", "first",
        "please", "many", "too", "full", "less", "just", "very", "mostly", "most",
        "few", "all", "everything"
        }
    no_stop_words = [word.text for word in doc if not (word.is_stop and word.text not in exceptions)]
    doc_clean = nlp_sm(' '.join(no_stop_words))
    # Filtering out punctuation and spaces
    no_punct_nor_space = [word.text for word in doc_clean if not word.is_punct and not word.is_space]
    doc_clean_2 = nlp_sm(' '.join(no_punct_nor_space))

    # Assessing review's sentiment polarity
    polarity = doc_clean_2._.blob.polarity
    return polarity

# Function comparing similarity of two product reviews
## Input: two reviews (format: string)
## Output: similarity score: from 0 (most dissimilar) to 1 (most similar)
def reviews_similarity(review_1, review_2):
    
    # Ensuring correct input format
    if not (isinstance(review_1, str) and isinstance(review_2, str)):
        raise Exception("Input type must be str!")
        
    # Process the input reviews using spaCy 'en_core_web_md'
    doc1 = nlp_md(review_1)
    doc2 = nlp_md(review_2)

    # Calculate the similarity between the processed reviews
    similarity_score = doc1.similarity(doc2)
    return similarity_score

# --------- Loading product reviews for sentiment analysis ---------
df = pd.read_csv("amazon_product_reviews.csv")
# Removing empty lines and adjusting indicies accordingly
reviews_data = df['reviews.text'].dropna().reset_index(drop=True)
# Make reviews_data dataset smaller for initial testing purposes, e.g. 100
reviews_data = reviews_data[0:500] # Comment out for testing the entire dataset

# ----------------------------------------------------------------
# ------------------------- Code testing -------------------------
print("\n-------- Testing the predict_sentiment function --------")
# Testing Case f_1: Using predict_sentiment function to calculate the number
# of positive/negative/neutral reviews in reviews_data

start_time_1 = time.time()  # Record start time

# Grouping reviews indicies into polarity types; later used in Testing Case f_3
negative = []
neutral = []
positive = []

num_reviews = len(reviews_data)
for i in range (0, num_reviews):
    review = reviews_data[i]
    polarity = predict_sentiment(review)
    if polarity < 0:
        negative.append(i)
    elif polarity == 0:
        neutral.append(i)
    else:
        positive.append(i)
        
# Calculating the number of negative/neutral/positive results       
num_negative = len(negative)
num_neutral = len(neutral)
num_positive = len(positive)
print(f"Negative: {num_negative}, neutral: {num_neutral}, positive {num_positive}")

# Checking if all reviews are analyzed
sum_polarities = num_negative + num_neutral + num_positive
if sum_polarities == num_reviews:
    print("Total number of polarity scores is correct.")
else:
    print("Error! Not all data was analysed.")
    
# Printing processing time
end_time_1 = time.time()  # Record end time
elapsed_time_1 = end_time_1 - start_time_1
print(f"Total computing time: {elapsed_time_1} seconds")

'''
Output (for 34,698 reviews)
    Negative: 1407, neutral: 2113, positive 31139
    Total number of polarity scores is correct.
    Total computing time: 738.0393946170807 seconds
Conclusions:
    1. Results seem probable.
    2. For number of possible error, see sentiment_analysis_report.
'''

print("\n----- Testing reviews_similarity function for simple sentences -----")
# Testing Case f_2: Using reviews_similarity function to compare the similarity
# between three pairs of chosen sentences: same, similar and dissimilar.

review_1 = "This is the best thing that happened to me after the invention of icecream."
review_2 = "I love everything about it!"
review_3 = "Monkeys are awful."

same = reviews_similarity(review_1, review_1)
similar = reviews_similarity(review_1, review_2)
opposite_1 = reviews_similarity(review_1, review_3)
opposite_2 = reviews_similarity(review_2, review_3)

print("\nTesting similarity of simple sentences:")
print(f"* same sentences (to verify the model): {same}")
print(f"* similar sentences: {similar}")
print(f"* opposite ones: {opposite_1}")
print(f"* other opposite ones: {opposite_2}")

# Testing how similar is the sentiment between these sentences
review_1_sent = predict_sentiment(review_1)
review_2_sent = predict_sentiment(review_2)
review_3_sent = predict_sentiment(review_3)

print("\nTesting sentiment of the simple sentences:")
print(f"* similar sentences: {review_1_sent} and {review_2_sent}")
print(f"* opposite ones: {review_1_sent} and {review_3_sent}")
print(f"* other opposite ones: {review_2_sent} and {review_3_sent}")
'''
Output:
    Testing similarity of simple sentences:
    * same sentences (to verify the model): 1.0
    * similar sentences: 0.5377868318016568
    * opposite ones: 0.5308918188473515
    * other opposite ones: 0.3673841043105134

    Testing sentiment of the simple sentences:
    * similar sentences: 1.0 and 0.5
    * opposite ones: 1.0 and -1.0
    * other opposite ones: 0.5 and -1.0
Conclusions:
    1. Similarity scores between opposite sentences can be almost the same
        (0.53) as for similar sentence (0.54).
    2. The similarity scores do not accurately reflect the degree of
        similarity between the meaning of sentences.
    3. It seems that the .similarity function analyzes similarity of words
        rather than the similarity of meaning, which is not the best.
    4. Thus, it might be better to assess polarity rather than similarity
        to correctly evaluate the similarity of meaning between sentences.
'''

print("\n-------- Testing reviews_similarity function for a dataset --------")

# Testing Case f_3: Using reviews_similarity function to compare the similarity
# between reviews within the same polarity group (positive/negative/neutral)

start_time_3_4 = time.time()  # Record start time

polarity_groups = {"negative": negative, "neutral": neutral, "positive": positive}
for label, reviews in polarity_groups.items():
    similarity = 0
    count = 0
    for i in range(len(reviews)):
        for j in range(i + 1, len(reviews)):
            review_1 = reviews_data[reviews[i]]
            review_2 = reviews_data[reviews[j]]
            similarity_score = reviews_similarity(review_1, review_2)
            similarity += similarity_score
            count += 1
    avg_similarity = similarity / count
    print(f"Average similarity for {label} reviews: {avg_similarity}")

# Testing Case f_4: calculate similarity between positive and negative reviews
diff_similarity = 0
diff_count = 0
for i in range(len(positive)):
    for j in range(len(negative)):
        review_1 = reviews_data[positive[i]]
        review_2 = reviews_data[negative[j]]
        similarity_score = reviews_similarity(review_1, review_2)
        diff_similarity += similarity_score
        diff_count += 1
diff_similarity_avg = diff_similarity / diff_count
print(f"Average similarity between positive and negative reviews: {diff_similarity_avg}")

# Printing processing time
end_time_3_4 = time.time()  # Record end time
elapsed_time_3_4 = end_time_3_4 - start_time_3_4
print(f"Total computing time: {elapsed_time_3_4} seconds")

'''
Output (for the first 500 reviews)
    Average similarity for negative reviews: 0.7836129640867298
    Average similarity for neutral reviews: 0.6641296058144789
    Average similarity for positive reviews: 0.7764690308338883
    Average similarity between positive and negative reviews: 0.7756452241150275
    Total computing time: 1966.601280927658 seconds
Conclusions:
    1. Average similarity of reviews in each polarity group (positive/neutral/
    positive) is the same as average similarity beween reviews in opposite groups
    (positive and negative)
    2. This corresponds to results from Testing Case f_2 where the lack of
    correlation between smilarity and polarity was shown.
'''