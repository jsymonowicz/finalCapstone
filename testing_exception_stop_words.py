'''
----------- T21 - Capstone Project - NLP Applications ------------
1. This code tests which stop words should not be removed from the
predict_sentiment function to preserve a true sentiment of a review.
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
import time
import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob

# NLP model for assessing sentiment
nlp_sm = spacy.load('en_core_web_sm')
nlp_sm.add_pipe('spacytextblob')

# Stop words to be preserved - for testing purposes words are added or removed
# to the dictionary, to reduce the error of changing polarity sign
exceptions = {
    "not", "no", "only","more", "really", "so", "much", "well", "first",
    "please", "many", "too", "full", "less", "just", "very", "mostly",
    "most", "few", "all", "everything"
    }

# Tested function for sentiment analysis of a string
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
    no_stop_words = [word.text for word in doc if not (word.is_stop and word.text.lower() in exceptions)]
    doc_clean = nlp_sm(' '.join(no_stop_words))
    # Filtering out punctuation and spaces
#    no_punct_nor_space = [word.text for word in doc_clean if not word.is_punct and not word.is_space]
#    doc_clean_2 = nlp_sm(' '.join(no_punct_nor_space))

    # Assessing review's sentiment polarity
    polarity = doc_clean._.blob.polarity
    return polarity

# Assessing polarity of the original sentence without filtering
def sentiment_no_filtering(review):
    review = review.lower().strip()
    doc = nlp_sm(review)
    polarity = doc._.blob.polarity
    return polarity

# Testing function returning text after removing stop words with exceptions
def removing_stop_words_except(review):
    review = review.lower().strip()
    doc = nlp_sm(review)
    no_stop_words = [word.text for word in doc if not (word.is_stop and word.text.lower() in exceptions)]
    doc_clean = nlp_sm(' '.join(no_stop_words))
    no_punct_nor_space = [word.text for word in doc_clean if not word.is_punct and not word.is_space]
    doc_clean_2 = nlp_sm(' '.join(no_punct_nor_space))
    return(doc_clean_2)

# ------------------------- Testing -------------------------------
# How different exception words influence predict_sentiment function's output

print("------------ Testing Case 2_1 ------------")
# Checking how negations with "n't" affect the sentiment
sentence_1 = "This is great and works!"
sentence_2 = "This isn't great and doesn't work!"
processed = removing_stop_words_except(sentence_2)
print(f"Original: {sentence_1}")
print(f"Score: {predict_sentiment(sentence_1)}")
print(f"Processed: {processed}")
print(f"Score: {predict_sentiment(sentence_2)}")
'''
Output:
    Original: This is great and works!
    Score: 1.0
    Processed: this is n't great and does n't work
    Score: 1.0
Conclusion:
    Model cannot handle negations when "n't" is used!
'''

print("------------ Testing Case 2_2 ------------")
# Testing exceptions on the entire dataset

# Loading product reviews for sentiment analysis
df = pd.read_csv("amazon_product_reviews.csv")
# Removing empty lines
reviews_data = df['reviews.text'].dropna().reset_index(drop=True)

errors_count = 0
# Reduce to 400 for preliminary testing
num_reviews = len(reviews_data)

# Record start time.
start_time = time.time()

for i in range (0, num_reviews):
    review = reviews_data[i]
    original = sentiment_no_filtering(review)
    filtered = predict_sentiment(review)
    if original * filtered < 0:
        errors_count += 1
# Checking how different are the processed and original texts.
# Comment out for testing of no. of errors only and not error type.
#        doc_filtered = removing_stop_words_except(review)
#        print(f"------------------ Review no. {i} ------------------")
#        print(f"-> Orginal sentence:\n{review}\n* Score: {original}")
#        print(f"-> Processed sentence:\n{doc_filtered}\n* Score: {filtered}")
        
print(f"------ Number of errors: {errors_count}/{num_reviews} ------")

end_time = time.time()  # Record end time
elapsed_time = end_time - start_time
print(f"Total computing time: {elapsed_time} seconds")

'''
Output:
* for expections = {"no", "not"}
    - Computing time: 1168 seconds
    - Number of errors: 552/34,659 reviews
* for all expections:
    - Computing time: 1003 seconds
    - Number of errors: 228/34,659 reviews

Conclusions:
* More erros detected than in Testing_text_preprocessing.py because now cases
    where punctuation changes polarity's sign are also considered.
* ~Half of polarity scores are better assessed without stop words than using
    the orignal sentence, see sentiment_analysis_report.
* In general, the function predict_sentiment is now reliable enough to be used,
    but cannot handle texts with many spelling mistakes.
'''