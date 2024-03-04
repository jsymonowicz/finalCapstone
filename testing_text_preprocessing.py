'''
----------- T21 - Capstone Project - NLP Applications ------------
1. This code tests how the removal of stop words, punctuation, and spaces
affects the polarity score of a review. The aim is to understand which type
of text preprocessing is suitable for preserving the true sentiment.
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
# Importing time library for getting computing time of testing
import time
# For natural language processing
import spacy
# For sentiment assessment
from spacytextblob.spacytextblob import SpacyTextBlob
# For dataset handling
import pandas as pd

# NLP model for assessing sentiment
nlp_sm = spacy.load('en_core_web_sm')
nlp_sm.add_pipe('spacytextblob')

# Tested function for sentiment analysis of a string
## Input: a review (format: string)
## Output: polarity score on a scale from -1 (most negative) to 1 (most positive)

# Stop words which should not be removed - see Testing Case 2
exceptions = {"not", "no"}

def predict_sentiment(review):
    
    # Ensuring correct input format
    if not isinstance(review, str):
        raise Exception("Input must be a string!")
        
    # Pre-processing of the input text
    # Removing capitalization and leading spaces
    review = review.lower().strip()
    # Processing review text using spaCy ‘en_core_web_sm’.
    doc = nlp_sm(review)
    # Filtering out punctuation and spaces.
    no_punct_nor_space = [word.text for word in doc if not word.is_punct and not word.is_space]
    doc_clean = nlp_sm(' '.join(no_punct_nor_space))
    
    # Filtering out stop words
    # Adding exceptions: stop words that should not be deleted - see Testing Case 2
    no_stop_words = [word.text for word in doc_clean if not (word.is_stop and word.text not in exceptions)]
    doc_clean_2 = nlp_sm(' '.join(no_stop_words))

    # Assessing review's sentiment polarity.
    polarity = doc_clean_2._.blob.polarity
    return polarity

# --------- Loading product reviews for testing of sentiment analysis ---------
df = pd.read_csv("amazon_product_reviews.csv")
# .dropna() removes empty lines and .reset_index(drop=True) adjusts indicies accordingly
reviews_data = df['reviews.text'].dropna().reset_index(drop=True)


# ----------------------------- Testing -----------------------------
# Influence of removing stop words and punctuation on the output

# Adding functions with/without stop words and punctuation for score comparison
# For code clarity, in these functions I removed most comments & input class check

# Assessing polarity of the original sentence without filtering
def sentiment_no_filtering(review):
    review = review.lower().strip()
    doc = nlp_sm(review)
    polarity = doc._.blob.polarity
    return polarity

# Assessing polarity when removing punctuation and spaces
def sentiment_no_punct(review):
    review = review.lower().strip()
    doc = nlp_sm(review)
    # Filtering out punctuation and spaces
    no_punct_nor_space = [word.text for word in doc if not word.is_punct and not word.is_space]
    doc_clean = nlp_sm(' '.join(no_punct_nor_space))
    polarity = doc_clean._.blob.polarity
    return polarity

# Assessing polarity when removing stop words
def sentiment_no_stop_words(review):
    review = review.lower().strip()
    doc = nlp_sm(review)
    # Filtering out stop words
    no_stop_words = [word.text for word in doc if not word.is_stop]
    doc_clean = nlp_sm(' '.join(no_stop_words))
    polarity = doc_clean._.blob.polarity
    return polarity

# Assessing polarity when removing punctuation, spaces, and stop words
def sentiment_no_punct_nor_stop_words(review):
    review = review.lower().strip()
    doc = nlp_sm(review)
    # Filtering out punctuation and spaces
    no_punct_nor_space = [word.text for word in doc if not word.is_punct and not word.is_space]
    doc_clean = nlp_sm(' '.join(no_punct_nor_space))
    # Filtering out stop words
    no_stop_words = [word.text for word in doc_clean if not word.is_stop]
    doc_clean_2 = nlp_sm(' '.join(no_stop_words))
    polarity = doc_clean_2._.blob.polarity
    return polarity

# Returing a sentence when removing stop words
def removing_stop_words(review):
    review = review.lower().strip()
    doc = nlp_sm(review)
    # Filtering out stop words
    no_stop_words = [word.text for word in doc if not word.is_stop]
    doc_clean = nlp_sm(' '.join(no_stop_words))
    return doc_clean

#print("\n------------- Testing Case 1 -------------")
# Checking whether removing stop words and punctuation affects review's polarity

# Record start time
start_time = time.time()

# Number of tested reviews. Change to 300 for initial testing
num_reviews = len(reviews_data)

# Storing polarity scores for sentences with/without stop words, punctuation and spaces
filtering_test = []

for i in range (0, num_reviews):
    no_filters = sentiment_no_filtering(reviews_data[i])
    no_stop_words = sentiment_no_stop_words(reviews_data[i])   
    no_punct = sentiment_no_punct(reviews_data[i])   
    no_punct_nor_stop_words = sentiment_no_punct_nor_stop_words(reviews_data[i])
    filtering_test.append([i, no_filters, no_stop_words, no_punct, no_punct_nor_stop_words])
#print(filtering_test) # Commented out after the inital manual check
'''
Conclusion from the manual study of the above 'filtering_test' list:
1. removing stop words significantly affects polarity's sign,
2. removing punctuation hardly affects polarity score,
3. when removing both stop words and punctuation, change of polarity's sign 
is triggered mostly by the lack of stop words.
'''

print("\n------------- Testing Case 2 -------------")
# Finding the cause of why removing stop words changes polarity sign

# Storing errors for further analysis
error_list = []
error_count = 0

for result in filtering_test:
    no_filter = result[1]
    no_stop_words = result[2]
    if no_filter * no_stop_words < 0:
        # Getting index of sentences with the detected error
        i = result[0]
        # Storing review's index and polarity score for the original text
        error_list.append([i, no_filter])
        error_count += 1
        # Checking how pre-processing altered review's text
        # Commented out to reduce computing time in Testing Case 3
#        doc_no_stop_words = removing_stop_words(reviews_data[i])
#        print(f"---------- No. {i} ----------")
#        print(f"-> Orginal sentence:\n{reviews_data[i]}\n* Score: {no_filter}")
#        print(f"-> Without stop words:\n{doc_no_stop_words}\n* Score: {no_stop_words}")
print(f"------ Number of errors: {error_count} ------")
'''
Conclusions: removing words "not" and "no" changes polarity score dramatically!
'''  

print("\n------------- Testing Case 3 -------------")
# Checking if exceptions {not, no} fix the error of polarity score sign change
# The original function "predict_sentiment" was updated to keep stop
# words "not" and "no" - see exceptions = {"not", "no"}

# Testing function outputing text after removing stop words with exceptions.
def removing_stop_words_except(review):
    review = review.lower().strip()
    doc = nlp_sm(review)
    no_punct_nor_space = [word.text for word in doc if not word.is_punct and not word.is_space]
    doc_clean = nlp_sm(' '.join(no_punct_nor_space))
    # Adding stop words exceptions that should not be deleted
    no_stop_words = [word.text for word in doc_clean if not (word.is_stop and word.text not in exceptions)]
    doc_clean_2 = nlp_sm(' '.join(no_stop_words))
    return(doc_clean_2)

# Counting number of error after adding "not" and "no" as exceptions
error_count_fixed = 0

for error in error_list:
    i = error[0]
    no_filter = error[1]
    review = reviews_data[i]
    fixed = predict_sentiment(review)
    if no_filter * fixed < 0:
        error_count_fixed += 1
        # Checking which reviews are still affected by the error
        doc_fixed = removing_stop_words_except(review)
        print(f"------- Error remains in review {i} -------")
        print(f"-> Orginal sentence:\n{review}\n* Score: {no_filter}")
        print(f"-> Processed sentence:\n{doc_fixed}\n* Score: {fixed}")
        
print(f"------ Number of errors after fix: {error_count_fixed} ------")

# Printing processing time
end_time = time.time()  # Record end time
elapsed_time = end_time - start_time
print(f"\nTotal computing time: {elapsed_time} seconds")
'''
Output:
* Computing time: 2362 seconds,
* Exceptions "not", "no" reduce error count from 564 to 356 (34,659 reviews).
Conclusions: 
1. Adding exceptions "not", "no" improves function's performance
2. After the fix, discrepancies in polarity scores are much smaller (+/-0.25)
3. The error remains in reviews with spelling mistakes (not fixable). 
4. Testing Case 3 only considers cases where the polarity sign is modified by
the removal of stop words.To account for all errors and more exceptions for
stop words, see Testing_exception_stop_words.py
'''