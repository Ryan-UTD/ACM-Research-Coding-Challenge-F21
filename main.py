# Import libraries and download necessary resources for text processing
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download() # Only needs to be downloaded once

# Read the entire input file into one string, and convert to lowercase
with open('input.txt', 'r') as file:
    text = file.read().replace('\n', ' ').lower()

# Separate the entire text into sentences
sentences = nltk.sent_tokenize(text)

# Perform stemming AND remove stopwords in each sentence
stemmer = PorterStemmer()
stemmed_sentences = ['' for _ in range(len(sentences))]
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    stemmed_sentences[i] = ' '.join(words)

# Combine the stemmed sentences into the three paragraphs
paragraphs = [' '.join(sentences[:14]), ' '.join(sentences[14:25]), ' '.join(sentences[25:])]
stemmed_paragraphs = [' '.join(stemmed_sentences[:14]), ' '.join(stemmed_sentences[14:25]),
                      ' '.join(stemmed_sentences[25:])]

# --- Perform sentiment analysis using pretrained analyzers ---

analysis1 = SentimentIntensityAnalyzer()

# Create dataframe of each paragraph's sentiment scores using each method

scores1 = [analysis1.polarity_scores(stemmed_paragraphs[i])['compound'] for i in range(len(stemmed_paragraphs))]
scores2 = [TextBlob(stemmed_paragraphs[i]).sentiment for i in range(len(stemmed_paragraphs))]
# Round the scores to 2 digits
scores1 = [round(num, 2) for num in scores1]
for i in range(len(scores2)):
    scores2[i] = [round(num, 2) for num in scores2[i]]
# Combine the scores into a dataframe
paragraph_scores = pd.DataFrame([scores1, scores2[0], scores2[1]]).transpose()
# Make the first column of the dataframe the paragraph number
paragraph_scores.columns = ['Vader Sentiment', 'TextBlob Sentiment', 'TextBlob Subjectivity']
paragraph_scores['Paragraph Number'] = paragraph_scores.index + 1
paragraph_scores = paragraph_scores[['Paragraph Number'] + [col for col in paragraph_scores.columns if
                                                            col != 'Paragraph Number']]
print('Polarity scores of each paragraph:\n', paragraph_scores)

# Create dataframe of Paragraph 1's scores by sentence

p1_scores1 = [analysis1.polarity_scores(stemmed_sentences[i])['compound'] for i in range(14)]
p1_scores2 = [TextBlob(stemmed_sentences[i]).sentiment for i in range(14)]
# Round the scores to 2 digits
p1_scores1 = [round(num, 2) for num in p1_scores1]
for i in range(len(p1_scores2)):
    p1_scores2[i] = [round(num, 2) for num in p1_scores2[i]]
p1_scores = pd.DataFrame([p1_scores1, p1_scores2[0], p1_scores2[1]]).transpose()
p1_scores.columns = ['Vader Sentiment', 'TextBlob Sentiment', 'TextBlob Subjectivity']
# Make the first column of the dataframe the sentence number
p1_scores['Sentence Number'] = p1_scores.index + 1
p1_scores = p1_scores[['Sentence Number'] + [col for col in p1_scores.columns if
                                              col != 'Sentence Number']]
print('\nPolarity scores of each sentence in Paragraph 1:\n', p1_scores)

# Create dataframe of Paragraph 2's scores by sentence

p2_scores1 = [analysis1.polarity_scores(stemmed_sentences[i])['compound'] for i in range(14,25)]
p2_scores2 = [TextBlob(stemmed_sentences[i]).sentiment for i in range(14,25)]
# Round the scores to 2 digits
p2_scores1 = [round(num, 2) for num in p2_scores1]
for i in range(len(p2_scores2)):
    p2_scores2[i] = [round(num, 2) for num in p2_scores2[i]]
p2_scores = pd.DataFrame([p2_scores1, p2_scores2[0], p2_scores2[1]]).transpose()
p2_scores.columns = ['Vader Sentiment', 'TextBlob Sentiment', 'TextBlob Subjectivity']
# Make the first column of the dataframe the sentence number
p2_scores['Sentence Number'] = p2_scores.index + 1
p2_scores = p2_scores[['Sentence Number'] + [col for col in p2_scores.columns if
                                              col != 'Sentence Number']]
print('\nPolarity scores of each sentence in Paragraph 2:\n', p2_scores)

# Create dataframe of Paragraph 3's scores by sentence

p3_scores1 = [analysis1.polarity_scores(stemmed_sentences[i])['compound'] for i in range(25, len(stemmed_sentences))]
p3_scores2 = [TextBlob(stemmed_sentences[i]).sentiment for i in range(25, len(stemmed_sentences))]
# Round the scores to 2 digits
p3_scores1 = [round(num, 2) for num in p3_scores1]
for i in range(len(p3_scores2)):
    p3_scores2[i] = [round(num, 2) for num in p3_scores2[i]]
p3_scores = pd.DataFrame([p3_scores1, p3_scores2[0], p3_scores2[1]]).transpose()
p3_scores.columns = ['Vader Sentiment', 'TextBlob Sentiment', 'TextBlob Subjectivity']
# Make the first column of the dataframe the sentence number
p3_scores['Sentence Number'] = p3_scores.index + 1
p3_scores = p3_scores[['Sentence Number'] + [col for col in p3_scores.columns if
                                              col != 'Sentence Number']]
print('\nPolarity scores of each sentence in Paragraph 3:\n', p3_scores)

# --- Make a final decision on the text's overall sentiment using the Vader analyzer ---
stemmed_text = ' '.join(stemmed_paragraphs)
overall_score = analysis1.polarity_scores(stemmed_text)['compound']
if overall_score <= -0.25:
    outcome = 'NEGATIVE'
elif -0.25 < overall_score < 0.25:
    outcome = 'NEUTRAL'
else:
    outcome = 'POSITIVE'

# Print the results
print("\nThe text's overall sentiment score using Vader is: {}\nThe sentiment value is: {}\n".format(outcome,
                                                                                                 overall_score))
# --- Print the number of words with sentiment <= -0.5, >= 0.5, and the words ---

words = nltk.word_tokenize(stemmed_text)
words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
words_sentiment = pd.DataFrame([analysis1.polarity_scores(words[i])['compound'] for i in range(len(words))])
numPositive = len(words_sentiment.loc[words_sentiment[0] >= 0.2])
numNegative = len(words_sentiment.loc[words_sentiment[0] <= -0.2])
print("There are {} words in the text with a Vader sentiment value >= 0.2.\n"
      "There are {} words in the text with a Vader sentiment value <= -0.2.".format(numPositive, numNegative))