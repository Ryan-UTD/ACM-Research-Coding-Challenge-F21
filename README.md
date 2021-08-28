# Ryan Sharp: ACM Research Coding Challenge (Fall 2021)
## Introduction
Hello and thank you for checking out my project!
In this program, I use two pretrained sentiment analysis tools in Python to create sentiment scores of the given input. I then compare the two scores to each other and to what I would have expected. Let's dive in!

## Note Regarding my Submission
I see that the "promptness of your submission" is considered when grading the assingment. I was on vacation when I received the invitation to complete this challenge, and I did not have access to my computer during the vacation. Now that I am back home, I can finally work on the project. Thank you for understanding!

##  How to Approach the Challenge
The biggest challenge with performing sentiment analysis on this text is we have no labeled, or training, data. This means we don't have any data to teach our model what text is positive and what is negative. To evaluate this text, then, a good option is to import a tool that has been "pretrained" or already has a list of sentiment scores for certain words. These tools examine the sentiment score of each word in a text and then calculate the text's overall score. The tools I chose are Vader from NLTK and TextBlob.

## About Vader and TextBlob
These two pretrained tools have a table of predefined words and their sentiment score from -1 (most negative) to +1 (most positive). These tools then utilize a "bag of words" approach, meaning they turn the text into a vector, each dimension being the number of occurences of a certain word in the text. Using a bag of words for sentiment analysis is simple and can be effective, but it does not know the sentiment of words outside of its vocabulary. In addition, it can't take into account the sentiment created from long sequences of words. Vader can notice the sentiment in short sequences of words, such as "very pleasant" and "not pleasant," but it can't identify sarcasm or other elements of language that can only be seen by looking deeply into a long sequence of words.
A pretrained tool that can identify sentiment in longer sequences of words is Flair, but I was having difficulties with successfully installing it. If I had the time to look deeper into this project, I would test out tools like Flair that use a LSTM (long short-term memory) neural network to detect more complext sentiment in natural language.

## Preprocessing the Text
Before we implement the tools Vader and TextBlob, I removed words known as "stopwords" that are used so often in language that they have no use for sentiment analysis. These are words such as "the," "a," "and," "is," and many more. The programmer can define his/her own list of stopwords to remove from the data, or he/she can use a previously created list of stopwords, which is what I did for this project. Removing these unnecessary words will make our data more concise and efficient, increasing the "usefulness per unit of data" that we will use for analysis.
In addition to removing stopwords, I performed stemming on the remaining words. This means removing the last few characters of certain words to reduce them to their fundamental form. For example, the words "gone," "goes" and "going" reduce to "go." This further increases the simplicity and efficiency of our model.
Now that useless words have been removed, and our words have been reduced to their fundamental forms, we are ready to use Vader and TextBlob for sentiment analysis.

## Results of my Project
**The overall sentiment score of the text is 0.9949, meaning extremely positive.** I calculated this score using Vader over TextBlob because TextBlob was not performing ideally (returning some NaN values), and the results that TextBlob did give for each sentence were much further off from the sentiment score I expected.
**This contrasts from what I expected** because the first two paragraphs, especially the second, contained words that I would expect to pull the score further in the negative direction. The first paragraph contained aggressive words such as "stop," "watch out," and "dreary chaos." The second paragraph contained frightening words like "devil," "screamed" and "murderer." However, the last paragraph contains many positive words that brought up the score, such as "pleasing," "ingenious" and "excellent." Becuase the last paragraph is so long and full of positive words, it brought up the overall score.
Another interesting result from my project is TextBlob also gives a subjectivity score, indicating how biased it thinks the text is. TextBlob gave the second paragraph a subjectivity score of 0.61, perhaps because of the amount of emotion and reactions described in that text.

## Pros of my Analysis Method
1. Removing the stopwords and performing the stemming makes our data more compact, efficient to use, and dense with usefulness.
2. My program runs quickly because Vader and TextBlob do not implement a complex machine learning algorithm such as a neural network.
3. Vader and TextBlob can be used on any sentence you wish to create, without the need for "training data" or model building.
## Cons of my Analysis Method
1. Vader and TextBlob cannot identify complex sentiment (such as sarcasm) in natural langauge that can only be found in long sequences of words.
2. TextBlob cannot identify negations in language, such as "not good" or "won't hurt," although Vader can.
3. TextBlob and Vader cannot assign a sentiment score to any word that is out of their vocabulary.
4. In my program, TextBlob gave many NaN answers to the sentiment and subjectivity scores, which indicates I'm using it incorrectly or it might not recognize any words in the sentences.
## Next Steps
1. Learn how to use Flair and other algorithms that use a LSTM (long short-term memory) neural network to identify more complex sentiment in natural language
2. Debug my implementation of TextBlob so it stops returning NaN values, and find out why it was giving sentiment scores further off from the scores I expected.

Thank you for reading about my project!

## Libraries Used
NLTK, Pandas, TextBlob

## Sources
1. Using Vader, TextBlob and Flair: https://medium.com/@b.terryjack/nlp-pre-trained-sentiment-analysis-1eb52a9d742c
2. Performing stemming and removing stopwords: https://www.youtube.com/watch?v=1OMmbtVmmbg&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=4
