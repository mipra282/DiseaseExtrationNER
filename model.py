# -*- coding: utf-8 -*-

### Ref - https://appliedmachinelearning.blog/2019/04/01/training-deep-learning-based-named-entity-recognition-from-scratch-disease-extraction-hackathon/
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import unicodedata
 
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense
from keras.layers import TimeDistributed, Dropout, Bidirectional


# Defining Constants
 
# Maximum length of text sentences
MAXLEN = 180
# Number of LSTM units
LSTM_N = 150
# batch size
BS=48

# Reading the training set
data = pd.read_csv("dataset/train.csv", encoding="latin1")
print(data.head(10))


# Reading the test set
test_data = pd.read_csv("dataset/test.csv", encoding="latin1")
test_data.head(10)

print("Number of uniques docs, sentences and words in Training set:\n",data.nunique())
print("\nNumber of uniques docs, sentences and words in Test set:\n",test_data.nunique())
 
# Creating a vocabulary
words = list(set(data["Word"].append(test_data["Word"]).values))
words.append("ENDPAD")
 
# Converting greek characters to ASCII characters eg. 'naïve café' to 'naive cafe'
words = [unicodedata.normalize('NFKD', str(w)).encode('ascii','ignore') for w in words]
n_words = len(words)
print("\nLength of vocabulary = ",n_words)
 
tags = list(set(data["tag"].values))
n_tags = len(tags)
print("\nnumber of tags = ",n_tags)
 
# Creating words to indices dictionary.
word2idx = {w: i for i, w in enumerate(words)}
# Creating tags to indices dictionary.
tag2idx = {t: i for i, t in enumerate(tags)}



def get_tagged_sentences(data):
    '''
    Objective: To get list of sentences along with labelled tags.
    Returns a list of lists of (word,tag) tuples.
    Each inner list contains a words of a sentence along with tags.
    '''
    agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(), s["tag"].values.tolist())]
    grouped = data.groupby("Sent_ID").apply(agg_func)
    sentences = [s for s in grouped]
    return sentences

def get_test_sentences(data):
    '''
    Objective: To get list of sentences.
    Returns a list of lists of words.
    Each inner list contains a words of a sentence.
    '''
 
    agg_func = lambda s: [w for w in s["Word"].values.tolist()]
    grouped = data.groupby("Sent_ID").apply(agg_func)
    sentences = [s for s in grouped]
    return sentences


# Getting training sentences in a list
sentences = get_tagged_sentences(data)
print("First 2 sentences in a word list format:\n",sentences[0:2])

# Getting test sentences in a list
test_sentences = get_test_sentences(test_data)
print("First 2 sentences in a word list format:\n",test_sentences[0:2])

# Converting words to indices for test sentences (Features)
# Converting greek characters to ASCII characters in train set eg. 'naïve café' to 'naive cafe'
X = [[word2idx[unicodedata.normalize('NFKD', str(w[0])).encode('ascii','ignore')] for w in s] for s in sentences]


# Converting words to indices for test sentences (Features)
# Converting greek characters to ASCII characters in test-set eg. 'naïve café' to 'naive cafe'
X_test = [[word2idx[unicodedata.normalize('NFKD', str(w)).encode('ascii','ignore')] for w in s] for s in test_sentences]
 
'''
Padding train and test sentences to 180 words.
Sentences of length greater than 180 words are truncated.
Sentences of length less than 180 words are padded with a high value.
'''
X = pad_sequences(maxlen=MAXLEN, sequences=X, padding="post", value=n_words - 1)
X_test = pad_sequences(maxlen=MAXLEN, sequences=X_test, padding="post", value=n_words - 1)
 
# Converting tags to indices for test sentences (labels)
y = [[tag2idx[w[1]] for w in s] for s in sentences]
# Padding tag labels to 180 words.
y = pad_sequences(maxlen=MAXLEN, sequences=y, padding="post", value=tag2idx["O"])
 
# Making labels in one hot encoded form for DL model
y = [to_categorical(i, num_classes=n_tags) for i in y]



# 180 dimensional word indices as input
input = Input(shape=(MAXLEN,))
 
# Embedding layer of same length output (180 dim embedding will be generated)
model = Embedding(input_dim=n_words, output_dim=MAXLEN, input_length=MAXLEN)(input)
 
# Adding dropout layer
model = Dropout(0.2)(model)
 
# Bidirectional LSTM to learn from both forward as well as backward context
model = Bidirectional(LSTM(units=LSTM_N, return_sequences=True, recurrent_dropout=0.1))(model)
 
# Adding a TimeDistributedDense, to applying a Dense layer on each 180 timesteps
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model) # softmax output layer
model = Model(input, out)
 
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X, np.array(y), batch_size=BS, epochs=2, validation_split=0.05, verbose=1)


# Predicting on trained model
pred = model.predict(X_test)
print("Predicted Probabilities on Test Set:\n",pred.shape)
# taking tag class with maximum probability
pred_index = np.argmax(pred, axis=-1)
print("Predicted tag indices: \n",pred_index.shape)


# Flatten both the features and predicted tags for submission
ids,tagids = X_test.flatten().tolist(), pred_index.flatten().tolist()
 
# converting each word indices back to words
words_test = [words[ind].decode('utf-8') for ind in ids]
# converting each predicted tag indices back to tags
tags_test = [tags[ind] for ind in tagids]
print("Length of words in Padded test set:",len(words_test))
print("Length of tags in Padded test set:",len(tags_test))
print("\nCheck few of words and predicted tags:\n",words_test[:10],tags_test[:10])


'''
The task here is to convert padded fixed 180 dimensional predicted tags
to variable length test set sentences.
1. If the sentences have word length shorter than 180,
   then predcited tags are skipped.
2. If the sentences have word length longer than 180,
   then all extra words are tagged with "O" tag class.
'''
 
i=0
j=1
predicted_tags = []
counts = test_data.groupby('Sent_ID')['id'].count().tolist()
 
for index,count in enumerate(counts):
    if count <= MAXLEN:
        predicted_tags.append(tags_test[i:i+count])
    else:
        predicted_tags.append(tags_test[i:i+MAXLEN])
        out = ['O']*(count-MAXLEN)
        predicted_tags.append(out)
 
    i=j*MAXLEN
    j=j+1
 
predictions_final = [item for sublist in predicted_tags for item in sublist]
print("\nLength of test set words and predicted tags should match.")
print("Length of predicted tags:",len(predictions_final))
print("Length of words in test set:",test_data['Word'].size)


df = pd.read_csv("sample_submission.csv", encoding="latin1")
# Creating a dataframe in the submission format
df_results = pd.DataFrame({'id':df['id'],'Sent_ID':df['Sent_ID'],'tag':predictions_final})
# writing csv submission file
df_results.to_csv('submission_final.csv',sep=",", index=None)
df_results.head()

# Relaxed/Partial F1 score on private leaderboard was 77.8%
# Partial F1 score: F1 score with considering partial disease name detection
from IPython.display import Image
Image(filename='/home/abhijeet/Pictures/F1_score.png')





