# -*- coding: utf-8 -*-
from zipfile import ZipFile 
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import pickle


#Load and extract all Gold Summaries 

with ZipFile('gold_summaries.zip', 'r') as zip: 
	file_names = zip.namelist()
	print('Extracting all the files now...') 
	zip.extractall() 
	print('Done!\n') 

#Load  all summaries into the corpus
summaries_sentence_list = []

print('Uploading raw data...')
for file_n in file_names[1:]:
	with open(file_n, 'r') as file:
		file_data = file.read()
		file_data = re.sub('\n', ' ', file_data)
		sents = sent_tokenize(file_data)
		if len(sents) != 0:
			summaries_sentence_list.append(sents)
print('Done!')

"""**Preprocessing**"""

summaries_clean_sentences = []  #processed summaries sentence list

for summary_sentences in summaries_sentence_list:

  # Clean sentences in each summary

  clean_sentences = [re.sub("'m", 'million', s) for s in summary_sentences]
  clean_sentences = [re.sub(r'[^a-zA-Z0-9]', ' ', s) for s in clean_sentences]
  clean_sentences = [s.lower() for s in clean_sentences]
  
  # After preprocecessing all sentences, add clean summary to the corpus of all summaries
  summaries_clean_sentences.append(clean_sentences)

stopwords = list(STOP_WORDS)

def remove_stopwords(words_list):
  sen_new = " ".join([i for i in words_list if i not in stopwords])
  return sen_new

summary_idx = 0
for clean_sentences in summaries_clean_sentences:
  # remove stopwords from the sentences
  # ??????????? [remove_stopwords(sentence.split()) for sentence in clean_sentences if remove_stopwords(sentence.split()) > 0]
  clean_sentences = [remove_stopwords(sentence.split()) for sentence in clean_sentences]
  summaries_clean_sentences[summary_idx] = clean_sentences
  summary_idx+=1

"""**Word Embeddings**"""

# Extract word vectors
word_embeddings = {}
'''f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()'''

print("Loading word embeddings...")
with open('word2vec.300d.GoldSum.pickle', 'rb') as fp:
      word_embeddings = pickle.load(fp) 

summaries_sentence_vectors = []

for clean_sentences in summaries_clean_sentences:
  sentence_vectors = []
  for sentence in clean_sentences:
    if len(sentence) != 0:
      #non-recognized words will be zeros
      v = sum([word_embeddings.get(w, np.zeros((300,))) for w in sentence.split()])/(len(sentence.split())+0.001)
    else:
      v = np.zeros((300,))  
    sentence_vectors.append(v)
  summaries_sentence_vectors.append(sentence_vectors)

"""**Summary Vectors**"""

summary_vectors = []

for sentence_vectors in summaries_sentence_vectors:
    sum_v = sum([s for s in sentence_vectors])/(len(sentence_vectors))
    summary_vectors.append(sum_v)

"""**Final Summary Vector**"""

#Total summary vector
summary_vector = sum(summary_vectors)/(len(summary_vectors))
print(summary_vector)

pickle_out = open("gold_summaries_300D_vectors.pickle","wb")
pickle.dump((summary_vector, summary_vectors), pickle_out)
pickle_out.close()
