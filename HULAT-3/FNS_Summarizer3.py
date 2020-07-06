# -*- coding: utf-8 -*-

# **IMPORT LIBRARIES AND DOWNLOAD NECCESSARY PACKAGES**

import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import pickle
import sys
import time
import os
import ntpath

""" CHECK ARGUMENTS """

if len(sys.argv) == 1 :
  print("Missing path of the document to summarize")
  exit()

if len(sys.argv) == 2 :
  print("Missing path to store the files")
  exit()

if len(sys.argv) > 3 :
  print("Too many arguments. Indicate path of document to summarize and path to upload the summary")
  exit()


"""# **REPORT PREPROCESSING**"""

init_time = time.perf_counter()

#Save output path
new_dir_path = sys.argv[2]

f_name_path = sys.argv[1]
f = open(f_name_path, "r")
raw_report = f.read()

#Next line cleaning for sentence tokenization
clean_report = re.sub('\n', ' ', raw_report)
#Original sentence list for the final Summary output  
original_sentence_list = sent_tokenize(clean_report)
#print('\n-----\n'.join(sent for sent in original_sentence_list))

"""Non Narrative Sentences Cleaning"""

stopwords = list(STOP_WORDS)
sentence_list = []  #processed sentence list
pos_tagged_sentence_list = []
#print(len(original_sentence_list))

# Analyse each sentence
# Non narrative sentences will be excluded 

sentences_window = 150
i=0
non_nar_index = []
for sent in original_sentence_list:
  
  if len(sentence_list) == sentences_window:
    break
  sent_words = word_tokenize(sent)
  sent_tag = nltk.pos_tag(sent_words)
  tags = [tag[1] for tag in sent_tag]
  narrative = ['V' == elem[0] for elem in tags]

  # Non Narrative Sentences will be removed
  if not any(narrative):
    non_nar_index.append(i)
  # Narrative Sentences will be preprocessed

  else:    

    #Process keywords
    sent = sent.replace("’m", " million")
    # Remove punctuation
    sent = re.sub(r'[^a-zA-Z0-9]', ' ', sent)
    # Lowercase
    sent = sent.lower()   

    # Tokenize words
    sent_words = word_tokenize(sent)

    # Remove stopwords 
    sent_words = [w for w in sent_words if not w in stopwords and len(w) > 1]

    # Join words into clen processed asentence
    clean_sent = ' '.join([sent_words[i] for i in range(len(sent_words))])  
    
    if len(word_tokenize(clean_sent)) > 1:
      # Append non empty sentece 
      sentence_list.append(clean_sent)  
    else:
      non_nar_index.append(i) 
    
  i+=1

for i in range(len(non_nar_index)):
  #remove non narrtaive sentences from original sentence list
  removed = original_sentence_list.pop(non_nar_index[i] - i) 

original_sentence_list = original_sentence_list[:sentences_window]
#print(len(original_sentence_list), len(sentence_list))

# Define Keywords list (extracted manually from previous analysis made in the summary corpus)
'''keywords = ['financial statement',
             'annual report',
             'growth',
             'audit committee',
             'executive director',
             'chairman',
             'stock',
             'shareholder',
             'stockholder',
             'dividend',
             'share',
             'acquisition',
             'year end',
             'this year',
             'annual report account',
             'board of directors',
             'million',
             'account',
             'production',
             'revenue',
             'company',
             'business',
             'plc',
             'profit']'''

# **TEXTRANK**"""

# download pretrained GloVe word embeddings
#! wget http://nlp.stanford.edu/data/glove.6B.zip
#! unzip glove*.zip

# Extract word vectors
word_embeddings = {}

#print("Loading word embeddings...")
with open('word2vec.300d.GoldSum.pickle', 'rb') as fp:
      word_embeddings = pickle.load(fp) 

'''
f = open('glove.6B.300d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
'''
#print("Translating sentences into vectors...")

sentence_vectors = []
for sent in sentence_list:
  if len(sent) != 0:
    #non-recognized words will be zeros    
    vector = sum([word_embeddings.get(w, np.zeros((300,))) for w in sent.split()])/(len(sent.split())+0.001)
  else:
    vector = np.zeros((300,))  
  sentence_vectors.append(vector)

#print("Number of vectors --->   ", len(sentence_vectors))

# similarity matrix

#print("Creating similarity matrix...")
sim_mat = np.zeros([len(sentence_list), len(sentence_list)])

for i in range(len(sentence_list)):
  for j in range(len(sentence_list)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,300), sentence_vectors[j].reshape(1,300))[0,0]
  #if i % 50 == 0:
    #print("Sentence", i, "out of", len(sentence_list))

#print("Building Textrank scores...")

nx_graph = nx.from_numpy_array(sim_mat)
text_rank_scores = nx.pagerank(nx_graph)



"""# **FEATURES SCORES**"""

#print("Calculating Feature Matrix...")

# Keywords Score function
'''def keywords_scores(sentences): 
  scores = []
  for sentence in sentences:
    kws_in_sent = 0
    for i in range(len(keywords)):
      if keywords[i] in sentence:
        kws_in_sent += 1
    scores.append(kws_in_sent/len(word_tokenize(sentence)))
  return scores'''  


scores = [text_rank_scores[i] for i in range(len(sentence_list))]

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(original_sentence_list)), reverse=True)

"""# **SUMMARY GENERATION**"""

# Generate summary

summary = ''
summarized_sentences = []
words_count = 0
removed = 0
for i in range(len(ranked_sentences)):
  #print("Sent           --> ", i)
  summarized_sentences.append(ranked_sentences[i][1])  
  words_count += len(ranked_sentences[i][1].split())
  if words_count > 1000:
    # Only if the summary is between 900 to 1000 words it will end adding senteces
    if (words_count - len(ranked_sentences[i][1].split())) > 900:
      #print(words_count - len(ranked_sentences[i][1].split()))
      summary = '\n'.join(summarized_sentences[:i-1])
      break
    #print(len(summarized_sentences), "  Frase --> ", i)
    summarized_sentences.pop(i-removed)
    removed += 1
    words_count -= len(ranked_sentences[i][1].split())
#print("Summary length:    ", len(summary.split()))
#print('Sentences used:    ', len(summarized_sentences))

#Upload file into output directory 

#Get report name
f_name = ntpath.basename(f_name_path)
idx = f_name.find('.txt')
#Output summary formatted name: 1234_summary.txt
f_out_name = f_name[:idx] + '_summary' + f_name[idx:]
#Set destination path and save summary
sum_path = os.path.join(new_dir_path, f_out_name)
f= open(sum_path,"wb")
utf8_summary= summary.encode('utf8')
f.write(utf8_summary)
f.close()

final_time = time.perf_counter()
print(f"Created file: --->  %s  Summarization time:   {final_time - init_time:0.4f} seconds" % f_out_name)
