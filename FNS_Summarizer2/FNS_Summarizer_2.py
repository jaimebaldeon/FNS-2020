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
import pickle
import os
import sys
import ntpath
import time
import meaningcloud
#pip install MeaningCloud-python
np.seterr(divide='ignore', invalid='ignore')

"""	CHECK ARGUMENTS	"""

if len(sys.argv) == 1 :
  print("Missing path of the document to summarize")
  exit()

if len(sys.argv) == 2 :
  print("Missing path to store the files")
  exit()

if len(sys.argv) > 3 :
  print("Too many arguments. Indicate path of document to summarize and path to upload the summary")
  exit()

""" **REPORT PREPROCESSING**"""

init_time = time.perf_counter()

#Save output path
new_dir_path = sys.argv[2]

#Get file path and read report
f_name_path = sys.argv[1]
f = open(f_name_path, "r")
raw_report = f.read()

#Next line cleaning for sentence tokenization
clean_report = re.sub('\n', ' ', raw_report)
#Original sentence list for the final output Summary   
original_sentence_list = sent_tokenize(clean_report)
ner_report = ' '.join([original_sentence_list[i]] for i in range(50))
#print(len(original_sentence_list))

"""Non Narrative Sentences Cleaning"""


sentence_list = []  #processed sentence list
stopwords = list(STOP_WORDS)

# Analyse each sentence
# Non narrative sentences will be excluded 

i=0
non_nar_index = []
for sent in original_sentence_list:
  
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

    # Join clean words to build clean sentence
    clean_sent = ' '.join([sent_words[i] for i in range(len(sent_words))])  

    # Append sentece 
    sentence_list.append(clean_sent)  
        
  i+=1

for i in range(len(non_nar_index)):
  #remove non narrative sentences from original sentence list
  removed = original_sentence_list.pop(non_nar_index[i] - i) 

#print("Sentences length --->   ", len(sentence_list))
#print('\n-----\n'.join(sent for sent in sentence_list))
# Extract word vectors
word_embeddings = {}

#print("Loading word embeddings...")
with open('word2vec.300d.GoldSum.pickle', 'rb') as fp:
      word_embeddings = pickle.load(fp) 

#print("Translating sentences into vectors...")

# Extract word vectors
'''
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
'''

sentence_vectors = []
for sent in sentence_list:
  if len(sent) != 0:
    #non-recognized words will be zeros    
    vector = sum([word_embeddings.get(w, np.zeros((300,))) for w in sent.split()])/(len(sent.split())+0.001)
  else:
    vector = np.zeros((300,))  
  sentence_vectors.append(vector)

#print("Number of vectors --->   ", len(sentence_vectors))

"""# **FEATURE EXTRACTION**"""

""" meaning cloud topic extraction"""

#print("Extracting name entities with MeaningCloud...")


#!pip install meaningcloud-python

license_key = 'b59424e48a94e5061bef29cbd29bdacd'

# We are going to make a request to the Topics Extraction API
topics_response = meaningcloud.TopicsResponse(meaningcloud.TopicsRequest(license_key, txt=ner_report, lang='en',topicType='e').sendReq())

# If there are no errors in the request, we print the output
if topics_response.isSuccessful():
  #print("\nThe request to 'Topics Extraction' finished successfully!\n")

  entities = topics_response.getEntities()
  important_entities = []

  if entities:
      #print("\tEntities detected (" + str(len(entities)) + "):\n")
      for entity in entities:
        ent = topics_response.getTopicForm(entity).lower() #topics_response.getTopicRelevance(entity)                              
        important_entities.append(ent)
        if len(important_entities) == 20:
          break
      
  else:
      print("\tNo entities detected!\n")
else:
  if topics_response.getResponse() is None:
      print("\nOh no! The request sent did not return a Json\n")
  else:
      print("\nOh no! There was the following error: " + topics_response.getStatusMsg() + "\n")


# Define Keywords list (extracted manually from previous analysis made in the summary corpus)

keywords = ['financial statement',
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
             'profit']

# NER Meaningcloud API score function
def entity_scores(sentences): 
  scores = []
  for sentence in sentences:
    ent_in_sent = 0
    for i in range(len(important_entities)):
      if important_entities[i] in sentence:
        ent_in_sent += 1
    scores.append(ent_in_sent/len(word_tokenize(sentence)))
  return scores

# Lemmatization with POS Tagging 
def get_wordnet_pos(tag):
    """Map POS tag to first character lemmatize() accepts"""
    tag = tag[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# Keywords Score function
def keywords_scores(sentences): 
  scores = []  
  lemmatizer=WordNetLemmatizer()
  for sentence in sentences:
    sent_words = word_tokenize(sentence)
    # Part of Speech Tagging
    tagged = nltk.pos_tag(sent_words)
    tags = [tag[1] for tag in tagged] 
    # Lemmatization
    lemma_words = [lemmatizer.lemmatize(sent_words[i], get_wordnet_pos(tags[i])) for i in range(len(sent_words))]    
    clean_sent = ' '.join([lemma_words[i] for i in range(len(lemma_words))]) 
    kws_in_sent = 0
    for i in range(len(keywords)):
      if keywords[i] in clean_sent:
        kws_in_sent += 1
    scores.append(kws_in_sent/len(word_tokenize(sentence)))
  return scores

# Gold summaries similarity function
def gold_summaries_similarity_scores(sentence_vectors):
  scores = []
  #Load gold summaries vector
  with open('gold_summaries_300D_vectors.pickle', 'rb') as fp:
      summary_vector, summary_vectors = pickle.load(fp)  
  # SImilarity between each sentence and gold summaries
  for sent in range(len(sentence_vectors)):    
    similarity = cosine_similarity(sentence_vectors[sent].reshape(1,300), summary_vector.reshape(1,300))[0,0]
    scores.append(similarity)
  return scores

def position_scores(sentences):
  scores = []
  for i in range(len(sentences)):    
    if i < 15:
      scores.append(0.9-i/1000)
    elif i < 50:
      scores.append(0.5-i/1000)  
    else:
      scores.append(0.1)
  return scores


"""**MODELLING**"""

#print("Calculating Feature Matrix...")

featureMatrix = []	#scores matrix
# Append feature scores
featureMatrix.append(keywords_scores(sentence_list))
featureMatrix.append(entity_scores(sentence_list))
featureMatrix.append(position_scores(sentence_list))
featureMatrix.append(gold_summaries_similarity_scores(sentence_vectors))
# Feature Matrix normalization (scores between 0 and 1)
featureMat = np.zeros((len(sentence_list), len(featureMatrix)))
for i in range(len(featureMatrix)) :
    for j in range(len(sentence_list)):
        featureMat[j][i] = featureMatrix[i][j]	#transpose matrix
#print(featureMat.max(axis=0))
featureMat_normed = featureMat / featureMat.max(axis=0)
# Sum up feature scores to get the final score for each sentence
scores = [sum(featureMat_normed[i]) for i in range(len(sentence_list))]

#Rank original sentences to be extracted 
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(original_sentence_list)), reverse=True)


"""# **SUMMARY GENERATION**"""

summary = u''
summarized_sentences = []
words_count = 0	#summary length
removed = 0

#Extract ranked senteces by relevance
for i in range(len(ranked_sentences)):
  summarized_sentences.append(ranked_sentences[i][1])  
  words_count += len(ranked_sentences[i][1].split())
  if words_count > 1000:
    # When the summary is between 900 to 1000 words it is completed
    if (words_count - len(ranked_sentences[i][1].split())) > 900:
      #Create summary excluding last sentence added
      summary = '\n'.join(summarized_sentences[:i])	
      break
    #Discard last sentece and extract the next one in the list
    summarized_sentences.pop(i-removed)
    removed += 1
    words_count -= len(ranked_sentences[i][1].split())

#print("Summary length:    ", len(summary.split()))
#print(summary)

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
