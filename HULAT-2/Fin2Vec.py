# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from zipfile import ZipFile 
import codecs
import torch 
import re
from collections import Counter
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


#Load and extract all Gold Summaries 

zip_name = 'gold_summaries.zip'
with ZipFile(zip_name, 'r') as zip: 
	file_names = zip.namelist()
	print('Extracting all the files now...') 
	zip.extractall() 
	print('Done!\n') 

#Load  all summaries into the corpus
corpus = u""

print('Uploading raw data...')

for file_n in file_names[1:]:
	#print("Reading '{0}'...".format(file_n))
	with codecs.open(file_n, 'r', 'utf-8') as file:
		corpus += file.read()
	#print("Corpus is now '{0}' characters long".format(len(corpus)))
print('Done!')


def preprocess(text):
    text = text.lower()
    text = re.sub("'m", "million", text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    words = text.split()
    
    # Remove all words with  5 or fewer occurences
    
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > 5]

    return trimmed_words


words = preprocess(corpus)
print(words[:30])

print("Total words in Gold Summaries: {}".format(len(words)))
print("Unique words in Gold Summaries: {}".format(len(set(words))))

"""# Dictionaries"""

def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: Two dictionaries, vocab_to_int, int_to_vocab
    """
    word_counts = Counter(words)
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create int_to_vocab dictionaries
    int_to_vocab = {i: word for i, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: i for i, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

vocab_to_int, int_to_vocab = create_lookup_tables(words)
int_words = [vocab_to_int[word] for word in words]

print(int_words[:30])

"""# Subsampling"""

import random 
import numpy as np

threshold = 1e-5
word_counts = Counter(int_words)

total_words = len(words)

freqs = {word : count/total_words for word, count in word_counts.items()}
drop_probability = {word: 1 - np.sqrt(threshold/(freqs[word])) for word in word_counts}

train_words = [word for word in int_words if random.random() < (1 - drop_probability[word])] 

print(train_words[:30])

"""# Making batches"""

def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''
    
    R = np.random.randint(1, window_size+1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = words[start:idx] + words[idx+1:stop+1]
    
    return list(target_words)

"""# Generating Batches"""

def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''
    
    n_batches = len(words)//batch_size
    
    # only full batches
    words = words[:n_batches*batch_size]
    
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for i in range(len(batch)):
            batch_x = batch[i]
            batch_y = get_target(batch, i, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield x, y

int_text = [i for i in range(20)]
x,y = next(get_batches(int_text, batch_size=4, window_size=5))

print('x\n', x)
print('y\n', y)

"""# Validation"""

def cosine_similarity(embedding, valid_size=16, valid_window=100, device='cpu'):
    """ Returns the cosine similarity of validation words with words in the embedding matrix.
        Here, embedding should be a PyTorch embedding module.
    """
    
    # Here we're calculating the cosine similarity between some random words and 
    # our embedding vectors. With the similarities, we can look at what words are
    # close to our random words.
    
    # sim = (a . b) / |a||b|
    
    embed_vectors = embedding.weight
    
    # magnitude of embedding vectors, |b|
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)
    
    # pick N words from our ranges (0,window) and (1000,1000+window). lower id implies more frequent 
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000,1000+valid_window), valid_size//2))
    valid_examples = torch.LongTensor(valid_examples).to(device)
    
    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t())/magnitudes
        
    return valid_examples, similarities

"""# Model"""

import torch
from torch import nn
import torch.optim as optim

class SkipGram(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()
        
        self.embed = nn.Embedding(n_vocab, n_embed)
        self.output = nn.Linear(n_embed, n_vocab)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.embed(x)
        scores = self.output(x)
        log_ps = self.log_softmax(scores)
        
        return log_ps

# check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

embedding_dim = 300 
print("Vocab size: {}".format(len(vocab_to_int)))
model = SkipGram(len(vocab_to_int), embedding_dim).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

print_every = 500
steps = 0
epochs = 5 

# Training
for e in range(epochs):
    
    running_loss = 0.0

    # get input and target batches
    for inputs, targets in get_batches(train_words, 512):
        
        steps += 1
        inputs, targets = torch.LongTensor(inputs), torch.LongTensor(targets)
        inputs, targets = inputs.to(device), targets.to(device)
        
        log_ps = model(inputs)
        loss = criterion(log_ps, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:                  

            print('Epoch  ----->    [%d, %5d] loss: %.3f' % (e + 1, steps, running_loss / 2000))
            running_loss = 0.0
            # getting examples and similarities      
            valid_examples, valid_similarities = cosine_similarity(model.embed, device=device)
            _, closest_idxs = valid_similarities.topk(6) # topk highest similarities
            
            valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
            for ii, valid_idx in enumerate(valid_examples):
                closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
            print("...")

# get embeddings from the model's embedding layer
embeddings = model.embed.weight.to('cpu').data.numpy()

word_embeddings = {}

for i in range(len(int_to_vocab)):
    word = int_to_vocab[i]
    coefs = embeddings[i]
    word_embeddings[word] = coefs

import pickle
pickle_out = open("fin2vec_300d.pickle", "wb")
pickle.dump(word_embeddings, pickle_out)
pickle_out.close()

