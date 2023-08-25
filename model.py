import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from nn import Linear, BatchNorm1d, Tanh, Embedding, FlattenConsecutive, Sequential 

with open('names.txt', 'r') as file:
    words = file.read().splitlines()

chars = sorted(list(set(''.join(words))))
strToInt = {s:i+1 for i,s in enumerate(chars)}
strToInt['.'] = 0
intToStr = {i:s for s,i in strToInt.items()}
vocab_size = len(intToStr)

block_size = 8 #context length/ how many chars are taken as input to predict next char?

def build_dataset(words):
  X, Y = [], []

  for w in words:
    context = [0] * block_size
    for char in w + '.':
     ix = strToInt[char]
     X.append(context)
     Y.append(ix)
     context = context[1:] + [ix]
  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X,Y

#create training, dev, and test subsets of the dataset
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
