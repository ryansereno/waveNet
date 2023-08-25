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


n_embd = 24 # the dimensionality of the character embedding vectors
n_hidden = 128 # the number of neurons in the hidden layer of the MLP

model = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(           n_hidden, vocab_size)
])

with torch.no_grad():
  #modify weights of last layer, to make it less confident/ more creative
  model.layers[-1].weight *= 0.1

parameters = model.parameters()
print("Total params: ", sum(p.nelement() for p in parameters)) # total number of parameters
for p in parameters:
  p.requires_grad = True

max_steps = 200_000
batch_size = 32

for i in range(max_steps):
  # construct minibatch
  ix = torch.randint(0, Xtr.shape[0], (batch_size,))
  Xb, Yb, = Xtr[ix], Ytr[ix]

  #forward pass
  logits = model(Xb)
  loss = F.cross_entropy(logits, Yb) #loss function

  #backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  #update
  lr = 0.1 if i < 150_000 else 0.01 # step learning rate decay
  for p in parameters:
    p.data += -lr * p.grad

  #track stats
  if i % 10_000 == 0:
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')

#switch to inference
for layer in model.layers:
  layer.training = False

#sample from model

for _ in range(20):
  out = []
  context = [0] * block_size
  while True:
    logits = model(torch.tensor([context]))
    probs = F.softmax(logits, dim=1)
    #sample from distrobution
    ix = torch.multinomial(probs, num_samples=1).item()
    #shift context window and record sample
    context = context[1:] + [ix]
    out.append(ix)
    #if end token, break
    if ix == 0:
      break

  print(''.join(intToStr[i] for i in out))
