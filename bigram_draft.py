import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# total characters
# print("length of dataset in characters", len(text))

# all unique characters that occur in this dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)

# very simple tokenizer, and encode/decode functions
# tokenize, translating individual characters to integers
# we create two lookup tables: stoi, and itos
# string-to-index mapping stoi = {'a': 0, 'b': 1, 'c': 2}
stoi = {char: i for i, char in enumerate(chars)}
# index-to-string mapping itos = {0: 'a', 1: 'b', 2: 'c'}
itos = {i: char for i, char in enumerate(chars)}

# decode/encode functions
# encode: take a string, output list of indexes / integers


def encode(s): return [stoi[c] for c in s]
# decode: take a list of indexes / integers, return a string
def decode(l): return ''.join(itos[i] for i in l)

# print(encode('hello world!'))
# print(decode(encode('hello world!')))


# now as we have tokenizer, we can tokenize entire dataset.
# we gonna store it torch.tensor
data = torch.tensor(encode(text), dtype=torch.long)
# data = data.to('cuda') # We can make tensor run on GPU optionally for better performance
# torch.Size([1115393]), torch.int64
# print(data.shape, data.dtype)
# print(data[:100])

# now, let's split up the data into train and validation sets
n = int(0.9*len(data))  # first 90% of the data is going to be training data
train_data = data[:n]
# now lets define some validation data
val_data = data[n:]

# training data block_size that we will split our data into chunks,
# because it is very ineficient to feed all data at once
block_size = 8
# in a chunk of 9 characters there are 8 individual examples of sequeneces
# [17, 46, 55, 56, 57,  1, 14, 46, 57]
# for example in context of 17, 46 comes next, in context of 17, 46, 55 comes next and so on...
# print(train_data[:block_size+1])

# Take the first `block_size` tokens as input context
x = train_data[:block_size]
# Take the next `block_size` tokens as targets, offset by 1
y = train_data[1:block_size:+1]
for t in range(len(y)):
    # The context grows with each iteration (e.g., [17], [17, 46], ...)
    context = x[:t+1]
    target = y[t]  # The target is the next token to predict
    # print(f"when input is {context} the target: {target}")

# batching?

batch_size = 4  # how many independent sequences we will process in parallel?
block_size = 8  # what is the max context length for predictions?


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    # what split we are taking look at? training or validation?
    data = train_data if split == 'train' else val_data
    # random positions to grab a chunk from
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # stack up at rows one dimensional tensors
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('------')
# print('targets:')
# print(yb.shape)
# print(yb)
# print('-----')

for b in range(batch_size):  # batch dimension
    for t in range(block_size):  # time dimensions
        context = xb[b, :t+1]
        target = yb[b, t]
        # print(f"when input is {context.tolist()} the target: {target}")

# now, lets feed this into neural network
# print(xb)


class BigramLanguageModel(nn.Module):  # subclass of n module

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # nn.Embedding a thin wrapper of tensor of shape
        self.token_embedding_table = nn.Embedding(
            vocab_size, vocab_size)  # size of vocab size by vocab size

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        # pytroch arranges batch by time by channel tensor (4, 8, vocabsize or 65)
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            # loss (quality of predictions, use negative likelilhood loss
            # loss is the cross entropy on predictions and targets
            # reshape the logits for cross_entropy func (stretch array to two-dimensions)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss  # score for next character in the sequence

        # we predict what comes next by
        # take this (B by T) and make it +1, +2, basicalty continue generation
    def generate(self, idx, max_new_tokens):
            # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)

# generating some tokens from [0,0..] empty situtation, then convert tensor into python list and decode it
# print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
# it is gibbrish so lets now train the model

# create pytorch optimisation object, learning rate
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
# optimizer gonna take the gradients and optimise parameters
batch_size = 32
for steps in range(10000):
    # take a sample of batch data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    # zeroing out all the gradients from the previosu step
    optimizer.zero_grad(set_to_none=True)
    # getting gradients for all the params
    loss.backward()
    # updae our params
    optimizer.step()

print(loss.item())
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=300)[0].tolist()))