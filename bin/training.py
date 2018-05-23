import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from data.data import Vocabulary, UnicodeCharsVocabulary, Batcher
from modules.model import Elmo
torch.set_default_tensor_type('torch.cuda.FloatTensor')
DTYPE = 'float32'
DTYPE_INT = 'int64'

options_file = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

def batch_to_ids(lm_vocab_file, batch, max_sentence_length, max_token_length, with_bos_eos):
    """
    Converts a batch of tokenized sentences to a tensor
    representing the sentences with encoded characters
    (len(batch), max sentence length, max word length).
    Parameters
    ----------
    batch : ``List[List[str]]``, required
        A list of tokenized sentences.
    Returns
    -------
        A tensor of padded character ids.
    """
    data = Batcher(lm_vocab_file, max_token_length, max_sentence_length).batch_sentences(batch, with_bos_eos)
    return torch.cuda.FloatTensor(data)

sentences, targets = [], []
with open ('sentences.small.train') as f:
    for line in f:
        sentence, t = [], []
        token_targets = line.lower().rstrip().split(' ')
        for token_target in token_targets:
            unpack = token_target.split('/')
            token, target = unpack[0], unpack[-1]
            sentence.append(token)
            t.append(target)
            sentences.append(sentence)
            targets.append(t)

num_Elmo_layers = 2
max_sentence_length = 50
batch_size = 32
class Net(nn.Module):
    def __init__(self, max_sentence_length, num_Elmo_layers, num_class):
        super(Net, self).__init__()
        self.max_sentence_length = max_sentence_length
        self.num_Elmo_layers = num_Elmo_layers
        self.num_class = num_class
        self.fc = nn.Linear(1024*self.num_Elmo_layers, self.num_class)

    def forward(self, inputs):
        x = self.fc(inputs_reshape)
        x = nn.softmax(x)
        return x
net = Net(237)
net.cuba()


vocal = "vocab-2016-09-10.txt"
character_ids = batch_to_ids("vocab-2016-09-10.txt", sentences, max_sentence_length, 50, with_bos_eos=False)
elmo = Elmo(options_file, weight_file, num_Elmo_layers, dropout=0)
embeddings = elmo(character_ids[:2000])
tgs = targets[:2000]
tgs_label = range(len(set(tgs)))
tgs_ =  []
for tg in tgs:
    if len(tg) <= 50 :
        tgs_ += tg[:50]
    else:
        tgs_ += tg
tgs_dict = {}
for l in tgs_:
    if l not in tgs_dict:
        tgs_dict[l] =  tgs_label.next()
tgs_idx= [tgs_dict[t] for t in tgs_]
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
tokens = torch.cat(embeddings['elmo_representations'])
mask = embeddings['mask']
tokens = tokens.view(-1, 1024*num_Elmo_layers)
mask = mask.view(-1)
inputs = torch.zeros([sum(mask), 1024*num_Elmo_layers], dtype = torch.float32)
i = 0
for idx, m in enumerate(mask):
    if m > 0:
        inputs[i] = tokens[idx]
        i += 1
inputs = inputs.view(batch_size, -1, 1024*num_Elmo_layers)
labels = torch.tensor(tgs_idx).view(batch_size, -1)

for epoch in range(50):
    running_loss = 0.0
    for data,label in zip(inputs, labels):
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
