from data.data import Vocabulary, UnicodeCharsVocabulary, Batcher
from modules.model import Elmo

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
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
vocab = set([tk for sent in sentences for tk in sent] +['<S>', '</S>', '<UNK>', '!!!MAXTERMID'])
with open ("vocab.txt", "w") as f:
    f.write('\n'.join(vocab))

num_Elmo_layers = 2
max_sentence_length = 50
batch_size = 16
class Net(nn.Module):
    def __init__(self, max_sentence_length, num_Elmo_layers, num_class):
        super(Net, self).__init__()
        self.max_sentence_length = max_sentence_length
        self.num_Elmo_layers = num_Elmo_layers
        self.num_class = num_class
        self.fc = nn.Linear(1024*self.num_Elmo_layers, self.num_class)

    def forward(self, inputs):
        x = self.fc(inputs)
        #x = nn.LogSoftmax(x)
        return x
net = Net(max_sentence_length, num_Elmo_layers, 237)
net.cuda()


character_ids = batch_to_ids("vocab.txt", sentences[:201], max_sentence_length, 50, with_bos_eos=False)
elmo = Elmo(options_file, weight_file, num_Elmo_layers, dropout=0)
embeddings = elmo(character_ids)
tgs = targets[:201]
tgs_label = list(range(len(set([tg for l in targets for tg in l]))))
tgs_ =  []
for tg in tgs:
    if len(tg) <= 50 :
        tgs_ += tg[:50]
    else:
        tgs_ += tg
tgs_dict = {}
for l in tgs_:
    if l not in tgs_dict:
        tgs_dict[l] = tgs_label.pop()
tgs_idx= [tgs_dict[tg] for tg in tgs_]
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
print (sum(mask))
inputs = inputs.view(batch_size, -1, 1024*num_Elmo_layers)
labels = torch.tensor(tgs_idx).view(batch_size, -1)

for epoch in range(50):
    running_loss = 0.0
    for data , label in zip(inputs, labels):
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, label)
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
    if epoch % 10 == 9:
        print('[%d] loss: %.3f' %(epoch + 1, running_loss))

print('Finished Training')
