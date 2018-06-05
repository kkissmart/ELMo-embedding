from data.data import Vocabulary, UnicodeCharsVocabulary, Batcher
from modules.model import Elmo

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
torch.set_default_tensor_type('torch.cuda.FloatTensor')


DTYPE = 'float32'
DTYPE_INT = 'int64'

options_file = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

train_file = "sentences.small.train"
validation_file = "sentences.small.dev"

def batch_to_ids(lm_vocab_file,
                 batch,
                 max_sentence_length,
                 max_token_length,
                 with_bos_eos):

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
    data = Batcher(lm_vocab_file,
                   max_token_length,
                   max_sentence_length).batch_sentences(batch, with_bos_eos)

    return torch.cuda.FloatTensor(data)

def parse_sentences_target(file_location):
    """
    Parse tain/validation file sentences into input and label.
    Parameters
    -------
    file_location: each line has one sentences only
                   with each token and its tag seperated by "/", required.
    Returns
    -------
        A dict with sentences: list[list[str]] length = # sentences,
        and variable length of each sentences
        targets: list[list[str]] same dimension as sentences
    """
    sentences, targets = [], []
    with open(file_location) as f:
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
    return {"sentences": sentences, "targets": targets}


training_data = parse_sentences_target(train_file)
sentences, targets = training_data["sentences"], training_data["targets"]
vocab = set([tk for sent in sentences for tk in sent] +['<S>', '</S>', '<UNK>', '!!!MAXTERMID'])
with open ("vocab.txt", "w") as f:
    f.write('\n'.join(vocab))

total_tags = set([tg for l in targets for tg in l])
tgs_dict = {}
for idx, tag in enumerate(list(total_tags)):
    tgs_dict[tag] = idx

num_Elmo_layers = 2
max_sentence_length = 50
batch_size = 64

class Net(nn.Module):
    def __init__(self, max_sentence_length, num_Elmo_layers, num_class):
        super(Net, self).__init__()
        self.max_sentence_length = max_sentence_length
        self.num_Elmo_layers = num_Elmo_layers
        self.num_class = num_class
        self.fc = nn.Linear(1024*self.num_Elmo_layers, self.num_class)

    def forward(self, inputs):
        x = self.fc(inputs)
        return x

net = Net(max_sentence_length, num_Elmo_layers, 237)
net.cuda()
elmo = Elmo(options_file, weight_file, num_Elmo_layers, dropout=0)
elmo_parameters = [p for p in elmo.parameters() if p.requires_grad]

optimizer = optim.SGD(list(net.parameters()) + elmo_parameters, lr=0.01, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

def get_targets_with_max_length(targets, max_sentence_length):

    tgs_ = []

    for tg in targets:
        if len(tg) <= max_sentence_length:
            tgs_ += [tgs_dict[tok] for tok in tg]
        else:
            tgs_ += [tgs_dict[tok] for tok in tg[:max_sentence_length]]

    return Variable(torch.cuda.LongTensor(tgs_))

for epoch in range(5):
    running_loss = 0.0
    total = 0.
    correct = 0.
    for i in range(len(sentences)//200):
        character_ids = batch_to_ids("vocab.txt", sentences[200*i : 200*(i+1)], max_sentence_length, 50, with_bos_eos=False)
        embeddings = elmo(character_ids)
        tgs = get_targets_with_max_length(targets[200*i : 200*(i+1)], max_sentence_length)
        tokens = Variable(torch.cat(embeddings['elmo_representations'], -1))
        mask = embeddings['mask']
        tokens = tokens.view(-1, 1024*num_Elmo_layers)
        mask = mask.view(-1)
        inputs = tokens.clone()
        j = 0
        for idx, m in enumerate(mask):
            if m > 0:
                inputs[j] = tokens[idx]
                j += 1
        inputs = inputs[:j]
        assert(inputs.shape[0] == tgs.shape[0]), "input and label have different shape"
        num_batchs = inputs.shape[0]//batch_size
        inputs = Variable(inputs[:num_batchs*batch_size].view(-1, batch_size, 1024*num_Elmo_layers))
        labels = Variable(tgs[:num_batchs*batch_size].view(-1, batch_size))
        for data , label in zip(inputs, labels):
            optimizer.zero_grad()
            outputs = net(data)
            _, predicts = torch.max(outputs, -1)
            assert(predicts.shape[0] == batch_size), "wrong shape"
            correct += (predicts == label).sum().item()
            total += batch_size
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    print('[%d] loss: %.3f acurracy: %.3f' %(epoch + 1, running_loss, correct * 1.0/total))

print('Finished Training')

test_data = parse_sentences_target(validation_file)
sentences_validation, targets_validation = test_data["sentences"], test_data["targets"]

total, correct =  0.0, 0.0
for i in range(len(sentences_validation)//10):
    character_ids = batch_to_ids("vocab.txt", sentences_validation[10*i : 10*(i+1)], max_sentence_length, 50, with_bos_eos=False)
    embeddings = elmo(character_ids)
    tgs = get_targets_with_max_length(targets_validation[10*i : 10*(i+1)], max_sentence_length)
    tokens = torch.cat(embeddings['elmo_representations'], -1)
    mask = embeddings['mask']
    tokens = tokens.view(-1, 1024*num_Elmo_layers)
    mask = mask.view(-1)
    inputs = tokens.clone()
    j = 0
    for idx, m in enumerate(mask):
        if m > 0:
            inputs[j] = tokens[idx]
            j += 1
    inputs = inputs[:j]
    assert(inputs.shape[0] == tgs.shape[0]), "input and label have different shape"
    outputs = net(inputs)
    _, predicts = torch.max(outputs, -1)
    correct += (predicts == tgs).sum().item()
    total += j
print ("validation result %.3f" %(correct*1.0/total))
