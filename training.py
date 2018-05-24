from data.data import Vocabulary, UnicodeCharsVocabulary, Batcher
from modules.model import Elmo

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
torch.set_default_tensor_type('torch.cuda.FloatTensor')


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


total_tags = set([tg for l in targets for tg in l])
tgs_dict = {}
for idx, tag in enumerate(list(total_tags)):
    tgs_dict[tag] = idx


def get_targets_with_max_length(targets, max_sentence_length):
    tgs_ = []
    for tg in targets:
        if len(tg) <= 50 :
            tgs_.append([tgs_dict[tok] for tok in tg] + [total_targets]* (50 -len(tg)))
        else:
            tgs_.append([tgs_dict[tok] for tok in tg[:50]])
    return torch.cuda.LongTensor(tgs_)


num_Elmo_layers = 2
max_sentence_length = 50
batch_size = 64
num_gpu = 4
max_load = 200
total_targets = 237
devices = list(range(min(num_gpu, torch.cuda.device_count())))
num_gpu = len(devices)


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


net = Net(max_sentence_length, num_Elmo_layers, total_targets+1)
net.cuda()
net_parameters = net.parameters()
net = nn.parallel.replicate(net, devices)
elmo = Elmo(options_file, weight_file, num_Elmo_layers, dropout=0)
elmo_parameters = [p for p in elmo.parameters() if p.requires_grad]
elmo = nn.parallel.replicate(elmo, devices)
criterion = torch.nn.CrossEntropyLoss()
criterion = nn.parallel.replicate(criterion, devices)
optimizer = optim.SGD(list(net_parameters)+list(elmo_parameters), lr=0.01, momentum=0.9)


labels = get_targets_with_max_length(targets, max_sentence_length)
num_rounds = int(len(sentences)) // max_load


for epoch in range(100):
    running_loss = 0.0
    correct = 0
    total = 0
    for k in range(num_rounds):
        character_ids = batch_to_ids("vocab.txt",
                                     sentences[max_load*k:max_load*(k+1)],
                                     max_sentence_length,
                                     50,
                                     with_bos_eos=False)

        labels = get_targets_with_max_length(targets[max_load*k:max_load*(k+1)],
                                             max_sentence_length)
        labels = nn.parallel.scatter(labels, devices)
        character_ids = nn.parallel.scatter(tuple([character_ids, ]), devices)
        embeddings = nn.parallel.parallel_apply(elmo, character_ids)
        tokens = [torch.cat(embeddings[i]["elmo_representations"], -1).view(-1, 1024*num_Elmo_layers) for i in devices]
        masks = [embeddings[i]["mask"].view(-1) for i in devices]
        num_samples = tokens[0].shape[0] // batch_size * batch_size
        inputs = [tokens[i][:num_samples] for i in devices]
        labels = [labels[i].view(-1)[:num_samples] for i in devices]
        masks = [masks[i].view(-1)[:num_samples] for i in devices]
        for round in range(num_samples // batch_size):
            input = [inputs[i][batch_size * round: batch_size*(round + 1)].unsqueeze(0) for i in devices]
            output = nn.parallel.parallel_apply(net, input)
            label = [labels[i][batch_size * round: batch_size*(round + 1)] for i in devices]
            mask = [masks[i][batch_size * round: batch_size*(round + 1)] for i in devices]
            label_ = []
            output_ = []
            for i in devices:
                l = label[i].clone()
                o = output[i].clone()
                num = 0
                for idx, m in enumerate(mask[i]):
                    if m > 0:
                        l[num] = label[i][idx]
                        o[num] = output[i][idx]
                        num += 1
                label_.append(l[:num])
                output_.append(o[:num])
                total += num
            predict = [torch.max(output_[i], -1)[1] for i in devices]
            correct += sum([(predict[i] == label_[i]).sum().item() for i in devices])
            loss = nn.parallel.parallel_apply(criterion, list(zip(output_, label_)))
            for idx in range(len(loss)):
                loss[idx] = loss[idx].unsqueeze(0)
            loss = torch.sum(nn.parallel.gather(loss, target_device=devices[-1]))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += int(loss)
        print ("epoch [%d] round [%d] accuracy: %.3f" %(epoch, k, correct*1.0/total))
    print('[%d] loss: %.3f accuracy: %.3f' %(epoch + 1, running_loss, correct*1.0/total))
print('Finished Training')
