from data.data import Vocabulary, UnicodeCharsVocabulary, Batcher
from modules.model import Elmo

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
torch.set_default_tensor_type('torch.cuda.FloatTensor')


options_file = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
train_file = "sentences.small.train"
validation_file = "sentences.small.dev"


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

    return Variable(torch.cuda.FloatTensor(data))

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

global tgs_dict
tgs_dict = {}
for idx, tag in enumerate(total_tags):

    tgs_dict[tag] = idx


def get_targets_with_max_length(targets, max_sentence_length):
    tgs_ = []
    for tg in targets:

        if len(tg) <= max_sentence_length:
            tgs_.append([tgs_dict[tok] for tok in tg] + [len(tgs_dict)+1] * (max_sentence_length -len(tg)))
        else:
            tgs_.append([tgs_dict[tok] for tok in tg[:50]])
    return Variable(torch.cuda.LongTensor(tgs_))

epoch_number = 1

num_Elmo_layers = 2
max_sentence_length = 50
batch_size = 64
num_gpu = 4
max_load = 200




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



def train(epoch_number,
          num_Elmo_layers,
          max_sentence_length,
          batch_size,
          num_gpu,
          max_load,
          options_file,
          weight_file,
          sentences,
          targets):
    devices = range(num_gpu)
    #num_gpu = len(devices)
    elmo = Elmo(options_file, weight_file, num_Elmo_layers, dropout=0)
    print(len(list(elmo.parameters())))
    elmo_parameters = [p for p in elmo.parameters() if p.requires_grad]
    net = Net(max_sentence_length, num_Elmo_layers, len(tgs_dict)).cuda()
    net_parameters = net.parameters()
    criterion = torch.nn.CrossEntropyLoss()
    parameters = list(net_parameters)+elmo_parameters
    optimizer = optim.SGD(parameters, lr=0.005, momentum=0.9)
    num_rounds = int(len(sentences)) // max_load
    for epoch in range(epoch_number):
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        for k in range(num_rounds):
            character_ids = batch_to_ids("vocab.txt",
                                         sentences[max_load*k:max_load*(k+1)],
                                         max_sentence_length,
                                         50,
                                         with_bos_eos=False)
            labels = get_targets_with_max_length(targets[max_load*k:max_load*(k+1)],
                                                 max_sentence_length)
            character_ids = nn.parallel.scatter(tuple([character_ids,]), devices)
            labels = nn.parallel.scatter(labels, devices)
            elmo_replicas = nn.parallel.replicate(elmo, devices)
            net_replicas = nn.parallel.replicate(net, devices)
            criterion_replicas = nn.parallel.replicate(criterion, devices)
            embeddings = nn.parallel.parallel_apply(elmo_replicas, character_ids)
            tokens = [torch.cat(embeddings[i]["elmo_representations"], -1).view(-1, 1024*num_Elmo_layers) for i in devices]
            masks = [embeddings[i]["mask"].view(-1) for i in devices]
            labels = [labels[i].view(-1) for i in devices]
            Rounds = tokens[0].shape[0] // batch_size
            for round in range(Rounds):
                input = [tokens[i][batch_size * round: batch_size*(round + 1)] for i in devices]
                label = [labels[i][batch_size * round: batch_size*(round + 1)] for i in devices]
                mask = [masks[i][batch_size * round: batch_size*(round + 1)] for i in devices]
                label_ = [0] * num_gpu
                input_ = [0] * num_gpu
                for i in devices:
                    lab = label[i].clone()
                    inp = input[i].clone()
                    num = 0
                    for idx, m in enumerate(mask[i]):
                        if m > 0:
                            lab[num] = label[i][idx]
                            inp[num] = input[i][idx]
                            num += 1
                    label_[i] = Variable(lab[:num])
                    input_[i] = Variable(inp[:num].unsqueeze(0))
                    total += num
                output = nn.parallel.parallel_apply(net_replicas, input_)
                predict = [torch.max(output[i], -1)[1] for i in devices]
                correct += sum([(predict[i] == label_[i]).sum().item() for i in devices])
                loss = nn.parallel.parallel_apply(criterion_replicas, list(zip(output, label_)))
                #print(loss)
                for idx in range(len(loss)):
                    loss[idx] = loss[idx].unsqueeze(0)
                loss_ = nn.parallel.gather(loss, target_device=devices[0]).mean()
                optimizer.zero_grad()
                loss_.backward()
                #print(net.fc.weight.grad.data)
                optimizer.step()
                running_loss += float(loss_)
                #print ("epoch [%d] round_within_load [%d] accuracy: %.3f total loss: %.3f" %(epoch, round, correct*1.0/total, running_loss) )
            #print ("epoch [%d] #_load [%d] accuracy: %.3f total loss: %.3f" %(epoch, k, correct*1.0/total, running_loss) )
        print('[%d] loss: %.3f accuracy: %.3f' %(epoch + 1, running_loss, correct*1.0/total))
    print('Finished Training')

if __name__ == "__main__":
    train(epoch_number,
          num_Elmo_layers,
          max_sentence_length,
          batch_size,
          num_gpu,
          max_load,
          options_file,
          weight_file,
          sentences,
          targets)
