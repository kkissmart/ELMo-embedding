import numpy as np
import torch


from data.data import Vocabulary, UnicodeCharsVocabulary, Batcher
from modules.model import Elmo

DTYPE = 'float32'
DTYPE_INT = 'int64'

options_file = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

def load_vocab(vocab_file, max_word_length=50):
    if max_word_length:
        return UnicodeCharsVocabulary(vocab_file, max_word_length,
                                      validate_file=True)
    else:
        return Vocabulary(vocab_file, validate_file=True)


def batch_to_ids(lm_vocab_file, batch, max_sentence_length, max_token_length):
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
    data = Batcher(lm_vocab_file, max_token_length, max_sentence_length).batch_sentences(batch)
    return torch.FloatTensor(data)

sentences = [['First', 'sentence', '.'], ['Another', '.']]
vocal = "vocab-2016-09-10.txt"
character_ids = batch_to_ids("vocab-2016-09-10.txt", sentences, 20, 50)
elmo = Elmo(options_file, weight_file, 2, dropout=0)
embeddings = elmo(character_ids)
