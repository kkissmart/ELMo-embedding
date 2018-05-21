# EMLo
By running training.py, it loads pretrained BiLM weights and applies EMLo embedding on top of it. 
the pretrained weights are from https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
the json conf file is in this repo.
the pretrained bilm conf needs to be consistent with weights.
This repo can be used to test on several langunage problems by change sentences variable in training.py. Training.py is able to read txt files, parse it into the right tensor dim and use pretrained bilm weights to convert raw string into EMLo embedding, and finnally this embeddings will be used as inputs to different tasks. 
