# EMLo
By running training.py, 
1) load pretrained BiLM weights 
2) apply EMLo embedding on top of the BiLM weights with EMLo Parameters.
3) read training data from sentences.small.train
4) pass the training data into X and label (NER labeling)
5) map the X into EMLo embeddings with EMLo parameters
6) use fc layer to cat EMLo embeddings
7) add one projection layer on EMLo embedding
8) use cross entropy as loss function
-the pretrained weights are from https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
-the json conf file is in this repo (elm_2*4096_512_2048cnn_2*highway_options.json
the pretrained bilm conf needs to be consistent with weights.
biLM architecture here is 
1) char embedding 262 unique chars into 16-dim vector
2) apply CNN with 2048 filters (with different resolutions)
3) apply two highway by-pass layers
4) project the output into 512 dim
5) cat forward and backward feed into 512*2 dim. 
This repo can be used to test on several langunage problems by change sentences variable in training.py (here NER is used for test purpose). 
Training.py is able to read large txt files, parse it into the right tensor dim with proper batch size and use pretrained bilm weights to convert raw string into EMLo embedding, and finnally this embeddings will be used as inputs to different tasks. 
