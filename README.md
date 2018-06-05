# ELMo
By running training.py, 
1) load pretrained BiLM weights 
2) apply ELMo embedding on top of the BiLM weights with EMLo Parameters.
3) read training data from sentences.small.train
4) pass the training data into X and label (POS labeling)
5) map the X into EMLo embeddings with EMLo parameters
6) concat ELMo embeddings
7) add one projection fc layer on EMLo embedding
8) use cross entropy as loss function

___________________________________________________________________________
the pretrained weights are saved in
elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
the json conf file is in this repo 
elmo_2*4096_512_2048cnn_2*highway_options.json
*the pretrained bilm conf needs to be consistent with weights*
biLM architecture here is 
1) char embedding 262 unique chars into 16-dim vector
2) apply CNN with 2048 filters (with different resolutions for each group of filters)
3) apply 2 highway bypass
4) apply projection layers
5) apply 2 layers of LSTM on top of the projection layer output 
    LSTM has memory dim of 4096 and hidden layer dim of 512
6) cat forward and backward feed into 512*2 dim. 

__________________________________________________________________________
ELMo layers conf
1) input num_tensors = # of biLM layers + 1 
    * 1 = context independent input, the one out of cnn+highway+projection, before LSTM
2) input tensor = [context independent input] + [first LSTM output] + [second LSTM output]
3) mixture = gamma * sum(weights * tensor)
    * weights are scalar with softmax. 
    * ELMo parameters:  (gamma, weights) * # of ELMo layers

__________________________________________________________________________
This repo can be used to test on several langunage problems by change sentences variable in training.py (here POS is used for test purpose). 
Training.py is able to read large txt files, parse it into the right tensor dim with proper batch size and use pretrained bilm weights to convert raw string into ELMo embedding, and finnally this embeddings will be used as inputs to different tasks. 
