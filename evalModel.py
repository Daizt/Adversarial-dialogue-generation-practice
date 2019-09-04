
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
from Models import EncoderRNN, Attn, LuongAttnDecoderRNN, GreedySearchDecoder
from trainModel import readData
from dataProcess import normalizeString, unicodeToAscii


# Predefined
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
corpus_name = "cornell movie-dialogs corpus"
PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 15

def main():
    ####################
    # Read word-index dict
    ####################
    cwd = os.getcwd()
    save_dir = os.path.join(cwd, "data")
    path2 = os.path.join(save_dir, "index2word.txt")

    _, num_words, word2index, index2word = readData(path2=path2)
    ####################
    # Configure models 
    ####################
    model_name = 'cb_model'
    attn_model = 'dot'
    #attn_model = 'general'
    #attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # Set checkpoint to load from; set to None if starting from scratch
    # loadFilename = None
    checkpoint_iter = 18000
    loadFilename = os.path.join(save_dir, model_name, corpus_name,
                               '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                               '{}_checkpoint.tar'.format(checkpoint_iter))


    # Load model if a loadFilename is provided
    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built successfully !')

	# Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    # Begin chatting (uncomment and run the following line to begin)
    evaluateInput(encoder, decoder, searcher, word2index, index2word)

def evaluate(encoder, decoder, searcher, sentence, word2index, index2word, max_length=MAX_LENGTH, device=device):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [word2index[word] for word in sentence.split(' ')] + [EOS_token]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes_batch)])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).view(-1, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length, device)
    # indexes -> words
    decoded_words = [index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, word2index, index2word):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, input_sentence, word2index, index2word)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

if __name__ == '__main__':
    main()