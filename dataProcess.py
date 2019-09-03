'''
This script creates a dictionary according to conversations, then trims infrequently seen words as well
as sentences that are too long or too short, finally save the trimmed conversation pairs and word-index 
dict as new files for further use.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import os
import unicodedata
from io import open

# Default word tokens
PAD_token = 0
SOS_token = 1
EOS_token = 2

# Define sentence length & word count threshold
MAX_LENGTH = 15
MIN_LENGTH = 2
MIN_COUNT = 3    

def main():
    # Load/Assemble voc and pairs
    corpus_name = "cornell movie-dialogs corpus"
    corpus = os.path.join("data", corpus_name)
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    # Load data & trim sentence length
    vocab, pairs = loadPrepareData(corpus_name, datafile)

    # Trim voc and pairs(Notice that the vocab is updated implicitly)
    pairs = trimRareWords(vocab, pairs, MIN_COUNT)

    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)

    # Save as new files
    cwd = os.getcwd()
    save_dir = os.path.join(cwd, "data")
    path1 = os.path.join(save_dir, "trimmed_pairs.txt")
    path2 = os.path.join(save_dir, "index2word.txt")

    with open(path1, 'w') as f:
        for pair in pairs:
            temp = [vocab.word2index[w] for w in pair[0].strip().split()]
            line = ''
            for idx in temp:
                line += str(idx) + ' '
            line = line.strip() + '||'
            temp = [vocab.word2index[w] for w in pair[1].strip().split()]
            for idx in temp:
                line += str(idx) + ' '
            line = line.strip() + '\n'
            f.write(line)

    with open(path2, 'w') as f:
        for k, v in vocab.index2word.items():
            line = str(k) + '||' + str(v) + '\n'
            f.write(line)

class Vocab(object):
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.index2word = {PAD_token:"PAD", SOS_token:"SOS", EOS_token:"EOS"}
        self.word2count = {}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(len(keep_words), 
                                                   len(self.word2index), 
                                                   len(keep_words) / len(self.word2index)))
        # Reinitialization
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Read query/response pairs and return a voc object
def readVocs(datafile):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('$')] for l in lines]
    return pairs

# Returns True if both sentences in a pair 'p' have suitable length
def filterPairs(pairs):
    # Input sequences need to preserve the last word for EOS token
    length_filter = lambda p: len(p[0].split(' ')) < MAX_LENGTH and \
                              len(p[1].split(' ')) < MAX_LENGTH and \
                              len(p[0].split(' ')) > MIN_LENGTH and \
                              len(p[1].split(' ')) > MIN_LENGTH
    return [p for p in pairs if length_filter(p)]

# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus_name, datafile):
    print("Start preparing training data ...")
    pairs = readVocs(datafile)
    vocab = Vocab(corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        vocab.addSentence(pair[0])
        vocab.addSentence(pair[1])
    print("Counted words:", vocab.num_words)
    return vocab, pairs

def trimRareWords(vocab, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    vocab.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in vocab.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in vocab.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

if __name__ == "__main__":
    main()