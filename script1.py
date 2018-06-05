import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random



############################# PREDEFINED #########################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TrainSetDir = "./data/s_given_t_train.txt"
TestSetDir = "./data/s_given_t_test.txt"
DicDir = "./data/movie_25000"

MaxLength = 15
MinLength = 5
SOS_Token = 0
EOS_Token = 25001
VocabSize = 25002
EmbeddingSize = 1024

############################ Data Preparation #############################

def index2sentence(l, dic):
    s = ''
    for index in l:
        s += dic[index] + ' '
    return s

def filterPair(p):
    return len(p[0])<MaxLength and len(p[1])<MaxLength and len(p[1])>MinLength

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def PrepareData(train_set_dir=TrainSetDir, test_set_dir=TestSetDir,
                dic_dir=DicDir):
    train_set = []
    test_set = []
    dic = {0:'<SOS>', 25001:'<EOS>'}

    print("Loading data...")
    with open(train_set_dir, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            train_set.append(line.strip('\n').split("|"))

    with open(test_set_dir, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            test_set.append(line.strip('\n').split("|"))

    with open(dic_dir, 'r', encoding='utf-8') as f:
        for i, word in enumerate(f.readlines()):
            dic[i + 1] = word.strip('\n')  # 注意：字典里的索引应该从1而不是0开始！

    for i in range(len(train_set)):
        train_set[i] = [[int(index) for index in train_set[i][0].split()],
                        [int(index) for index in train_set[i][1].split()]]

    for i in range(len(test_set)):
        test_set[i] = [[int(index) for index in test_set[i][0].split()],
                        [int(index) for index in test_set[i][1].split()]]

    print("{} training pairs loaded, {} testing pairs loaded. ".format(len(train_set), len(test_set)))

    train_set = filterPairs(train_set)
    test_set = filterPairs(test_set)

    print("Trimmed to {} training pairs and {} testing pairs. ".format(len(train_set), len(test_set)))

    return train_set, test_set, dic


TrainSet, TestSet, index2word = PrepareData()
random_pair = random.choice(TrainSet)
print("Showing a random pair of train_set: \n>> {}\n>> {}"
      .format(index2sentence(random_pair[0], index2word), index2sentence(random_pair[1], index2word)))


def tensorFromPair(pair):
    input_tensor = pair[0]
    input_tensor = torch.tensor(input_tensor, dtype=torch.long).view(-1,1)
    target_tensor = pair[1]
    target_tensor.append(EOS_Token)
    target_tensor = torch.tensor(target_tensor, dtype=torch.long).view(-1,1)

    return input_tensor, target_tensor

############################ Generator ###################################
class EncoderG(nn.Module):
    def __init__(self, vocab_size=VocabSize, embedding_size=EmbeddingSize, drop_out = 0.):
        super(EncoderG, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.hidden_size = self.embedding_size   # the hidden size of GRU equals embedding size by default
        self.drop_out = drop_out

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, dropout = self.drop_out)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.embedding_size).to(device)

class DecoderG(nn.Module):
    def __init__(self, vocab_size=VocabSize, embedding_size=EmbeddingSize, drop_out=0., max_length=MaxLength):
        super(DecoderG, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = self.embedding_size  # the hidden size of GRU equals embedding size by default
        self.drop_out = drop_out
        self.max_length = max_length

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.attn = nn.Linear(embedding_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.embedding_size*2, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, dropout=self.drop_out)
        self.out = nn.Linear(self.embedding_size, self.vocab_size)

    def forward(self, word_in, hidden,encoder_outputs):
        embedded = self.embedding(word_in).view(1, 1, -1)
        attn_weights = F.softmax(self.attn(torch.cat((hidden[0], embedded[0]), dim=1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = self.attn_combine(torch.cat((embedded[0], attn_applied[0]), dim=1)).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zero(1, 1, self.embedding_size).to(device)

def GenForward(encoder_G, decoder_G, input_tensor, max_length=MaxLength):
    '''Using Generator to generate answer given an input_tensor'''

    input_length = input_tensor.size(0)
    encoder_hidden = encoder_G.initHidden()
    encoder_outputs = torch.zeros(max_length, encoder_G.hidden_size)
    decoder_outputs = []

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder_G(input_tensor[i], encoder_hidden)
        encoder_outputs[i] = encoder_output[0][0]

    decoder_input = torch.tensor([SOS_Token])
    decoder_hidden = encoder_hidden

    for i in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder_G(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()
        decoder_outputs.append(decoder_input.item())
        if decoder_input.item() == EOS_Token:
            break

    return decoder_outputs

############################# Discriminator ##############################


############################# Pretraining ################################

###################### A demo of using Generator ########################
# Gen_encoder = EncoderG()
# Gen_decoder = DecoderG()
# input_tensor, target_tensor = tensorFromPair(random_pair)
# Gen_output = GenForward(Gen_encoder,Gen_decoder,input_tensor)
#
# print("A test of Generator: \nsource: {}\ntarget: {}\ngenerated: {}"
#       .format(index2sentence(random_pair[0], index2word),
#               index2sentence(random_pair[1], index2word),
#               index2sentence(Gen_output, index2word)))
##########################################################################

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def pretrainG(encoder, decoder, num_iter=10000, print_loss_every=1000, max_length=MaxLength, learning_rate=0.01,
              teacher_forcing_ratio = 0.5):
    start_time = time.time()
    total_loss = 0
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    ## training data loading ##
    training_pairs = [tensorFromPair(random.choice(TrainSet)) for i in range(num_iter)]

    criterion = nn.NLLLoss()

    ## training for a single pair ##
    for iter in range(1,num_iter+1):
        ## data loading ##
        training_pair = training_pairs[iter-1]
        input_tensor = training_pair[0].to(device)
        target_tensor = training_pair[1].to(device)

        encoder_hidden = encoder.initHidden().to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size).to(device)

        loss = 0

        ## encoding ##
        for i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
            encoder_outputs[i] = encoder_output[0][0]

        decoder_input = torch.tensor([SOS_Token]).to(device)
        decoder_hidden = encoder_hidden.to(device)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        ## decoding ##
        if use_teacher_forcing:
           # teacher forcing
           for di in range(target_length):
               decoder_output, decoder_hidden, decoder_attention = decoder(
                   decoder_input, decoder_hidden, encoder_outputs)
               decoder_input = target_tensor[di]

               loss += criterion(decoder_output, target_tensor[di])
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])

                if decoder_input.item() == EOS_Token:
                    break

        ## BPTT & Parameters updating ##
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

        if iter % print_loss_every == 0:
            average_loss = total_loss/print_loss_every
            total_loss = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start_time, iter / num_iter),
                                         iter, iter / num_iter * 100, average_loss))

    return 0

## pretrain Generator ##
Gen_encoder = EncoderG().to(device)
Gen_decoder = DecoderG().to(device)

try:
    Gen_encoder.load_state_dict(torch.load("./Param/Gen_encoder_params.pkl"))
    Gen_decoder.load_state_dict(torch.load("./Param/Gen_decoder_params.pkl"))
    print("Model parameters loaded.")
except FileNotFoundError:
    print("Model parameters loading failed.")

pretrainG(Gen_encoder, Gen_decoder, num_iter=200, print_loss_every=50)

torch.save(Gen_encoder.state_dict(), "./Param/Gen_encoder_params.pkl")
torch.save(Gen_decoder.state_dict(), "./Param/Gen_decoder_params.pkl")

# # 保存和加载整个模型
# torch.save(model_object, 'model.pkl')
# model = torch.load('model.pkl')
#
# # 仅保存和加载模型参数(推荐使用)
# torch.save(model_object.state_dict(), 'params.pkl')
# model_object.load_state_dict(torch.load('params.pkl'))















































