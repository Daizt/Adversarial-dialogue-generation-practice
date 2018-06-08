import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random



############################# PREDEFINED ################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TrainSetDir = "./data/s_given_t_dialogue_length2_6.txt"
TestSetDir = "./data/s_given_t_test.txt"
DicDir = "./data/movie_25000"
GenPairsDir = "./data/generated_dialogue.txt"

MaxLength = 20
MinLength = 5
SOS_Token = 0
EOS_Token = 25001
VocabSize = 25002
EmbeddingSize = 512

############################ Data Preparation #############################

def index2sentence(l, dic):
    s = ''
    for index in l:
        s += dic[index] + ' '
    return s

def filterPair(p):
    return (len(p[0])<MaxLength and len(p[1])<MaxLength and len(p[1])>MinLength and
            1 not in p[0] and 1 not in p[1])

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def PrepareData(set1_dir=None, set2_dir=None,
                dic_dir=None):
    set1 = []
    set2 = []
    dic = {0:'<SOS>', 25001:'<EOS>'}

    print("Loading data...")
    if set1_dir != None:
        with open(set1_dir, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                set1.append(line.strip('\n').split("|"))
        for i in range(len(set1)):
            set1[i] = [[int(index) for index in set1[i][0].split()],
                            [int(index) for index in set1[i][1].split()]]

    if set2_dir != None:
        with open(set2_dir, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                set2.append(line.strip('\n').split("|"))
        for i in range(len(set2)):
            set2[i] = [[int(index) for index in set2[i][0].split()],
                           [int(index) for index in set2[i][1].split()]]

    if dic_dir !=None:
        with open(dic_dir, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f.readlines()):
                dic[i + 1] = word.strip('\n')  # 注意：字典里的索引应该从1而不是0开始！

    print("{} training pairs loaded, {} testing pairs loaded. ".format(len(set1), len(set2)))

    set1 = filterPairs(set1)
    set2 = filterPairs(set2)

    print("Trimmed to {} training pairs and {} testing pairs. ".format(len(set1), len(set2)))

    return set1, set2, dic


TrainSet, TestSet, index2word = PrepareData(set1_dir=TrainSetDir,
                                            set2_dir=TestSetDir,
                                            dic_dir=DicDir)

# random_pair = random.choice(TrainSet)
# print("Showing a random pair of train_set: \n>> {}\n>> {}"
#       .format(index2sentence(random_pair[0], index2word), index2sentence(random_pair[1], index2word)))

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
        return torch.zeros(1, 1, self.embedding_size)#.to(device)

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
        return torch.zero(1, 1, self.embedding_size)#.to(device)

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
class hierEncoder(nn.Module):
    def __init__(self, vocab_size=VocabSize, embedding_size=EmbeddingSize):
        super(hierEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = self.embedding_size  # the hidden size of GRU equals embedding size by default

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.gru1 = nn.GRU(self.embedding_size, self.embedding_size)
        self.gru2 = nn.GRU(self.embedding_size, self.embedding_size)
        self.linear1 = nn.Linear(self.embedding_size, 128)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, pair, tag):
        # pair为对话 {x, y} ；tag为标签，其中 0 表示人生成的对话， 1 表示generator生成的对话。二者类型均为torch.tensor()
        # example: pair = torch.tensor([[3,2,11,12],[1,3,24,123,5]]), tag = torch.tensor([0])
        x_length = pair[0].size(0)
        y_length = pair[1].size(0)

        hidden = self.initHidden()
        for i in range(x_length):
            embedded_x = self.embedding(pair[0][i]).view(1, 1, -1)
            _, hidden = self.gru1(embedded_x, hidden)
        hidden_x = hidden     # x句的编码结果

        hidden = self.initHidden()
        for i in range(y_length):
            embedded_y = self.embedding(pair[1][i]).view(1, 1, -1)
            _, hidden = self.gru1(embedded_y, hidden)
        hidden_y = hidden     # y句的编码结果

        hidden = self.initHidden()
        _, hidden = self.gru2(hidden_x, hidden)
        _, hidden = self.gru2(hidden_y, hidden)
        hidden_xy = hidden    # 得到{x，y}编码结果

        output = F.relu(self.linear1(hidden_xy.squeeze()))
        output = F.relu(self.linear2(output)).view(1, -1)
        output = F.log_softmax(output, dim=1)        ## 注意此处的输出为 log_softmax

        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.embedding_size)

############################# Pretraining ################################

###################### A demo of using Generator ########################
# Gen_encoder = EncoderG()
# Gen_decoder = DecoderG()
# input_tensor, target_tensor = tensorFromPair(random_pair)
# Gen_output = GenForward(Gen_encoder,Gen_decoder,input_tensor)
#
# print("A test of Generator: \n<source>: {}\n<target>: {}\n<generated>: {}"
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

def pretrainG(encoder, decoder, batch_size=128, max_length=MaxLength, learning_rate=0.01,
              teacher_forcing_ratio = 0.5):
    start_time = time.time()
    total_loss = 0
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    ## training data loading ##
    training_pairs = [tensorFromPair(random.choice(TrainSet)) for i in range(batch_size)]

    criterion = nn.NLLLoss()

    for iter in range(1, batch_size + 1):
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

        # if iter % print_loss_every == 0:
        #     average_loss = total_loss/print_loss_every
        #     total_loss = 0
        #     print('%s progress: (%d %d%%) average loss: %.4f' % (timeSince(start_time, iter / batch_size),
        #                                                          iter, iter / batch_size * 100, average_loss))

    print("Time consumed: {} Average loss: {} ".format(asMinutes(time.time()-start_time), total_loss/batch_size))

    return 0

########################### pretrain Generator #########################################
Gen_encoder = EncoderG().to(device)
Gen_decoder = DecoderG().to(device)

try:
    Gen_encoder.load_state_dict(torch.load("./ModelParams/Gen_encoder_params.pkl"))
    Gen_decoder.load_state_dict(torch.load("./ModelParams/Gen_decoder_params.pkl"))
    print("Model parameters loaded.")
except FileNotFoundError:
    print("Model parameters loading failed.")

for epoch in range(5):
    pretrainG(Gen_encoder, Gen_decoder)
    ans = input("Do you want to stop training and save the model? [y/n]")
    if ans == 'y':
        break
    else:
        print("Proceed with training...")

try:
    torch.save(Gen_encoder.state_dict(), "./ModelParams/Gen_encoder_params.pkl")
    torch.save(Gen_decoder.state_dict(), "./ModelParams/Gen_decoder_params.pkl")
    print("Model parameters saved successfully.")
except:
    print("Failed to save model parameters.")

#########################################################################################

############# Test Generator & Provide negative data for training Discriminator ##################
# Gen_encoder = EncoderG()
# Gen_decoder = DecoderG()
#
# try:
#     Gen_encoder.load_state_dict(torch.load("./ModelParams/Gen_encoder_params.pkl"))
#     Gen_decoder.load_state_dict(torch.load("./ModelParams/Gen_decoder_params.pkl"))
#     print("Model parameters loaded.")
# except FileNotFoundError:
#     print("Model parameters loading failed.")
#
# train_pairs = [tensorFromPair(random.choice(TrainSet)) for i in range(5)]
# print("----------------Evaluation on training set: --------------------- ")
# for i in range(5):
#     Gen_output = GenForward(Gen_encoder,Gen_decoder,train_pairs[i][0])
#     print("--------------------------------------------------------")
#     print("<source>: {}\n<target>: {}\n<generated>: {}"
#       .format(index2sentence(train_pairs[i][0].squeeze().numpy(), index2word),
#               index2sentence(train_pairs[i][1].squeeze().numpy(), index2word),
#               index2sentence(Gen_output, index2word)))
#
# test_pairs = [tensorFromPair(random.choice(TestSet)) for i in range(5)]
# print("----------------Evaluation on testing set: -----------------------")
# for i in range(5):
#     Gen_output = GenForward(Gen_encoder,Gen_decoder,test_pairs[i][0])
#     print("--------------------------------------------------------")
#     print("<source>: {}\n<target>: {}\n<generated>: {}"
#       .format(index2sentence(test_pairs[i][0].squeeze().numpy(), index2word),
#               index2sentence(test_pairs[i][1].squeeze().numpy(), index2word),
#               index2sentence(Gen_output, index2word)))
#
# # 生成数据
# with open("./data/generated_dialogue.txt", 'a', encoding='utf-8') as f:
#     for i in range(1000):
#         s = ''
#         x, _ = tensorFromPair(random.choice(TrainSet))
#         generated = GenForward(Gen_encoder, Gen_decoder, x)
#         x = x.squeeze().numpy()
#         # print("input: ",x)
#         # print("generate: ",generated)
#         for num in x:
#             s += str(num) + ' '
#         s += "|"
#         for num in generated:
#             s += str(num) + ' '
#         s += '\n'
#         f.write(s)

######################################################################################################

###################### 模型保存笔记 ####################################
# # 保存和加载整个模型
# torch.save(model_object, 'model.pkl')
# model = torch.load('model.pkl')
#
# # 仅保存和加载模型参数(推荐使用)
# torch.save(model_object.state_dict(), 'params.pkl')
# model_object.load_state_dict(torch.load('params.pkl'))
#######################################################################

########################### Pretrain Discriminator ############################################






###############################################################################################












































