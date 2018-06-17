import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random
import numpy as np
import math


"""原数据提供的词典当中有大量无效索引，其对应的索引值为数字，将数据经
过过滤后得
############到的对话存入trimmed_dialogue.txt文件当中作为训练数据。"""
################# PREDEFINED ################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TrainSetDir = "./data/s_given_t_train.txt"
TestSetDir = "./data/s_given_t_test.txt"
TrimmedSetDir = "./data/trimmed_dialogue.txt"
InvalidIndexDir = "./data/invalid_index.txt"
DicDir = "./data/movie_25000"
GenSetDir = "./data/generated_dialogue.txt"

MaxLength = 20
MinLength = 5
SOS_Token = 0
EOS_Token = 25001
VocabSize = 25002
EmbeddingSize = 512


############################ Data Preparation #############################

def index2sentence(l, dic):
    # example l = [1, 2, 3]
    s = ''
    for index in l:
        s += dic[index] + ' '
    return s

invalid_index = []
# 读取无效索引值
with open(InvalidIndexDir, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        invalid_index.append(int(line.strip('\n')))

def filterPair(p, invalid_index=invalid_index):
    # example: p = [[1,2,3],[4,5,6]]

    return (len(p[0])<MaxLength and
            len(p[1])<MaxLength and
            len(p[1])>MinLength and
            1 not in p[0] and
            1 not in p[1] and
            len(set(invalid_index) & set(p[0])) == 0 and
            len(set(invalid_index) & set(p[1])) == 0)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def PrepareData(set1_dir=None, set2_dir=None,
                dic_dir=None,if_filter=True):
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

    if dic_dir != None:
        with open(dic_dir, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f.readlines()):
                dic[i + 1] = word.strip('\n')  # 注意：字典里的索引应该从1而不是0开始！

    print("{} training pairs loaded, {} testing pairs loaded. ".format(len(set1), len(set2)))

    if if_filter:
        set1 = filterPairs(set1)
        set2 = filterPairs(set2)
        print("Trimmed to {} training pairs and {} testing pairs. ".format(len(set1), len(set2)))

    return set1, set2, dic


############################## 用作数据预处理 ###############################
# Loading positive data and negative data
# TrainSet, TestSet, index2word = PrepareData(set1_dir=TrainSetDir,
#                                             set2_dir=TestSetDir,
#                                             dic_dir=DicDir)
# GenSet, _, _ = PrepareData(set1_dir=GenSetDir)
#
# # 保存过滤后的数据
# with open("./data/trimmed_dialogue.txt", 'w', encoding='utf-8') as f:
#     for pair in TrainSet:
#         s = ''
#         for num in pair[0]:
#             s+=str(num)+' '
#         s += '|'
#         for num in pair[1]:
#             s += str(num) +' '
#         s += '\n'
#         f.write(s)

# random_pair = random.choice(TrainSet)
# print("Showing a random pair of train_set: \n>> {}\n>> {}"
#       .format(index2sentence(random_pair[0], index2word),
#       index2sentence(random_pair[1], index2word)))
##############################################################################


########################### 数据加载 #####################################

TrainSet, TestSet, index2word = PrepareData(set1_dir=TrainSetDir,
                                            set2_dir=TestSetDir,
                                            dic_dir=DicDir,
                                            if_filter=True)
GenSet, _, _ = PrepareData(set1_dir=GenSetDir, if_filter=False)

#########################################################################

def tensorFromPair(pair, to_device=True):
    input_tensor = pair[0]
    target_tensor = pair[1]

    if to_device:
        input_tensor = torch.tensor(input_tensor, dtype=torch.long).view(-1,1).to(device)

        if target_tensor[-1] != EOS_Token:
            target_tensor.append(EOS_Token)
        target_tensor = torch.tensor(target_tensor, dtype=torch.long).view(-1,1).to(device)

    else:
        input_tensor = torch.tensor(input_tensor, dtype=torch.long).view(-1, 1)

        if target_tensor[-1] != EOS_Token:
            target_tensor.append(EOS_Token)
        target_tensor = torch.tensor(target_tensor, dtype=torch.long).view(-1, 1)

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
        '''example: input = torch.tensor([12])'''
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
        return torch.zeros(1, 1, self.embedding_size)#.to(device)


def GenForward(encoder_G,
               decoder_G,
               input_tensor,
               max_length=MaxLength,
               to_device=False,
               if_beam_search=False,
               beam_search_k=2):
    '''Using Generator to generate answer given an input_tensor'''
    # input_tensor example: torch.tensor([[21],[11],[1]], dtype=torch.long)
    # output type: list(). example: decoder_outputs = [1,2,3]

    input_length = input_tensor.size(0)

    if to_device:
        encoder_hidden = encoder_G.initHidden().to(device)
        encoder_outputs = torch.zeros(max_length, encoder_G.hidden_size).to(device)
    else:
        encoder_hidden = encoder_G.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder_G.hidden_size)

    # Encoding
    for k in range(input_length):
        encoder_output, encoder_hidden = encoder_G(input_tensor[k], encoder_hidden)
        encoder_outputs[k] = encoder_output[0][0]

    if to_device:
        decoder_input = torch.tensor([SOS_Token]).to(device)
    else:
        decoder_input = torch.tensor([SOS_Token])

    decoder_hidden = encoder_hidden

    # Decoding

    # 采用beam search
    if if_beam_search:
        # 用来记录概率最大的k个选择的隐藏层, type: [torch.tensor()]
        hidden_log = [decoder_hidden for _ in range(beam_search_k)]

        # 用来记录最大的k个概率, type: [float]
        prob_log = [0 for _ in range(beam_search_k)]

        # 用来记录概率最大的k个选择的解码输出, type: [[int]]
        decoder_outputs = np.empty([beam_search_k,1]).tolist()

        # 先进行第一步解码
        decoder_output, decoder_hidden, decoder_attention = decoder_G(decoder_input, decoder_hidden,
                                                                        encoder_outputs)
        # 选择概率最大的k个选项
        topv, topi = decoder_output.topk(beam_search_k)

        for k in range(beam_search_k):
            # 记录隐藏层, type: [torch.tensor()]
            hidden_log[k] = decoder_hidden

            # 记录概率（默认降序排列）, type: [float]
            prob_log[k] += topv.squeeze()[k].item()

            # 记录输出（与prob_log的概率对应）, type: [int]
            decoder_outputs[k].append(topi.squeeze()[k].item())
            decoder_outputs[k].pop(0)   # 删除初始化时存入的元素

        # beam search
        for ei in range(max_length-1):
            # 用以暂时存储概率在后续进行比较
            if to_device:
                temp_prob_log = torch.tensor([]).to(device)
                temp_output_log = torch.tensor([], dtype=torch.long).to(device)
                temp_hidden_log = []
            else:
                temp_prob_log = torch.tensor([])
                temp_hidden_log = []
                temp_output_log = torch.tensor([],dtype=torch.long)

            for k in range(beam_search_k):
                if to_device:
                    decoder_input = torch.tensor([decoder_outputs[k][-1]], dtype=torch.long).to(device)

                else:
                    decoder_input = torch.tensor([decoder_outputs[k][-1]], dtype=torch.long)

                decoder_hidden = hidden_log[k]
                decoder_output, decoder_hidden, _ = decoder_G(decoder_input, decoder_hidden,
                                                                            encoder_outputs)
                # 初步比较
                topv, topi = decoder_output.topk(beam_search_k)

                temp_prob_log = torch.cat([temp_prob_log, topv], dim=1)
                temp_hidden_log.append(decoder_hidden)
                temp_output_log = torch.cat([temp_output_log, topi], dim=1)

            # 最终比较（在 k*K 个候选项中选出概率最大的 k 个选项）
            temp_topv, temp_topi = temp_prob_log.topk(beam_search_k)

            # 记录结果(保持概率降序排列)
            for k in range(beam_search_k):
                ith = int(temp_topi.squeeze()[k].item()/beam_search_k)
                hidden_log[k] = temp_hidden_log[ith]

                prob_log[k] += temp_topv.squeeze()[k].detach().item()

                decoder_outputs[k].append(temp_output_log.squeeze()[temp_topi.squeeze()[k].item()].detach().item())

            # <EOS>
            pass

        return decoder_outputs, prob_log

    # 采用贪心选择
    else:
        decoder_outputs = []
        for k in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder_G(decoder_input, decoder_hidden,
                                                                          encoder_outputs)
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

    def forward(self, pair, to_device=False):
        # pair为对话 {x, y} 类型为torch.tensor()
        x_length = pair[0].size(0)
        y_length = pair[1].size(0)

        if to_device:
            hidden = self.initHidden().to(device)
        else:
            hidden = self.initHidden()

        for i in range(x_length):
            embedded_x = self.embedding(pair[0][i]).view(1, 1, -1)
            _, hidden = self.gru1(embedded_x, hidden)
        hidden_x = hidden     # x句的编码结果

        if to_device:
            hidden = self.initHidden().to(device)
        else:
            hidden = self.initHidden()

        for i in range(y_length):
            embedded_y = self.embedding(pair[1][i]).view(1, 1, -1)
            _, hidden = self.gru1(embedded_y, hidden)
        hidden_y = hidden     # y句的编码结果

        if to_device:
            hidden = self.initHidden().to(device)
        else:
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
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.8)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate, momentum=0.8)

    ## training data loading ##
    training_pairs = [tensorFromPair(random.choice(TrainSet)) for i in range(batch_size)]

    criterion = nn.NLLLoss()

    for iter in range(1, batch_size + 1):
        ## data loading ##
        training_pair = training_pairs[iter-1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

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

                # 减小生成重复序列的概率
               topv, topi = decoder_output.topk(1)
               if di >0:
                   loss += criterion(decoder_output, target_tensor[di]) + torch.exp(decoder_output[0][prev_max])
               else:
                   loss += criterion(decoder_output, target_tensor[di])
               prev_max = topi.squeeze().item() # 记录前一次输出的分布中概率最大的单词

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                if di > 0:
                    loss += criterion(decoder_output, target_tensor[di]) + torch.exp(decoder_output[0][prev_max])
                else:
                    loss += criterion(decoder_output, target_tensor[di])
                    prev_max = topi.squeeze().item()  # 记录前一次输出的分布中概率最大的单词

                if decoder_input.item() == EOS_Token:
                    break

        ## BPTT & Parameters updating (every sentence)##
        total_loss += loss
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()


    # ## BPTT & Parameters updating (every batch)##
    #     total_loss += loss
    # total_loss.backward()
    # encoder_optimizer.step()
    # decoder_optimizer.step()

    print("Time consumed: {} Average loss: {:.2f} ".format(asMinutes(time.time()-start_time),
                                                          total_loss.item()/batch_size))


def pretrainD(modelD, learning_rate=0.01, batch_size=128, to_device=True):
    # prepare data
    pos_data = [tensorFromPair(random.choice(TrainSet), to_device=to_device) for _ in range(batch_size)]
    neg_data = [tensorFromPair(random.choice(GenSet), to_device=to_device) for _ in range(batch_size)]

    # define optimizer & criterion
    discOptimizer = optim.SGD(modelD.parameters(), lr=learning_rate, momentum=0.8)
    criterion = nn.NLLLoss()
    discOptimizer.zero_grad()

    # some predefined variable
    # 注意：规定 Discriminator 的输出概率的含义为 [positive_probability, negative_probability]
    if to_device:
        posTag = torch.tensor([0]).to(device)
        negTag = torch.tensor([1]).to(device)
    else:
        posTag = torch.tensor([0])
        negTag = torch.tensor([1])

    loss = 0
    start_time = time.time()

    for iter in range(batch_size):
        # choose positive or negative pair randomly
        pick_positive_data = True if random.random() < 0.5 else False
        if pick_positive_data:
            output = modelD(pos_data[iter],to_device=to_device)
            loss += criterion(output, posTag)
        else:
            output = modelD(neg_data[iter],to_device=to_device)
            loss += criterion(output, negTag)

    # BPTT & params updating
    loss.backward()
    discOptimizer.step()

    print("Time consumed: {} Batch loss: {:.2f} ".format(asMinutes(time.time()-start_time),
                                                          loss.item()))

########################### pretrain Generator #########################################
# Gen_encoder = EncoderG().to(device)
# Gen_decoder = DecoderG().to(device)
#
# try:
#     Gen_encoder.load_state_dict(torch.load("./ModelParams/Gen_encoder_params.pkl"))
#     Gen_decoder.load_state_dict(torch.load("./ModelParams/Gen_decoder_params.pkl"))
#     print("Model parameters loaded successfully.")
# except FileNotFoundError:
#     print("Model parameters loading failed.")
#
# Interrupt_Flag = False
# print("Start training...(You can stop training by enter 'Ctrl + C')")
# for epoch in range(1000):
#     if Interrupt_Flag:
#         print("Stop training...")
#         break
#     else:
#         try:
#             pretrainG(Gen_encoder, Gen_decoder, learning_rate=0.001)
#         except KeyboardInterrupt:
#             Interrupt_Flag = True
#
# try:
#     torch.save(Gen_encoder.state_dict(), "./ModelParams/Gen_encoder_params.pkl")
#     torch.save(Gen_decoder.state_dict(), "./ModelParams/Gen_decoder_params.pkl")
#     print("Model parameters saved successfully.")
# except:
#     print("Failed to save model parameters.")

#########################################################################################

# ############# Test Generator & Provide negative data for training Discriminator ##################
############### Test Generator (without beam search) #########################
# Gen_encoder = EncoderG()
# Gen_decoder = DecoderG()
#
# try:
#     Gen_encoder.load_state_dict(torch.load("./ModelParams/Gen_encoder_params.pkl"))
#     Gen_decoder.load_state_dict(torch.load("./ModelParams/Gen_decoder_params.pkl"))
#     print("Model parameters loaded successfully.")
# except FileNotFoundError:
#     print("Model parameters loading failed.")
#
# train_pairs = [tensorFromPair(random.choice(TrainSet),to_device=False) for i in range(5)]
# print("----------------Evaluation on training set: --------------------- ")
# for i in range(5):
#     Gen_output = GenForward(Gen_encoder,Gen_decoder,train_pairs[i][0])
#     print("--------------------------------------------------------")
#     print("<source>: {}\n<target>: {}\n<generated>: {}"
#       .format(index2sentence(train_pairs[i][0].squeeze().numpy(), index2word),
#               index2sentence(train_pairs[i][1].squeeze().numpy(), index2word),
#               index2sentence(Gen_output, index2word)))
#
# test_pairs = [tensorFromPair(random.choice(TestSet),to_device=False) for i in range(5)]
# print("----------------Evaluation on testing set: -----------------------")
# for i in range(5):
#     Gen_output = GenForward(Gen_encoder,Gen_decoder,test_pairs[i][0])
#     print("--------------------------------------------------------")
#     print("<source>: {}\n<target>: {}\n<generated>: {}"
#       .format(index2sentence(test_pairs[i][0].squeeze().numpy(), index2word),
#               index2sentence(test_pairs[i][1].squeeze().numpy(), index2word),
#               index2sentence(Gen_output, index2word)))

#################################################################################

################### Test Generator (with beam search) ###########################
# Gen_encoder = EncoderG()
# Gen_decoder = DecoderG()
#
# try:
#     Gen_encoder.load_state_dict(torch.load("./ModelParams/Gen_encoder_params.pkl"))
#     Gen_decoder.load_state_dict(torch.load("./ModelParams/Gen_decoder_params.pkl"))
#     print("Model parameters loaded successfully.")
# except FileNotFoundError:
#     print("Model parameters loading failed.")
#
# train_pairs = [tensorFromPair(random.choice(TrainSet),to_device=False) for i in range(5)]
# test_pairs = [tensorFromPair(random.choice(TestSet),to_device=False) for i in range(5)]
# beam_search_k = 2
#
# print("----------------Evaluation on training set: --------------------- ")
# for i in range(5):
#     Gen_outputs, prob_log = GenForward(Gen_encoder,
#                                        Gen_decoder,
#                                        train_pairs[i][0],
#                                        if_beam_search=True,
#                                        beam_search_k=beam_search_k)
#
#     print("--------------------------------------------------------")
#     print("<Source>: {}\n<Target>: {}"
#       .format(index2sentence(train_pairs[i][0].squeeze().numpy(), index2word),
#               index2sentence(train_pairs[i][1].squeeze().numpy(), index2word)))
#     print("<Generated>: ")
#     for k in range(beam_search_k):
#         print("<Prob>: {}, <Sentence>: {}".format(math.exp(prob_log[k]),
#                                                   index2sentence(Gen_outputs[k], index2word)))
#
# print("----------------Evaluation on testing set: -----------------------")
# for i in range(5):
#     Gen_outputs, prob_log = GenForward(Gen_encoder,
#                                        Gen_decoder,
#                                        test_pairs[i][0],
#                                        if_beam_search=True,
#                                        beam_search_k=beam_search_k)
#     print("--------------------------------------------------------")
#     print("<Source>: {}\n<Target>: {}"
#       .format(index2sentence(test_pairs[i][0].squeeze().numpy(), index2word),
#               index2sentence(test_pairs[i][1].squeeze().numpy(), index2word)))
#     print("<Generated>: ")
#     for k in range(beam_search_k):
#         print("<Prob>: {}, <Sentence>: {}".format(math.exp(prob_log[k]),
#                                                   index2sentence(Gen_outputs[k], index2word)))


#################################################################################


################### generate negative data ########################################################
# with open("./data/generated_dialogue.txt", 'w', encoding='utf-8') as f:
#     for i in range(100):
#         s = ''
#         x, _ = tensorFromPair(random.choice(TrainSet),to_device=False)
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

########################### Pretrain Discriminator ############################################
# Discriminator = hierEncoder().to(device)
#
# try:
#     Discriminator.load_state_dict(torch.load("./ModelParams/Disc_params.pkl"))
#     print("Model parameters loaded successfully.")
# except FileNotFoundError:
#     print("Model parameters loading failed.")
#
# Interrupt_Flag = False
# print("Start training...(You can stop training by enter 'Ctrl + C')")
# for epoch in range(1000):
#     if Interrupt_Flag:
#         print("Stop training...")
#         break
#     else:
#         try:
#             pretrainD(Discriminator, learning_rate=0.001)
#         except KeyboardInterrupt:
#             Interrupt_Flag = True
#
# try:
#     torch.save(Discriminator.state_dict(), "./ModelParams/Disc_params.pkl")
#     print("Model parameters saved successfully.")
# except:
#     print("Failed to save model parameters.")

###############################################################################################

#################################### Test Discriminator #######################################
# DiscModel = hierEncoder()
#
# try:
#     DiscModel.load_state_dict(torch.load("./ModelParams/Disc_params.pkl"))
#     print("Model parameters loaded successfully.")
# except FileNotFoundError:
#     print("Model parameters loading failed.")
#
# # 测试时的模型不需要在cuda上运行，所以生成测试数据时传入参数 to_device=False
# posData = [tensorFromPair(random.choice(TrainSet), to_device=False) for _ in range(5)]
# negData = [tensorFromPair(random.choice(GenSet), to_device=False) for _ in range(5)]
#
# print("----------------Evaluation on positive pairs: --------------------- ")
# print("----------------DiscOutput denotes: [p(positive), p(negative)] ----------------- ")
# for i in range(5):
#     Disc_output = torch.exp(DiscModel(posData[i]))
#     print("--------------------------------------------------------")
#     print("<source>: {}\n<target>: {}\n<DiscOutput>: {}"
#       .format(index2sentence(posData[i][0].squeeze().numpy(), index2word),
#               index2sentence(posData[i][1].squeeze().numpy(), index2word),
#               Disc_output))
#
# print("----------------Evaluation on negative pairs: --------------------- ")
# print("----------------DiscOutput denotes: [p(positive), p(negative)] ----------------- ")
# for i in range(5):
#     Disc_output = torch.exp(DiscModel(negData[i]))
#     print("--------------------------------------------------------")
#     print("<source>: {}\n<target>: {}\n<DiscOutput>: {}"
#       .format(index2sentence(negData[i][0].squeeze().numpy(), index2word),
#               index2sentence(negData[i][1].squeeze().numpy(), index2word),
#               Disc_output))

###############################################################################################

######################################### REINFORCE #########################################

def REINFORCE_TRAINING(ModelGEncoder,
                       ModelGDecoder,
                       ModelD,
                       num_iter=1,
                       D_steps=5,
                       G_steps=1,
                       batch_size=128,
                       learning_rate=0.01,
                       if_use_MC=False,
                       if_use_critic=False,
                       teacher_forcing_ratio=0.5,
                       max_length=MaxLength):

    '''num_iter: 训练迭代次数
       D_steps: 每次迭代训练Discriminator次数
       G_steps: 每次迭代训练Generator次数
       teacher_forcing_rate: 训练Generator时采用teacher_forcing的概率'''

    DiscOptimizer = optim.SGD(ModelD.parameters(), lr=learning_rate, momentum=0.8)

    CriterionNLLLoss = nn.NLLLoss()

    posTag = torch.tensor([0]).to(device)
    negTag = torch.tensor([1]).to(device)

    for iter in range(num_iter):

        ############## Train Discriminator ##############
        print("-------- < Train Discriminator... > --------")
        total_loss = 0
        start_time = time.time()
        for di in range(D_steps):
            # Sample {(x,y)} from real data
            for ei in range(batch_size):
                # Sample (x, y_hat) from Generator
                pos_pair = tensorFromPair(random.choice(TrainSet))
                neg_y = GenForward(ModelGEncoder, ModelGDecoder, pos_pair[0], to_device=True)
                neg_y = torch.tensor(neg_y, dtype=torch.long).view(-1, 1).to(device)
                neg_pair = [pos_pair[0], neg_y]

                # Update D
                loss = 0  # 单个样本的loss
                DiscOptimizer.zero_grad()
                output = ModelD(pos_pair, to_device=True)
                loss += CriterionNLLLoss(output, posTag)
                loss.backward()
                DiscOptimizer.step()
                total_loss += loss

                loss = 0
                DiscOptimizer.zero_grad()
                output = ModelD(neg_pair, to_device=True)
                loss += CriterionNLLLoss(output, negTag)
                loss.backward()
                DiscOptimizer.step()
                total_loss += loss

        print("Time Consumed: <{}>\nBatch Loss: <{:.4f}> ".format(asMinutes(time.time() - start_time),
                                                               total_loss.item()/D_steps))

        ################ Train Generator ###############
        print("-------- < Train Generator... > --------")
        Total_NLLLoss = 0
        Total_Expected_Reward = 0
        teacher_forcing_training_count = 0   # 用来计算平均NLLLoss
        reinforce_count = 0   # 用来计算平均reward
        start_time = time.time()

        for gi in range(G_steps):

            rewards = []
            for ei in range(batch_size):
                # Sample (x, y) from real data
                real_pair = tensorFromPair(random.choice(TrainSet))
                # Sample (x, y_hat) from generator
                gen_y = GenForward(ModelGEncoder, ModelGDecoder, real_pair[0], to_device=True)
                gen_y = torch.tensor(gen_y, dtype=torch.long).view(-1, 1).to(device)
                gen_pair = [real_pair[0], gen_y]

                # Compute reward r for {(x, y_hat)} using D
                reward = torch.exp(ModelD(gen_pair, to_device=True)).squeeze()[0].item()
                rewards.append(reward)

                # Update G
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

                # 采用teacher_forcing
                nllloss = 0  # 单个样本(一句话)的loss
                if use_teacher_forcing:

                    teacher_forcing_training_count += 1

                    GenEncoderOptimizer = optim.SGD(ModelGEncoder.parameters(), lr=learning_rate, momentum=0.8)
                    GenDecoderOptimizer = optim.SGD(ModelGDecoder.parameters(), lr=learning_rate, momentum=0.8)

                    GenEncoderOptimizer.zero_grad()
                    GenDecoderOptimizer.zero_grad()

                    input_tensor = real_pair[0]
                    target_tensor = real_pair[1]

                    encoder_hidden = ModelGEncoder.initHidden().to(device)

                    input_length = input_tensor.size(0)
                    target_length = target_tensor.size(0)

                    encoder_outputs = torch.zeros(max_length, ModelGEncoder.hidden_size).to(device)
                    ## encoding ##
                    for i in range(input_length):
                        encoder_output, encoder_hidden = ModelGEncoder(input_tensor[i], encoder_hidden)
                        encoder_outputs[i] = encoder_output[0][0]

                    decoder_input = torch.tensor([SOS_Token]).to(device)
                    decoder_hidden = encoder_hidden.to(device)

                    ## decoding ##
                    for di in range(target_length):
                        decoder_output, decoder_hidden, decoder_attention = ModelGDecoder(
                            decoder_input, decoder_hidden, encoder_outputs)

                        decoder_input = target_tensor[di]  # detach from history as input

                        # 减小生成重复序列的概率
                        topv, topi = decoder_output.topk(1)
                        if di > 0:
                            nllloss += CriterionNLLLoss(decoder_output, target_tensor[di]) + torch.exp(
                                decoder_output[0][prev_max])
                        else:
                            nllloss += CriterionNLLLoss(decoder_output, target_tensor[di])
                        prev_max = topi.squeeze().item()  # 记录前一次输出的分布中概率最大的单词

                        if decoder_input.item() == EOS_Token:
                            break

                    ## BPTT & Parameters updating (every sentence)##
                    Total_NLLLoss += nllloss
                    nllloss.backward()
                    GenEncoderOptimizer.step()
                    GenDecoderOptimizer.step()

                # REINFORCE
                else:
                    reinforce_count += 1
                    probability = 0  # 生成一句话的log概率
                    expected_reward = 0  # 一句话的期望reward

                    if if_use_critic:
                        # 训练critic并使用critic计算baseline
                        pass

                    else:

                        # 使用reward平均值作为baseline
                        baseline = sum(rewards)/len(rewards)

                        # 根据原公式采用REINFORCE算法计算梯度时（r - b）并不参与求导，只相当于改变了learning_rate
                        if (reward - baseline) > 0:
                            # as_lr = learning_rate * (reward - baseline)
                            as_lr = learning_rate * (1 + reward)
                        else:
                            as_lr = learning_rate * reward

                        GenEncoderOptimizer = optim.SGD(ModelGEncoder.parameters(), lr=as_lr, momentum=0.8)
                        GenDecoderOptimizer = optim.SGD(ModelGDecoder.parameters(), lr=as_lr, momentum=0.8)

                        GenEncoderOptimizer.zero_grad()
                        GenDecoderOptimizer.zero_grad()

                        # forward propagation
                        input_tensor = real_pair[0]
                        encoder_hidden = ModelGEncoder.initHidden().to(device)
                        input_length = input_tensor.size(0)
                        encoder_outputs = torch.zeros(max_length, ModelGEncoder.hidden_size).to(device)
                        ## encoding ##
                        for i in range(input_length):
                            encoder_output, encoder_hidden = ModelGEncoder(input_tensor[i], encoder_hidden)
                            encoder_outputs[i] = encoder_output[0][0]

                        decoder_input = torch.tensor([SOS_Token]).to(device)
                        decoder_hidden = encoder_hidden.to(device)
                        ## decoding ##
                        for di in range(max_length):
                            decoder_output, decoder_hidden, decoder_attention = ModelGDecoder(
                                decoder_input, decoder_hidden, encoder_outputs)
                            topv, topi = decoder_output.topk(1)
                            decoder_input = topi.squeeze().detach()  # detach from history as input

                            # 计算生成 {y_hat} 的log概率
                            probability += topv.squeeze()

                            if decoder_input.item() == EOS_Token:
                                break


                        # 计算单句话的期望reward
                        expected_reward += reward * torch.exp(probability).squeeze().item()
                        # 计算总的期望reward
                        Total_Expected_Reward += expected_reward
                        # 由于REINFORCE的目标是最大化期望的reward，所以将log(p)取反
                        probability = -probability

                        # back propagation & update params
                        probability.backward()
                        GenEncoderOptimizer.step()
                        GenDecoderOptimizer.step()

        print("Time Consumed: <{}>\nAverage NLLLoss : <{:.2f}>\nExpected Reward Approximation: <{:.2f}> "
              .format(asMinutes(time.time() - start_time),
                      Total_NLLLoss.squeeze().item() / teacher_forcing_training_count,
                      Total_Expected_Reward / reinforce_count))
        print("-----------------------------------------------------------------------------------")


########### REINFORCE training ###########

Gen_encoder = EncoderG().to(device)
Gen_decoder = DecoderG().to(device)

try:
    Gen_encoder.load_state_dict(torch.load("./ModelParams/Gen_encoder_params.pkl"))
    Gen_decoder.load_state_dict(torch.load("./ModelParams/Gen_decoder_params.pkl"))
    print("Generator Model parameters loaded successfully.")
except FileNotFoundError:
    print("Generator Model parameters loading failed.")


Discriminator = hierEncoder().to(device)

try:
    Discriminator.load_state_dict(torch.load("./ModelParams/Disc_params.pkl"))
    print("Discriminator Model parameters loaded successfully.")
except FileNotFoundError:
    print("Discriminator Model parameters loading failed.")


Interrupt_Flag = False
print("Start training...(You can stop training by enter 'Ctrl + C')")
for epoch in range(1000):
    if Interrupt_Flag:
        print("Stop training...")
        break
    else:
        try:
            REINFORCE_TRAINING(Gen_encoder, Gen_decoder, Discriminator)
        except KeyboardInterrupt:
            Interrupt_Flag = True

try:
    torch.save(Gen_encoder.state_dict(), "./ModelParams/Gen_encoder_params.pkl")
    torch.save(Gen_decoder.state_dict(), "./ModelParams/Gen_decoder_params.pkl")
    print("Generator Model parameters saved successfully.")
except:
    print("Failed to save Generator model parameters.")

try:
    torch.save(Discriminator.state_dict(), "./ModelParams/Disc_params.pkl")
    print("Discriminator Model parameters saved successfully.")
except:
    print("Failed to save Discriminator model parameters.")






































