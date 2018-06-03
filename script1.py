import torch
import torch.nn as nn
import torch.nn.functional as F



############################# PREDEFINED #########################
device = torch.device('cuda0' if torch.cuda.is_available() else 'cpu')
MaxLength = 15
data1 = []
data2 = []
index2word = {}
s = ''
t = ''


############################ Data Preparation #############################
with open("./data/s_given_t_test.txt", 'r', encoding='utf-8') as f:
    for line in f.readlines():
        data1.append(line.strip('\n').split("|"))

with open("./data/movie_25000", 'r', encoding='utf-8') as f:
    for i,word in enumerate(f.readlines()):
        index2word[str(i+1)] = word.strip('\n')                  # 注意：字典里的索引应该从1而不是0开始！

#with open('./data/s_given_t_dialogue_length2_3.txt', 'r', encoding='utf-8') as f:
#    for line in f.readlines():
#        data2.append(line.strip('\n').split('|'))


for i in range(len(data1)):
    data1[i] = [data1[i][0].split(" "), data1[i][1].split(" ")]

print(data1[255])

for index in data1[255][0]:
    s += index2word[index] + ' '
for index in data1[255][1]:
    t += index2word[index] + ' '

print(s + '\n' + t)

############################ Generator ###################################
class EncoderG(nn.Module):
    def __init__(self, vocab_size, embedding_size, drop_out = 0):
        super(EncoderG, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.drop_out = drop_out

        self.embedding = nn.Embeddding(self.vocab_size, self.embedding_size, drop_out = self.drop_out)
        self.gru = nn.GRU(self.embedding_size, self.embedding_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.embedding_size)

class DecoderG(nn.Module):
    def __init__(self, vocab_size, embedding_size, drop_out=0, max_length=MaxLength):
        super(DecoderG, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.drop_out = drop_out
        self.max_length = max_length

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.attn = nn.Linear(embedding_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.embedding_size*2, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.embedding_size, drop_out=self.drop_out)
        self.out = nn.Linear(self.embedding_size, self.vocab_size)

    def forward(self, hidden, word_in, encoder_outputs):
        embedded = self.embedding(word_in).view(1, 1, -1)
        attn_weights = F.softmax(self.attn(torch.cat((hidden[0], embedded[0]), dim=1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = self.attn_combine(torch.cat((embedded[0], attn_applied[0]), dim=1)).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zero(1, 1, self.embedding_size)











































