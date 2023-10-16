import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init
from math import ceil
import numpy as np
import sys
import torch.optim as optim
import pandas as pd
import re
import pickle
import ECD_Attention
import numpy as np
import matplotlib.pyplot as plt
import tqdm
MAX_SEQ_LEN = 18
BATCH_SIZE = 512
data = pd.read_csv('data/AMPdb_data.csv', skiprows=1, usecols=range(3), header=None, names=['ID', 'seq', 'len'])
all_sequences = np.asarray(data['seq'])

CHARACTER_DICT = {
    'A': 1, 'C': 2, 'E': 3, 'D': 4, 'F': 5, 'I': 6, 'H': 7,
    'K': 8, 'M': 9, 'L': 10, 'N': 11, 'Q': 12, 'P': 13, 'S': 14,
    'R': 15, 'T': 16, 'W': 17, 'V': 18, 'Y': 19, 'G': 20, '0': 21, 'U': 22, 'Z': 23, 'X': 24}
INDEX_DICT = {
    1: 'A', 2: 'C', 3: 'E', 4: 'D', 5: 'F', 6: 'I', 7: 'H',
    8: 'K', 9: 'M', 10: 'L', 11: 'N', 12: 'Q', 13: 'P', 14: 'S',
    15: 'R', 16: 'T', 17: 'W', 18: 'V', 19: 'Y', 20: 'G', 21: '0', 22: 'U', 23: 'Z', 24: 'X'}


def sequence_to_vector(sequence):
    default = np.asarray([25] * (MAX_SEQ_LEN))
    for i, character in enumerate(sequence[:MAX_SEQ_LEN]):
        default[i] = CHARACTER_DICT[character]
    return default.astype(int)


def vector_to_sequence(vector):
    return ''.join([INDEX_DICT.get(item, '0') for item in vector])


all_data = []
for i in range(len(all_sequences)):
    all_data.append(sequence_to_vector(all_sequences[i]))


class Generator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, oracle_init=False):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.diff = difformer.DIFFormer(embedding_dim, hidden_dim, out_channels=128, use_graph=False)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)

        if oracle_init:
            for p in self.parameters():
                nn.init.normal_(p, 0, 1)

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, inp, hidden):
        emb = self.embeddings(inp)
        emb = emb.view(1, -1, self.embedding_dim)
        emb1 = emb.view(-1, self.embedding_dim)
        z = self.diff(emb1, edge_index=None)
        # print("z={}".format(z.size()))  # 16,128
        # out, hidden = self.gru(emb, hidden)  # 1,16,128  # 1,1,128
        out = self.gru2out(z.view(-1, self.hidden_dim))  # 16,26
        out = F.log_softmax(out, dim=1)  # 16.26
        # out, hidden = self.gru(emb, hidden)
        # out = self.gru2out(out.view(-1, self.hidden_dim))
        # out = F.log_softmax(out, dim=1)
        return out, hidden

    def sample(self, num_samples, start_letter=0):
        samples = torch.zeros(num_samples, self.max_seq_len).type(torch.LongTensor)
        samples_p = torch.zeros(num_samples, self.max_seq_len).type(torch.FloatTensor)

        h = self.init_hidden(num_samples)  # (1,100,128)
        inp = autograd.Variable(torch.LongTensor([start_letter] * num_samples))

        if self.gpu:
            samples = samples.cuda()
            inp = inp.cuda()

        for i in range(self.max_seq_len):
            out, h = self.forward(inp, h)
            out_p, _ = torch.max(torch.exp(out), dim=1)
            out = torch.multinomial(torch.exp(out), 1)
            samples_p[:, i] = out_p
            samples[:, i] = out.view(-1).data
            inp = out.view(-1)
        return samples, samples_p

    def batchNLLLoss(self, inp, target):
        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)
        target = target.permute(1, 0)
        h = self.init_hidden(batch_size)

        loss = 0

        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            loss += loss_fn(out, target[i])

        return loss  # per batch

    def batchPGLoss(self, inp, target, reward):
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)
        target = target.permute(1, 0)
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            for j in range(batch_size):
                loss += -out[j][target.data[i][j]] * reward[j]

        return loss / batch_size


class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, dropout=0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2 * 2 * hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2 * 2 * 1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, input, hidden):
        emb = self.embeddings(input)
        emb = emb.permute(1, 0, 2)
        _, hidden = self.gru(emb, hidden)
        hidden = hidden.permute(1, 0, 2).contiguous()
        out = self.gru2hidden(hidden.view(-1, 4 * self.hidden_dim))
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)
        out = torch.sigmoid(out)
        return out

    def batchClassify(self, inp):
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return out.view(-1)

    def batchBCELoss(self, inp, target):
        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return loss_fn(out, target)


def prepare_generator_batch(samples, start_letter=0, gpu=False):
    batch_size, seq_len = samples.size()
    inp = torch.zeros(batch_size, seq_len)
    target = samples
    inp[:, 0] = start_letter
    inp[:, 1:] = target[:, :seq_len - 1]

    inp = inp.type(torch.LongTensor)
    target = target.type(torch.LongTensor)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


def prepare_discriminator_data(pos_samples, neg_samples, gpu=False):
    inp = torch.cat((pos_samples, neg_samples), 0).type(torch.LongTensor)
    target = torch.ones(pos_samples.size()[0] + neg_samples.size()[0])
    target[pos_samples.size()[0]:] = 0
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]
    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


def batchwise_sample(gen, num_samples, batch_size):
    samples = []
    for i in range(int(ceil(num_samples / float(batch_size)))):
        samples.append(gen.sample(batch_size))

    return torch.cat(samples, 0)[:num_samples]


def batchwise_oracle_nll(gen, oracle, num_samples, batch_size, max_seq_len, start_letter=0, gpu=False):
    s = batchwise_sample(gen, num_samples, batch_size)
    oracle_nll = 0
    for i in range(0, num_samples, batch_size):
        inp, target = prepare_generator_batch(s[i:i + batch_size], start_letter, gpu)
        oracle_loss = oracle.batchNLLLoss(inp, target) / max_seq_len
        oracle_nll += oracle_loss.data.item()

    return oracle_nll / (num_samples / batch_size)


def train_generator_MLE(gen, gen_opt, oracle, real_data_samples, epochs):
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
            inp, target = prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                  gpu=CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

            if (i / BATCH_SIZE) % ceil(
                    ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:
                print('.', end='')
                sys.stdout.flush()

        total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN

        print(' average_train_NLL = %.4f' % (total_loss))


def train_generator_PG(gen, gen_opt, oracle, dis, num_batches):
    for batch in range(num_batches):
        s = gen.sample(BATCH_SIZE * 2)
        inp, target = prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()


def train_discriminator(discriminator, dis_opt, real_data_samples, generator, oracle, d_steps, epochs):
    indice = random.sample(range(len(real_data_samples)), 100)
    indice = torch.tensor(indice)
    pos_val = real_data_samples[indice]
    neg_val = generator.sample(100)
    val_inp, val_target = prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):
        s = batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        dis_inp, dis_target = prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out > 0.5) == (target > 0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred > 0.5) == (val_target > 0.5)).data.item() / 200.))

            loss_d.append(total_loss)


CUDA = torch.cuda.is_available()

VOCAB_SIZE = 26
MAX_SEQ_LEN = 18
START_LETTER = 0
POS_NEG_SAMPLES = len(all_data)
GEN_EMBEDDING_DIM = 3
GEN_HIDDEN_DIM = 128
DIS_EMBEDDING_DIM = 3
DIS_HIDDEN_DIM = 128
num_outputs = 20000

if __name__ == '__main__':
    def euclidean_distance(a, b):
        return np.linalg.norm(a - b)

    oracle = Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA, oracle_init=True)

    gen = Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    dis = Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)

    loss_g = []
    loss_d = []

    if CUDA:
        oracle = oracle.cuda()
        gen = gen.cuda()
        dis = dis.cuda()

        oracle_samples = torch.Tensor(all_data).type(torch.LongTensor)

        oracle_samples = oracle_samples.cuda()

    else:
        oracle_samples = torch.IntTensor(all_data).type(torch.LongTensor)

    gen.load_state_dict(torch.load(r'weight/gen_500.pth', map_location=torch.device('cpu')))
    dis.load_state_dict(torch.load(r'weight/dis_500.pth', map_location=torch.device('cpu')))

    gen.eval()
    dis.eval()

    a, b = gen.sample(num_outputs)
    a = a.tolist()
    b = b.tolist()

    f = open('outputs.txt', 'w+')
    f_seq = open("seq.txt", 'w+')
    print('\nGeneration Start')






    for i in range(num_outputs):
        seq = (vector_to_sequence(a[i]))
        percent = (b[i])
        percent = np.array(percent)
        percent = np.round(percent, 4)
        percent = list(percent)
        ALP = sum(percent) / len(percent)

        seq = re.sub('[X]+$', '', seq)
        check_x = re.search('[0]', seq)
        f.write("%.2f" % ALP + ">" + str(i) + ">" + seq + ">" + str(percent) + '\n')
        f_seq.write(seq+'\n')
    print('\nGeneration Finished')
