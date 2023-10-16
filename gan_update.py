import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import gan_diff  # 导入原始模型的定义，确保与之前训练的模型相匹配
import tqdm
import sys

CUDA = torch.cuda.is_available()
AX_SEQ_LEN = 18
BATCH_SIZE = 512
data = pd.read_csv('selected_data.csv')
all_sequences = np.asarray(data['sequence'])
all_data = []
for i in range(len(all_sequences)):
    all_data.append(gan_diff.sequence_to_vector(all_sequences[i]))

VOCAB_SIZE = 26
OCAB_SIZE = 26
MAX_SEQ_LEN = 18
START_LETTER = 0
POS_NEG_SAMPLES = len(all_data)
BATCH_SIZE = 16
ADV_TRAIN_EPOCHS = 1  # 原始100
MLE_TRAIN_EPOCHS = 1  # 原始50
GEN_EMBEDDING_DIM = 3
GEN_HIDDEN_DIM = 128
NUM_PG_BATCHES = 1
GEN_lr = 0.00005
dis_lr = 0.00005
DIS_EMBEDDING_DIM = 3
DIS_HIDDEN_DIM = 128
D_STEPS = 1  # 原始30
D_EPOCHS = 1  # 原始10
ADV_D_EPOCHS = 5
ADV_D_STEPS = 1

gen_model = 'gen_500.pth'
dis_model = 'dis_500.pth'

if __name__ == '__main__':
    oracle = gan_diff.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA, oracle_init=True)
    gen = gan_diff.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA).cuda()
    dis = gan_diff.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA).cuda()

    # loss_g = []
    # loss_d = []

    if CUDA:
        oracle = oracle.cuda()
        gen = gen.cuda()
        dis = dis.cuda()

        oracle_samples = torch.Tensor(all_data).type(torch.LongTensor)
        oracle_samples = oracle_samples.cuda()
    else:
        oracle_samples = torch.IntTensor(all_data).type(torch.LongTensor)
    gen.load_state_dict(torch.load(r'models/gen_500.pth', map_location=torch.device('cpu')))
    dis.load_state_dict(torch.load(r'models/dis_500.pth', map_location=torch.device('cpu')))

    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=GEN_lr)
    gan_diff.train_generator_MLE(gen, gen_optimizer, oracle, oracle_samples, MLE_TRAIN_EPOCHS)
    print('Finished Generator MLE Training...')

    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adam(dis.parameters(), lr=dis_lr)  # adagrad
    gan_diff.train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, D_STEPS, D_EPOCHS)

    print('\nStarting Adversarial Training...')
    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch + 1))
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        gan_diff.train_generator_PG(gen, gen_optimizer, oracle, dis, NUM_PG_BATCHES)
        print('\nAdversarial Training Discriminator : ')
        gan_diff.train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, ADV_D_STEPS, ADV_D_EPOCHS)

    torch.save(gen.state_dict(), './models/' + gen_model)
    torch.save(dis.state_dict(), './models/' + dis_model)

    print('\n Update training completed successfully.Model saved.')
