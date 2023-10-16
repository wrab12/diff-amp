
import numpy as np
import torch
import pandas as pd
from Bio import SeqIO
import torch.utils.data
from sklearn.model_selection import KFold, ShuffleSplit
import re
import vocab
from sklearn.model_selection import train_test_split
# from transformers import T5Tokenizer,XLNetTokenizer
def AAI_embedding(seq,max_len=200):
    f=open('data/AAindex.txt')
    text=f.read()
    f.close()
    text=text.split('\n')
    while '' in text:
        text.remove('')
    cha=text[0].split('\t')
    while '' in cha:
        cha.remove('')
    cha=cha[1:]
    index=[]
    for i in range(1,len(text)):
        temp=text[i].split('\t')
        while '' in temp:
            temp.remove('')
        temp=temp[1:]
        for j in range(len(temp)):
            temp[j]=float(temp[j])
        index.append(temp)
    index=np.array(index)
    AAI_dict={}
    for j in range(len(cha)):
        AAI_dict[cha[j]]=index[:,j]
    AAI_dict['X']=np.zeros(531)
    all_embeddings=[]
    for each_seq in seq:
        temp_embeddings=[]
        for each_char in each_seq:
            temp_embeddings.append(AAI_dict[each_char])
        if max_len>len(each_seq):
            zero_padding=np.zeros((max_len-len(each_seq),531))
            data_pad=np.vstack((temp_embeddings,zero_padding))
        elif max_len==len(each_seq):
            data_pad=temp_embeddings
        else:
            data_pad=temp_embeddings[:max_len]
        all_embeddings.append(data_pad)
    all_embeddings=np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()


def PAAC_embedding(seq,max_len=200):
    f=open('data/PAAC.txt')
    text=f.read()
    f.close()
    text=text.split('\n')
    while '' in text:
        text.remove('')
    cha=text[0].split('\t')
    while '' in cha:
        cha.remove('')
    cha=cha[1:]
    index=[]
    for i in range(1,len(text)):
        temp=text[i].split('\t')
        while '' in temp:
            temp.remove('')
        temp=temp[1:]
        for j in range(len(temp)):
            temp[j]=float(temp[j])
        index.append(temp)
    index=np.array(index)
    AAI_dict={}
    for j in range(len(cha)):
        AAI_dict[cha[j]]=index[:,j]
    AAI_dict['X']=np.zeros(3)
    all_embeddings=[]
    for each_seq in seq:
        temp_embeddings=[]
        for each_char in each_seq:
            temp_embeddings.append(AAI_dict[each_char])
        if max_len>len(each_seq):
            zero_padding=np.zeros((max_len-len(each_seq),3))
            data_pad=np.vstack((temp_embeddings,zero_padding))
        elif max_len==len(each_seq):
            data_pad=temp_embeddings
        else:
            data_pad=temp_embeddings[:max_len]
        all_embeddings.append(data_pad)
    all_embeddings=np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()

def PC6_embedding(seq,max_len=200):
    f=open('data/6-pc')
    text=f.read()
    f.close()
    text=text.split('\n')
    while '' in text:
        text.remove('')
    text=text[1:]
    AAI_dict={}
    for each_line in text:
        temp=each_line.split(' ')
        while '' in temp:
            temp.remove('')
        for i in range(1,len(temp)):
            temp[i]=float(temp[i])
        AAI_dict[temp[0]]=temp[1:]
    AAI_dict['X']=np.zeros(6)
    all_embeddings=[]
    for each_seq in seq:
        temp_embeddings=[]
        for each_char in each_seq:
            temp_embeddings.append(AAI_dict[each_char])
        if max_len>len(each_seq):
            zero_padding=np.zeros((max_len-len(each_seq),6))
            data_pad=np.vstack((temp_embeddings,zero_padding))
        elif max_len==len(each_seq):
            data_pad=temp_embeddings
        else:
            data_pad=temp_embeddings[:max_len]
        all_embeddings.append(data_pad)
    all_embeddings=np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()



def BLOSUM62_embedding(seq,max_len=200):
    f=open('data/blosum62.txt')
    text=f.read()
    f.close()
    text=text.split('\n')
    while '' in text:
        text.remove('')
    cha=text[0].split(' ')
    while '' in cha:
        cha.remove('')
    index=[]
    for i in range(1,len(text)):
        temp=text[i].split(' ')
        while '' in temp:
            temp.remove('')
        for j in range(len(temp)):
            temp[j]=float(temp[j])
        index.append(temp)
    index=np.array(index)
    BLOSUM62_dict={}
    for j in range(len(cha)):
        BLOSUM62_dict[cha[j]]=index[:,j]
    all_embeddings=[]
    for each_seq in seq:
        temp_embeddings=[]
        for each_char in each_seq:
            temp_embeddings.append(BLOSUM62_dict[each_char])
        if max_len>len(each_seq):
            zero_padding=np.zeros((max_len-len(each_seq),23))
            data_pad=np.vstack((temp_embeddings,zero_padding))
        elif max_len==len(each_seq):
            data_pad=temp_embeddings
        else:
            data_pad=temp_embeddings[:max_len]
        all_embeddings.append(data_pad)
    all_embeddings=np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()


def onehot_embedding(seq,max_len=200):
    char_list='ARNDCQEGHILKMFPSTWYVX'
    char_dict={}
    for i in range(len(char_list)):
        char_dict[char_list[i]]=i
    all_embeddings=[]
    for each_seq in seq:
        temp_embeddings=[]
        for each_char in each_seq:
            codings=np.zeros(21)
            if each_char in char_dict.keys():
                codings[char_dict[each_char]]=1
            else:
                codings[20]=1
            temp_embeddings.append(codings)
        if max_len>len(each_seq):
            zero_padding=np.zeros((max_len-len(each_seq),21))
            data_pad=np.vstack((temp_embeddings,zero_padding))
        elif max_len==len(each_seq):
            data_pad=temp_embeddings
        else:
            data_pad=temp_embeddings[:max_len]
              
        all_embeddings.append(data_pad)
    all_embeddings=np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()
        



def index_encoding(sequences,max_len=200):
    '''
    Modified from https://github.com/openvax/mhcflurry/blob/master/mhcflurry/amino_acid.py#L110-L130

    Parameters
    ----------
    sequences: list of equal-length sequences

    Returns
    -------
    np.array with shape (#sequences, length of sequences)
    '''
    seq_list=[]
    for s in sequences:
        temp=list(s)
        while len(temp)<max_len:
            temp.append(20)
        temp=temp[:max_len]
        seq_list.append(temp)

    df = pd.DataFrame(seq_list)
    encoding = df.replace(vocab.AMINO_ACID_INDEX)
    encoding = encoding.values.astype(int)
    return encoding







class MetagenesisData(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Dataset(object):
    def __init__(self,
            fasta=None,
            label=None,
            sep=',',
            use_phy_feat=False,
            phy_feat_fl=None,random_seed=42):


        self.fasta = fasta


        self.sep=sep
        self.use_phy_feat = use_phy_feat
        self.rng = np.random.RandomState(random_seed)
        self.native_sequence =fasta

        if use_phy_feat==True:
            self.phy_feat_fl=phy_feat_fl
            self.phy_feat=self._read_features()
            
            
    def _read_features(self):
        df=pd.read_csv(self.phy_feat_fl,sep='\t',index_col=0)
        feat=df.values
        feat=torch.from_numpy(feat.astype(np.float32))

        return feat
            
    def _read_native_sequence(self):
        sequence=[]
        for seq_record in SeqIO.parse(self.fasta, "fasta"):
            sequence.append(str(seq_record.seq).upper())

        return sequence
    
    def _read_labels(self):

        df=pd.read_csv(self.label,sep=self.sep,index_col=0)
        labels=df.values
        return labels
    


    def encode_seq_enc_onehot(self, sequences,max_length):
        seq_enc = onehot_embedding(sequences, max_len=max_length)
        return seq_enc
    def encode_seq_enc_BLOSUM62(self, sequences,max_length):
        seq_enc = BLOSUM62_embedding(sequences, max_len=max_length)
        return seq_enc
    def encode_seq_enc_AAI(self, sequences,max_length):
        seq_enc = AAI_embedding(sequences, max_len=max_length)
        return seq_enc
    def encode_seq_enc_PAAC(self, sequences,max_length):
        seq_enc = PAAC_embedding(sequences, max_len=max_length)
        return seq_enc
    def encode_seq_enc_PC6(self, sequences,max_length):
        seq_enc = PC6_embedding(sequences, max_len=max_length)
        return seq_enc
    # def encode_seq_enc_bertT5(self, sequences,max_length):
    #     seq_enc = bertT5_embedding(sequences, max_len=max_length)
    #     return seq_enc

    def encode_seq_enc(self, sequences,max_length):

        seq_enc = index_encoding(sequences,max_length)
        seq_enc = torch.from_numpy(seq_enc.astype(float))
        return seq_enc
    
    # def encode_seq_token(self,sequences):
    #     seq_enc=[]
    #     for each_seq in sequences:
    #         seq_enc.append(self.tokenizer.encode(each_seq))
    #     return seq_enc   


    # def encode_glob_feat(self, sequences):
    #     feat = self.tape_encoder.encode(sequences)
    #     feat = torch.from_numpy(feat).float()
    #     return feat

    def build_data(self, max_length):
        
        sequences = self.native_sequence
        seq_enc_onehot = self.encode_seq_enc_onehot(sequences,max_length=max_length)

        
        # train_seq_enc_bertencoding,train_seq_mask = self.encode_seq_enc_bertT5(train_sequences, max_length=max_length)
        # valid_seq_enc_bertencoding,valid_seq_mask = self.encode_seq_enc_bertT5(valid_sequences, max_length=max_length)
        # test_seq_enc_bertencoding,test_seq_mask = self.encode_seq_enc_bertT5(test_sequences, max_length=max_length)
        
        seq_enc_blosum62 = self.encode_seq_enc_BLOSUM62(sequences,max_length=max_length)
        seq_enc_AAI = self.encode_seq_enc_AAI(sequences,max_length=max_length)
        seq_enc_PAAC = self.encode_seq_enc_PAAC(sequences,max_length=max_length)
        seq_enc_PC6 = self.encode_seq_enc_PC6(sequences,max_length=max_length)
        seq_enc = self.encode_seq_enc(sequences,max_length=max_length)    



        samples = []

        # print(labels)
        print(len(sequences))
        for i in range(len(sequences)):
            sample = {
                'sequence':sequences[i],

                'seq_enc': seq_enc[i],
                'seq_enc_onehot': seq_enc_onehot[i],
                # 'seq_enc_bert':train_seq_enc_bertencoding[i],
                # 'seq_enc_mask':train_seq_mask[i],
                'seq_enc_pc6': seq_enc_PC6[i],

                'seq_enc_BLOSUM62': seq_enc_blosum62[i],
                'seq_enc_PAAC': seq_enc_PAAC[i],
                'seq_enc_AAI': seq_enc_AAI[i]
            }
            if self.use_phy_feat:
                sample['phy_feat'] = self.phy_feat[i]
            samples.append(sample)
        data = MetagenesisData(samples)

        
        return data


    def get_dataloader(self,  max_length=200,batch_size=128,
                       resample_train_valid=False):
        
        data = self.build_data(max_length)
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)

        return data_loader
        # return data
import torch
import torch.nn as nn
device = torch.device("cpu")
    
if __name__ == '__main__':

    dataset = Dataset(
        fasta='static/uploads/Insect.fasta',
        sep=',')
    train_loader = dataset.get_dataloader(
        batch_size=32,max_length=200)
    # y_pred=[]
    y_true=[]
    all_seqs=[]
    # for batch in train_loader:
    #     seq=batch['sequence'].to(device)
    #     y = batch['label'].to(device)
    #     x = batch['seq_enc'].to(device).int()
    #     # AAI_feat = batch['seq_enc_AAI'].to(device)
    #     # onehot_feat = batch['seq_enc_onehot'].to(device)
    #     # BLOSUM62_feat = batch['seq_enc_BLOSUM62'].to(device)
    #     # PAAC_feat = batch['seq_enc_PAAC'].to(device)
    #     # # bert_feat=batch['seq_enc_bert'].to(device)
    #     # # bert_mask=batch['seq_enc_mask'].to(device)
    #     # outputs=net(AAI_feat,onehot_feat,BLOSUM62_feat,PAAC_feat)
    #     # outputs = model(x)
    #     # y_pred.extend(outputs.cpu().numpy())
    #     y_true.extend(y.cpu().numpy())
    #     all_seqs.extend(seq.cpu().numpy())
#     print(df.head())
#     print(len(loader.__iter__()))
#     (loader, df), (_, _) = dataset.get_dataloader('train_valid',
#         batch_size=32, return_df=True, resample_train_valid=True)
#     print(df.head())
    # print(len(train_loader.__iter__()))
    # loader, df = dataset.get_dataloader('test',
    #     batch_size=32, return_df=True, resample_train_valid=True)
    # print(next(loader.__iter__()))

# sequence = ['GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQ','GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQ']
# seq_tape=AAI_embedding(sequence)

