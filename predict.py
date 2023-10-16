from Bio import SeqIO
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import scipy.stats
import pathlib
import copy
import time
import vocab
from model import SequenceMultiTypeMultiCNN_1
from data_feature import Dataset
import pandas as pd
import argparse
from tqdm import tqdm

device = torch.device("cuda:0")


def return_y(data_iter, net):
    y_pred = []

    all_seq = []
    for batch in data_iter:
        all_seq += batch['sequence']

        AAI_feat = batch['seq_enc_AAI'].to(device)
        onehot_feat = batch['seq_enc_onehot'].to(device)
        BLOSUM62_feat = batch['seq_enc_BLOSUM62'].to(device)
        PAAC_feat = batch['seq_enc_PAAC'].to(device)
        # bert_feat=batch['seq_enc_bert'].to(device)
        # bert_mask=batch['seq_enc_mask'].to(device)
        outputs = net(AAI_feat, onehot_feat, BLOSUM62_feat, PAAC_feat)
        # print(outputs.size())#32,1
        # outputs = model(x)
        y_pred.extend(outputs.cpu().numpy())

    return y_pred, all_seq


def testing(batch_size, patience, n_epochs, testfasta, seq_len, cdhit_value, cv_number, save_file, model_file):
    model = SequenceMultiTypeMultiCNN_1(d_input=[531, 21, 23, 3], vocab_size=21, seq_len=seq_len,
                                        dropout=0.1, d_another_h=128, k_cnn=[2, 3, 4, 5, 6], d_output=1).to(device)

    dataset = Dataset(fasta=testfasta)
    test_loader = dataset.get_dataloader(batch_size=batch_size, max_length=seq_len)

    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu'))['state_dict'])
    model.eval()
    with torch.no_grad():
        new_y_pred, all_seq = return_y(test_loader, model)

    final_y_pred = copy.deepcopy(new_y_pred)

    final_y_pred = np.array(final_y_pred).T[0].tolist()

    pred_dict = {'seq': all_seq, 'predictions': final_y_pred}
    pred_df = pd.DataFrame(pred_dict)
    pred_df.to_csv(save_file, index=None)


all_function_names = ['antibacterial', 'antigram-positive', 'antigram-negative', 'antifungal', 'antiviral', \
     'anti_mammalian_cells', 'antihiv', 'antibiofilm', 'anticancer', 'antimrsa', 'antiparasitic', \
     'hemolytic', 'chemotactic', 'antitb', 'anurandefense', 'cytotoxic', \
     'endotoxin', 'insecticidal', 'antimalarial', 'anticandida', 'antiplasmodial', 'antiprotozoal']


# os.environ['CUDA_LAUNCH_BLOCKING'] = 1


def predict(test_file):
    fas_id = []
    fas_seq = []
    for seq_record in SeqIO.parse(test_file, "fasta"):
        fas_seq.append(str(seq_record.seq).upper())
        fas_id.append(str(seq_record.id))

    seq_len = 200
    batch_size = 32
    cdhit_value = 40
    vocab_size = len(vocab.AMINO_ACIDS)

    epochs = 300
    temp_save_AMP_filename = '%s ' % (time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
    for cv_number in tqdm(range(10)):
        testing(testfasta=fas_seq,
                model_file=f'models/AMP_1st/textcnn_cdhit_40_{cv_number}.pth.tar',
                save_file=f'tmp_save/{temp_save_AMP_filename}_{cv_number}.csv',
                batch_size=batch_size, patience=10, n_epochs=epochs, seq_len=seq_len, cdhit_value=cdhit_value
                , cv_number=cv_number)

    pred_prob = []
    for cv_number in tqdm(range(10)):
        df = pd.read_csv(f'tmp_save/{temp_save_AMP_filename}_{cv_number}.csv')
        data = df.values.tolist()
        temp = []
        for i in tqdm(range(len(data))):
            temp.append(data[i][1])
        pred_prob.append(temp)
    pred_prob = np.average(pred_prob, 0)
    pred_AMP_label = []
    for i in tqdm(range(len(pred_prob))):
        if pred_prob[i] > 0.5:
            pred_AMP_label.append('Yes')
        else:
            pred_AMP_label.append('No')

    for function_name in all_function_names:
        temp_dir_list = os.listdir('tmp_save')
        if function_name not in temp_dir_list:
            os.mkdir('tmp_save/ ' + function_name)
        for cv_number in tqdm(range(10)):
            testing(testfasta=fas_seq,
                    model_file=f'models/AMP_2nd/{function_name}/textcnn_cdhit_100_0.pth.tar',
                    save_file=f'tmp_save/{function_name}/{temp_save_AMP_filename}_{cv_number}.csv',
                    batch_size=batch_size, patience=10, n_epochs=epochs, seq_len=seq_len, cdhit_value=cdhit_value
                    , cv_number=cv_number)

    all_function_pred_label = []
    for function_name in all_function_names:

        function_threshold_df = pd.read_csv(f'models/AMP_2nd_threashold/{function_name}_yd_threshold.csv', index_col=0)
        function_thresholds = function_threshold_df.values[:, 0]

        each_function_data = []

        for cv_number in tqdm(range(10)):
            df = pd.read_csv(f'tmp_save/{function_name}/{temp_save_AMP_filename}_{cv_number}.csv')
            data = df.values.tolist()
            temp = []
            for i in tqdm(range(len(data))):

                if data[i][1] > function_thresholds[cv_number]:
                    temp.append(1)
                else:
                    temp.append(0)
            each_function_data.append(temp)
        each_function_data = np.average(each_function_data, 0)
        pred_each_function_label = []
        for i in tqdm(range(len(each_function_data))):
            if each_function_data[i] > 0.5:
                pred_each_function_label.append('Yes')
            else:
                pred_each_function_label.append('No')

        all_function_pred_label.append(pred_each_function_label)

    all_function_cols = ['antibacterial', 'anti-Gram-positive', 'anti-Gram-negative', 'antifungal', 'antiviral', \
         'anti-mammalian-cells', 'anti-HIV', 'antibiofilm', 'anticancer', 'anti-MRSA', 'antiparasitic', \
         'hemolytic', 'chemotactic', 'anti-TB', 'anurandefense', 'cytotoxic', \
         'endotoxin', 'insecticidal', 'antimalarial', 'anticandida', 'antiplasmodial', 'antiprotozoal']

    pred_contents_dict = {'name': fas_id, 'sequence': fas_seq, 'AMP': pred_AMP_label}
    for i in tqdm(range(len(all_function_cols))):
        pred_contents_dict[all_function_cols[i]] = all_function_pred_label[i]

    pred_contents_df = pd.DataFrame(pred_contents_dict)

    for function_name in all_function_names:
        for cv_number in tqdm(range(10)):
            os.remove(f'tmp_save/{function_name}/{temp_save_AMP_filename}_{cv_number}.csv')
    for cv_number in tqdm(range(10)):
        os.remove(f'tmp_save/{temp_save_AMP_filename}_{cv_number}.csv')

    return pred_contents_df
    # master.insert_one({'Test Report': res_val})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='proposed model')

    parser.add_argument('-output_file_name', default='prediction_output', type=str)

    parser.add_argument('-test_fasta_file', default='AMPpos.fasta', type=str)
    args = parser.parse_args()

    output_file_name = args.output_file_name
    test_file = args.test_fasta_file
    flag = 0


    if flag == 0:
        pred_df = predict(test_file)
        pred_df.to_csv(output_file_name + '.csv')
