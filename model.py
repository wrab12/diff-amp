
import math
from math import sqrt
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class PositionalEmbedding(nn.Module):
    '''
    Modified from Annotated Transformer
    http://nlp.seas.harvard.edu/2018/04/03/attention.html
    '''
    def __init__(self, d_model, max_len=1024):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros((max_len, d_model), requires_grad=False).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class InputPositionEmbedding(nn.Module):
    def __init__(self, vocab_size=None, embed_dim=None, dropout=0.1,
                init_weight=None, seq_len=None):
        super(InputPositionEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.position_embed = PositionalEmbedding(embed_dim, max_len=seq_len)
        self.reproject = nn.Identity()
        if init_weight is not None:
            self.embed = nn.Embedding.from_pretrained(init_weight)
            self.reproject = nn.Linear(init_weight.size(1), embed_dim)

    def forward(self, inputs):
        # print(inputs.size())
        x = self.embed(inputs)
        # print(x.size())
        x = x + self.position_embed(inputs)
        # print(x)
        x = self.reproject(x)
        x = self.dropout(x)
        return x






class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att



class AggregateLayer(nn.Module):
    def __init__(self, d_model=None, dropout=0.1):
        super(AggregateLayer, self).__init__()        
        self.attn = nn.Sequential(collections.OrderedDict([
            ('layernorm', nn.LayerNorm(d_model)),
            ('fc', nn.Linear(d_model, 1, bias=False)),
            ('dropout', nn.Dropout(dropout)),
            ('softmax', nn.Softmax(dim=1))
        ]))

    def forward(self, context):
        '''
        Parameters
        ----------
        context: token embedding from encoder (Transformer/LSTM)
                (batch_size, seq_len, embed_dim)
        '''

        weight = self.attn(context)
        # (batch_size, seq_len, embed_dim).T * (batch_size, seq_len, 1) *  ->
        # (batch_size, embed_dim, 1)
        output = torch.bmm(context.transpose(1, 2), weight)
        output = output.squeeze(2)
        return output



class GlobalPredictor(nn.Module):
    def __init__(self, d_model=None, d_h=None, d_out=None, dropout=0.5):
        super(GlobalPredictor, self).__init__()
        self.predict_layer = nn.Sequential(collections.OrderedDict([
            ('batchnorm', nn.BatchNorm1d(d_model)),
            ('fc1', nn.Linear(d_model, d_h)),
            ('tanh', nn.Tanh()),
            ('dropout', nn.Dropout(dropout)),
            ('fc2', nn.Linear(d_h, d_out)),
            ('sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        x = self.predict_layer(x)
        return x


# class SequenceLSTM(nn.Module):
#     """Container module with an encoder, a recurrent module, and a decoder."""

#     def __init__(self, d_input=None, d_embed=20, d_model=128,
#                 vocab_size=None, seq_len=None,
#                 dropout=0.1, lstm_dropout=0,
#                 nlayers=1, bidirectional=False,
#                 proj_loc_config=None):
class SequenceLSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, d_input=None, d_embed=20, d_model=128,
                vocab_size=None, seq_len=None,
                dropout=0.1, lstm_dropout=0,
                nlayers=1, bidirectional=False,d_another_input=531,d_another_embed=128):
        super(SequenceLSTM, self).__init__()

        self.embed = InputPositionEmbedding(vocab_size=vocab_size,
                    seq_len=seq_len, embed_dim=d_embed)

        self.lstm = nn.LSTM(input_size=d_input,
                            hidden_size=d_model//2 if bidirectional else d_model,
                            num_layers=nlayers, dropout=lstm_dropout,
                            bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        self.proj_loc_layer = nn.Linear(d_another_input, d_another_embed)


    def forward(self, x, loc_feat=None):
        # print(x)
        x = self.embed(x)
        # print(x)
        if loc_feat is not None:
            loc_feat = self.proj_loc_layer(loc_feat)
            x = torch.cat([x, loc_feat], dim=2)
            # print(x)
        x = x.transpose(0, 1).contiguous()
        # print(x.size())
        x, _ = self.lstm(x)
        # print(x.size())
        x = x.transpose(0, 1).contiguous()
        # print(x.size())
        x = self.drop(x)
        return x


class SequenceCNNLSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, d_input=None, d_model=128,
                vocab_size=None, seq_len=None,
                dropout=0.1, lstm_dropout=0,
                nlayers=1, bidirectional=False,d_another_h=[64,32],d_output=1):
        super(SequenceCNNLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=d_input,
                            hidden_size=d_model//2 if bidirectional else d_model,
                            num_layers=nlayers, dropout=lstm_dropout,
                            bidirectional=bidirectional)
        self.conv1d = nn.Conv1d(in_channels=d_model,out_channels=d_another_h[0],kernel_size=3)

        self.relu=nn.ReLU()
        self.pooling=nn.MaxPool1d(kernel_size=seq_len-d_another_h[0]+1)

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_another_h[0], d_output)
        self.sigmoid=nn.Sigmoid()


    def forward(self, x):
        # print(x)
        x = x.transpose(0, 1).contiguous()
        x, _ = self.lstm(x)
        x = x.transpose(0, 1).contiguous()
        x = self.drop(x)
        x = x.permute(0,2,1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = x.view(-1, x.size(1)) 
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
    
    
    

    
    
class SequenceCNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, d_input=None,
                vocab_size=None, seq_len=None,
                dropout=0.1,d_another_h=[64,32],d_output=1):
        super(SequenceCNN, self).__init__()


        self.conv1d = nn.Conv1d(in_channels=d_input,out_channels=d_another_h[0],kernel_size=3)

        self.relu=nn.ReLU()
        self.pooling=nn.MaxPool1d(kernel_size=seq_len-d_another_h[0]+1)

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_another_h[0], d_output)
        self.sigmoid=nn.Sigmoid()


    def forward(self, x):
        # print(x)
        x = x.permute(0,2,1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = x.view(-1, x.size(1)) 
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class SequenceMultiCNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, d_input=None,
                vocab_size=None, seq_len=None,
                dropout=0.1,d_another_h=64,k_cnn=[2,3,4,5,6],d_output=1):
        super(SequenceMultiCNN, self).__init__()



        self.convs = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input, 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
#                              nn.BatchNorm1d(num_features=config.feature_size), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_another_h*len(k_cnn), d_output)
        self.fcblock=nn.Sequential(
                                               nn.BatchNorm1d(d_another_h*len(k_cnn)),
                                               nn.LeakyReLU(),
                                               nn.Linear(d_another_h*len(k_cnn),128),
                                               nn.BatchNorm1d(128),
                                               nn.LeakyReLU(),
                                               nn.Linear(128,64),
                                               nn.BatchNorm1d(64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64,1)
                                            )
        self.sigmoid=nn.Sigmoid()



    def forward(self, x):
        # print(x)
        x = x.permute(0,2,1)
        out = [conv(x) for conv in self.convs] 
        x = torch.cat(out, dim=1)

        x = x.view(-1, x.size(1)) 
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class SequenceMultiCNN_1(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, d_input=None,
                vocab_size=None, seq_len=None,
                dropout=0.1,d_another_h=64,k_cnn=[2,3,4,5,6],d_output=1):
        super(SequenceMultiCNN_1, self).__init__()



        self.convs = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input, 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
#                              nn.BatchNorm1d(num_features=config.feature_size), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_another_h*len(k_cnn), d_output)
        self.fcblock=nn.Sequential(
                                               nn.BatchNorm1d(d_another_h*len(k_cnn)),
                                               nn.LeakyReLU(),
                                               nn.Linear(d_another_h*len(k_cnn),128),
                                               nn.BatchNorm1d(128),
                                               nn.LeakyReLU(),
                                               nn.Linear(128,64),
                                               nn.BatchNorm1d(64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64,1)
                                            )
        self.sigmoid=nn.Sigmoid()



    def forward(self, AAI_feat,onehot_feat,BLOSUM62_feat,PAAC_feat):
        # print(x)
        x=torch.cat([AAI_feat,onehot_feat,BLOSUM62_feat,PAAC_feat],dim=2)
        # print(x.size())
        x = x.permute(0,2,1)
        out = [conv(x) for conv in self.convs] 
        x = torch.cat(out, dim=1)

        x = x.view(-1, x.size(1)) 
        x = self.fcblock(x)
        x = self.sigmoid(x)
        return x


class SequenceMultiCNNAGG(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, d_input=None,
                vocab_size=None, seq_len=None,
                dropout=0.1,d_another_h=64,k_cnn=[2,3,4,5,6],d_output=1):
        super(SequenceMultiCNNAGG, self).__init__()

        self.conv_1=nn.Conv1d(in_channels=d_input, out_channels=d_another_h, kernel_size=k_cnn[0])
        self.conv_2=nn.Conv1d(in_channels=d_input, 
                                out_channels=d_another_h, 
                                kernel_size=k_cnn[1])
        self.conv_3=nn.Conv1d(in_channels=d_input, 
                                out_channels=d_another_h, 
                                kernel_size=k_cnn[2])
        self.conv_4=nn.Conv1d(in_channels=d_input, 
                                out_channels=d_another_h, 
                                kernel_size=k_cnn[3])
        self.conv_5=nn.Conv1d(in_channels=d_input, 
                                out_channels=d_another_h, 
                                kernel_size=k_cnn[4])
        self.relu_1=nn.ReLU()
        self.relu_2=nn.ReLU()
        self.relu_3=nn.ReLU()
        self.relu_4=nn.ReLU()
        self.relu_5=nn.ReLU()
        self.agg_1=AggregateLayer(d_another_h)
        self.agg_2=AggregateLayer(d_another_h)
        self.agg_3=AggregateLayer(d_another_h)
        self.agg_4=AggregateLayer(d_another_h)
        self.agg_5=AggregateLayer(d_another_h)

        
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_another_h*len(k_cnn), d_output)
        self.sigmoid=nn.Sigmoid()



    def forward(self, x):
        # print(x)
        x = x.permute(0,2,1)
        out_1= self.conv_1(x)
        out_2= self.conv_2(x)
        out_3= self.conv_3(x)
        out_4= self.conv_4(x)
        out_5= self.conv_5(x)
        out_1=out_1.permute(0,2,1)
        out_2=out_2.permute(0,2,1)
        out_3=out_3.permute(0,2,1)
        out_4=out_4.permute(0,2,1)
        out_5=out_5.permute(0,2,1)
        # print(out_1.size())
        out_1=self.agg_1(out_1)
        # print(out_1.size())
        out_2=self.agg_1(out_2)
        out_3=self.agg_1(out_3)
        out_4=self.agg_1(out_4)
        out_5=self.agg_1(out_5)
        out = torch.cat([out_1,out_2,out_3,out_4,out_5], dim=1)
        # print(out.size())
        # x = x.view(-1, x.size(1)) 
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
    
    

class SequenceMultiTypeMultiCNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, d_input=[531,21,23,3],
                vocab_size=None, seq_len=None,
                dropout=0.1,d_another_h=64,k_cnn=[2,3,4,5,6],d_output=1):
        super(SequenceMultiTypeMultiCNN, self).__init__()



        self.convs_1 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[0], 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
                              nn.BatchNorm1d(num_features=d_another_h), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])
        self.convs_2 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[1], 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
                              nn.BatchNorm1d(num_features=d_another_h), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])
        self.convs_3 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[2], 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
                              nn.BatchNorm1d(num_features=d_another_h), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])
        # self.convs_4 = nn.ModuleList([
        #         nn.Sequential(nn.Conv1d(in_channels=d_input[3], 
        #                                 out_channels=d_another_h, 
        #                                 kernel_size=h),
        #                       nn.BatchNorm1d(num_features=d_another_h), 
        #                       nn.ReLU(),
        #                       nn.MaxPool1d(kernel_size=seq_len-h+1))
        #              for h in k_cnn
        #             ])
        self.maxpool_1=nn.MaxPool1d(kernel_size=5)
        self.maxpool_2=nn.MaxPool1d(kernel_size=5)
        self.maxpool_3=nn.MaxPool1d(kernel_size=5)
        # self.maxpool_4=nn.MaxPool1d(kernel_size=5)
        self.drop = nn.Dropout(dropout)
        self.batchnorm1d=nn.BatchNorm1d(num_features=d_another_h*3)
        self.fc = nn.Linear(d_another_h*3, d_output)
        self.sigmoid=nn.Sigmoid()



    def forward(self, AAI_feat,onehot_feat,BLOSUM62_feat):
        # print(x)
        AAI_feat = AAI_feat.permute(0,2,1)
        out_1 = [conv(AAI_feat) for conv in self.convs_1] 
        onehot_feat = onehot_feat.permute(0,2,1)
        out_2 = [conv(onehot_feat) for conv in self.convs_2] 
        BLOSUM62_feat = BLOSUM62_feat.permute(0,2,1)
        out_3 = [conv(BLOSUM62_feat) for conv in self.convs_3] 

        out_1 = torch.cat(out_1, dim=2)
        # print(out_1.size())
        out_1=self.maxpool_1(out_1)
        # print(out_1.size())
        out_2 = torch.cat(out_2, dim=2)
        out_2=self.maxpool_2(out_2)
        out_3 = torch.cat(out_3, dim=2)
        out_3=self.maxpool_3(out_3)

        x=torch.cat([out_1,out_2,out_3],dim=1)
        x = x.view(-1, x.size(1)) 
        x = self.batchnorm1d(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class SequenceMultiTypeMultiCNN_1(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, d_input=[531,21,23,3],
                vocab_size=None, seq_len=None,
                dropout=0.1,d_another_h=64,k_cnn=[2,3,4,5,6],d_output=1):
        super(SequenceMultiTypeMultiCNN_1, self).__init__()
        
        self.batchnorm_4=nn.BatchNorm1d(num_features=d_input[3])
        self.convs_1 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[0], 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
                              nn.BatchNorm1d(num_features=d_another_h), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])
        self.convs_2 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[1], 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
                              nn.BatchNorm1d(num_features=d_another_h), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])
        self.convs_3 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[2], 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
                              nn.BatchNorm1d(num_features=d_another_h), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])
        self.convs_4 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[3], 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
                              nn.BatchNorm1d(num_features=d_another_h), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])
        self.maxpool_1=nn.MaxPool1d(kernel_size=len(k_cnn))
        self.maxpool_2=nn.MaxPool1d(kernel_size=len(k_cnn))
        self.maxpool_3=nn.MaxPool1d(kernel_size=len(k_cnn))
        self.maxpool_4=nn.MaxPool1d(kernel_size=len(k_cnn))
        # self.maxpool_1=nn.AvgPool1d(kernel_size=5)
        # self.maxpool_2=nn.AvgPool1d(kernel_size=5)
        # self.maxpool_3=nn.AvgPool1d(kernel_size=5)
        # self.maxpool_4=nn.AvgPool1d(kernel_size=5)
        self.drop = nn.Dropout(dropout)
        
        self.fc_1 = nn.Linear(d_another_h*len(k_cnn), d_output)
        self.fc_2 = nn.Linear(d_another_h*len(k_cnn), d_output)
        self.fc_3 = nn.Linear(d_another_h*len(k_cnn), d_output)
        self.fc_4 = nn.Linear(d_another_h*len(k_cnn), d_output)
        self.fc = nn.Linear(4*d_another_h, d_output)
        self.sigmoid=nn.Sigmoid()



    def forward(self, AAI_feat,onehot_feat,BLOSUM62_feat,PAAC_feat):
        # print(x)
        AAI_feat = AAI_feat.permute(0,2,1)
        out_1 = [conv(AAI_feat) for conv in self.convs_1] 
        onehot_feat = onehot_feat.permute(0,2,1)
        out_2 = [conv(onehot_feat) for conv in self.convs_2] 
        BLOSUM62_feat = BLOSUM62_feat.permute(0,2,1)
        out_3 = [conv(BLOSUM62_feat) for conv in self.convs_3] 
        
        PAAC_feat = PAAC_feat.permute(0,2,1)
        PAAC_feat = self.batchnorm_4(PAAC_feat)
        out_4 = [conv(PAAC_feat) for conv in self.convs_4] 
        out_1 = torch.cat(out_1, dim=2)
        # print(out_1.size())
        out_1=self.maxpool_1(out_1)
        # print(out_1.size())
        out_2 = torch.cat(out_2, dim=2)
        out_2=self.maxpool_2(out_2)
        out_3 = torch.cat(out_3, dim=2)
        out_3=self.maxpool_3(out_3)
        out_4 = torch.cat(out_4, dim=2)
        out_4=self.maxpool_4(out_4)
        # print(out_4.size())
        x=torch.cat([out_1,out_2,out_3,out_4],dim=1)
        x = x.view(-1, x.size(1)) 
        # out=torch.cat([out_1,out_2,out_3,out_4], dim=1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x




class SequenceMultiTypeMultiCNN_2(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, d_input=[531,21,23,3],
                vocab_size=None, seq_len=None,
                dropout=0.1,d_another_h=64,k_cnn=[2,3,4,5,6],d_output=1):
        super(SequenceMultiTypeMultiCNN_2, self).__init__()
        
        self.batchnorm_4=nn.BatchNorm1d(num_features=d_input[3])
        self.convs_1 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[0], 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
                              nn.BatchNorm1d(num_features=d_another_h), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])
        self.convs_2 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[1], 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
                              nn.BatchNorm1d(num_features=d_another_h), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])
        self.convs_3 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[2], 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
                              nn.BatchNorm1d(num_features=d_another_h), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])
        self.convs_4 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[3], 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
                              nn.BatchNorm1d(num_features=d_another_h), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])
        self.maxpool_1=nn.MaxPool1d(kernel_size=len(k_cnn))
        self.maxpool_2=nn.MaxPool1d(kernel_size=len(k_cnn))
        self.maxpool_3=nn.MaxPool1d(kernel_size=len(k_cnn))
        self.maxpool_4=nn.MaxPool1d(kernel_size=len(k_cnn))
        # self.maxpool_1=nn.AvgPool1d(kernel_size=5)
        # self.maxpool_2=nn.AvgPool1d(kernel_size=5)
        # self.maxpool_3=nn.AvgPool1d(kernel_size=5)
        # self.maxpool_4=nn.AvgPool1d(kernel_size=5)
        self.drop = nn.Dropout(dropout)
        
        self.fc_1 = nn.Linear(d_another_h*len(k_cnn), d_output)
        self.fc_2 = nn.Linear(d_another_h*len(k_cnn), d_output)
        self.fc_3 = nn.Linear(d_another_h*len(k_cnn), d_output)
        self.fc_4 = nn.Linear(d_another_h*len(k_cnn), d_output)
        self.fc = nn.Linear(4*d_another_h, d_output)
        self.sigmoid=nn.Sigmoid()



    def forward(self, AAI_feat,onehot_feat,BLOSUM62_feat,PAAC_feat):
        # print(x)
        AAI_feat = AAI_feat.permute(0,2,1)
        out_1 = [conv(AAI_feat) for conv in self.convs_1] 
        onehot_feat = onehot_feat.permute(0,2,1)
        out_2 = [conv(onehot_feat) for conv in self.convs_2] 
        BLOSUM62_feat = BLOSUM62_feat.permute(0,2,1)
        out_3 = [conv(BLOSUM62_feat) for conv in self.convs_3] 
        
        PAAC_feat = PAAC_feat.permute(0,2,1)
        PAAC_feat = self.batchnorm_4(PAAC_feat)
        out_4 = [conv(PAAC_feat) for conv in self.convs_4] 
        out_1 = torch.cat(out_1, dim=2)
        # print(out_1.size())
        out_1=self.maxpool_1(out_1)
        # print(out_1.size())
        out_2 = torch.cat(out_2, dim=2)
        out_2=self.maxpool_2(out_2)
        out_3 = torch.cat(out_3, dim=2)
        out_3=self.maxpool_3(out_3)
        out_4 = torch.cat(out_4, dim=2)
        out_4=self.maxpool_4(out_4)
        # print(out_4.size())
        x=torch.cat([out_1,out_2,out_3,out_4],dim=1)
        x = x.view(-1, x.size(1)) 
        # out=torch.cat([out_1,out_2,out_3,out_4], dim=1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x,out_1,out_2,out_3,out_4





class SequenceMultiCNNLSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, d_input=None, d_model=128,
                vocab_size=None, seq_len=None,
                dropout=0.1, lstm_dropout=0,
                nlayers=1, bidirectional=False,d_another_h=64,k_cnn=[2,3,4,5,6],d_output=1):
        super(SequenceMultiCNNLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=d_input,
                            hidden_size=d_model//2 if bidirectional else d_model,
                            num_layers=nlayers, dropout=lstm_dropout,
                            bidirectional=bidirectional)

        self.convs = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_model, 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
#                              nn.BatchNorm1d(num_features=config.feature_size), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_another_h*len(k_cnn), d_output)
        self.sigmoid=nn.Sigmoid()



    def forward(self, x):
        # print(x)
        x = x.transpose(0, 1).contiguous()
        x, _ = self.lstm(x)
        x = x.transpose(0, 1).contiguous()
        x = self.drop(x)
        x = x.permute(0,2,1)
        out = [conv(x) for conv in self.convs] 
        x = torch.cat(out, dim=1)

        x = x.view(-1, x.size(1)) 
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    



class TranformerModel(nn.Module):
    def __init__(self, vocab_size=24,hidden_dim=25,d_embed=512,max_length=500):
        super(TranformerModel,self).__init__()

        # self.embedding = nn.Embedding(vocab_size, d_embed, padding_idx=0)
        self.embed = InputPositionEmbedding(vocab_size=vocab_size,
                    seq_len=max_length, embed_dim=d_embed)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        
        self.gru = nn.GRU(d_embed, hidden_dim, num_layers=2, 
                               bidirectional=True, dropout=0.2)
        
        
        self.block1=nn.Sequential(nn.Linear(d_embed*max_length,1024),
                                            nn.BatchNorm1d(1024),
                                            nn.LeakyReLU(),
                                            nn.Linear(1024,256),
                                 )

        self.block2=nn.Sequential(
                                               nn.BatchNorm1d(256),
                                               nn.LeakyReLU(),
                                               nn.Linear(256,128),
                                               nn.BatchNorm1d(128),
                                               nn.LeakyReLU(),
                                               nn.Linear(128,64),
                                               nn.BatchNorm1d(64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64,1)
                                            )
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        x=self.embed(x)
        output=self.transformer_encoder(x).permute(1, 0, 2)
        # print(output.size())
        # output,hn=self.gru(output)
        # print(output.size())
        # print(hn.size())
        output=output.permute(1,0,2)
        # hn=hn.permute(1,0,2)
        
        output=output.reshape(output.shape[0],-1)
        # hn=hn.reshape(output.shape[0],-1)
        # print(output.size())
        # print(hn.size())
        # output=torch.cat([output,hn],1)
        # print(output.size())
        output=self.block1(output)
        output=self.block2(output)
        output=self.sigmoid(output)
        # print(output.size())
        return output


    def __init__(self, config):
        super().__init__(config)
        self.bert = ProteinBertModel(config)
        self.embedding = nn.Embedding(21, 512, padding_idx=0)
        # self.embed = InputPositionEmbedding(vocab_size=21,
        #             seq_len=300, embed_dim=512)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        
        self.gru = nn.GRU(512, 25, num_layers=2, 
                                bidirectional=True, dropout=0.2)
        
        
        self.block1=nn.Sequential(nn.Linear(50*300+25*4,1024),
                                            nn.BatchNorm1d(1024),
                                            nn.LeakyReLU(),
                                            nn.Linear(1024,256),
                                  )

        self.block2=nn.Sequential(
                                                nn.BatchNorm1d(256+768),
                                                nn.LeakyReLU(),
                                                nn.Linear(256,128),
                                                nn.BatchNorm1d(128),
                                                nn.LeakyReLU(),
                                                nn.Linear(128,64),
                                                nn.BatchNorm1d(64),
                                                nn.LeakyReLU(),
                                                nn.Linear(64,1)
                                            )
        self.sigmoid=nn.Sigmoid()
    def forward(self, x, input_ids=None, input_mask=None,):
        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        average = torch.mean(sequence_output, dim=1)
        x=self.embedding(x)
        output=self.transformer_encoder(x).permute(1, 0, 2)
        print(output.size())
        output,hn=self.gru(output)
        print(output.size())
        print(hn.size())
        output=output.permute(1,0,2)
        hn=hn.permute(1,0,2)
        
        output=output.reshape(output.shape[0],-1)
        hn=hn.reshape(output.shape[0],-1)
        # print(output.size())
        # print(hn.size())
        output=torch.cat([output,hn],1)
        # print(output.size())
        output=self.block1(output)
        output=torch.cat([output,average],1)
        output=self.block2(output)
        output=self.sigmoid(output)
        # print(output.size())
        return output



class TranformerModelNOGRU(nn.Module):
    def __init__(self, vocab_size=24,hidden_dim=25,d_embed=512,max_length=500):
        super(TranformerModelNOGRU,self).__init__()

        # self.embedding = nn.Embedding(vocab_size, d_embed, padding_idx=0)
        self.embed = InputPositionEmbedding(vocab_size=vocab_size,
                    seq_len=max_length, embed_dim=d_embed)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        
        self.gru = nn.GRU(d_embed, hidden_dim, num_layers=2, 
                               bidirectional=True, dropout=0.2)
        
        
        self.block1=nn.Sequential(nn.Linear(d_embed*max_length,1024),
                                            nn.BatchNorm1d(1024),
                                            nn.LeakyReLU(),
                                            nn.Linear(1024,256),
                                 )

        self.block2=nn.Sequential(
                                               nn.BatchNorm1d(256),
                                               nn.LeakyReLU(),
                                               nn.Linear(256,128),
                                               nn.BatchNorm1d(128),
                                               nn.LeakyReLU(),
                                               nn.Linear(128,64),
                                               nn.BatchNorm1d(64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64,1)
                                            )
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        x=self.embed(x)
        output=self.transformer_encoder(x).permute(1, 0, 2)
        # print(output.size())
        # output,hn=self.gru(output)
        # print(output.size())
        # print(hn.size())
        output=output.permute(1,0,2)
        # hn=hn.permute(1,0,2)
        
        output=output.reshape(output.shape[0],-1)
        # hn=hn.reshape(output.shape[0],-1)
        # print(output.size())
        # print(hn.size())
        # output=torch.cat([output,hn],1)
        # print(output.size())
        output=self.block1(output)
        output=self.block2(output)
        output=self.sigmoid(output)
        # print(output.size())
        return output


# class LSTMPredictor(nn.Module):
#     def __init__(self, d_embed=20, d_model=128, d_h=128, d_out=1,
#                 vocab_size=None, seq_len=None,
#                 dropout=0.1, lstm_dropout=0, nlayers=1, bidirectional=False,
#                 use_loc_feat=True, use_glob_feat=True,
#                 proj_loc_config=None, proj_glob_config=None):
class LSTMPredictor(nn.Module):
    def __init__(self, d_embed=20, d_model=128, d_h=128, d_out=1,
                vocab_size=None, seq_len=None,use_loc_feat=True,
                dropout=0.1, lstm_dropout=0, nlayers=1, bidirectional=False,
                d_another_input=531,d_another_embed=128):
        super(LSTMPredictor, self).__init__()
        # self.seq_lstm = SequenceLSTM(
        #     d_input=d_embed + (proj_loc_config['d_out'] if use_loc_feat else 0),
        #     d_embed=d_embed, d_model=d_model,
        #     vocab_size=vocab_size, seq_len=seq_len,
        #     dropout=dropout, lstm_dropout=lstm_dropout,
        #     nlayers=nlayers, bidirectional=bidirectional,
        #     proj_loc_config=proj_loc_config)
        self.seq_lstm = SequenceLSTM(
            d_input=d_embed + (d_another_embed if use_loc_feat else 0),
            d_embed=d_embed, d_model=d_model,
            vocab_size=vocab_size, seq_len=seq_len,
            dropout=dropout, lstm_dropout=lstm_dropout,
            nlayers=nlayers, bidirectional=bidirectional,
            d_another_input=d_another_input,d_another_embed=d_another_embed)
        
        # self.proj_glob_layer = proj_glob_config['layer'](
        #     proj_glob_config['d_in'], proj_glob_config['d_out']
        # )
        # self.aggragator = AggregateLayer(
        #     d_model = d_model + (proj_glob_config['d_out'] if use_glob_feat else 0))
        self.aggragator = AggregateLayer(
            d_model = d_model)
        # self.predictor = GlobalPredictor(
        #     d_model = d_model + (proj_glob_config['d_out'] if use_glob_feat else 0),
        #     d_h=d_h, d_out=d_out)
        self.predictor = GlobalPredictor(
            d_model = d_model,
            d_h=d_h, d_out=d_out)
        
    def forward(self, x, glob_feat=None, loc_feat=None):
        x = self.seq_lstm(x, loc_feat=loc_feat)
        # print(x.shape)
        # if glob_feat is not None:
        #     glob_feat = self.proj_glob_layer(glob_feat)
        #     print(glob_feat.shape)
        #     x = torch.cat([x, glob_feat], dim=2)
        #     print(x.shape)
        x = self.aggragator(x)
        # print(x)
        output = self.predictor(x)
        # print(output.shape)
        return output


if __name__ == "__main__":
    # model = LSTMPredictor(
    #     d_model=128, d_h=128, nlayers=1,
    #     vocab_size=21, seq_len=500,bidirectional=True,
    #     d_another_input=531,d_another_embed=128)
    
    model = SequenceMultiCNN_1(d_input=578,
                vocab_size=21, seq_len=500,
                dropout=0.1,d_another_h=64,d_output=1)

    # model = SequenceMultiCNNLSTM(d_input=531, d_model=128,
    #             vocab_size=21, seq_len=500,
    #             dropout=0.1, d_another_h=64,k_cnn=[2,3,4,5,6],d_output=1)
    model = SequenceMultiTypeMultiCNN_1(d_input=[531,21,23,3], 
                vocab_size=21, seq_len=500,
                dropout=0.1, d_another_h=64,k_cnn=[2,3,4,5,6],d_output=1)
    x = torch.randint(0, 21, (128, 300))
    ids = torch.randint(0, 21, (128, 300))
    masks=torch.ones(128, 300)
    # glob_feat = torch.rand((128, 500, 768))
    AAI_feat = torch.rand((128, 500, 531))
    onehot_feat = torch.rand((128, 500, 21))
    blosum62_feat = torch.rand((128, 500, 23))
    PAAC_feat = torch.rand((128, 500, 3))
    # y = model(x, glob_feat=glob_feat, loc_feat=loc_feat)    
    # print(y.size())

    
    # model=InputPositionEmbedding(vocab_size=21,
    #             seq_len=500, embed_dim=128)
    # # model = LSTMPredictor(
    # #     d_model=128, d_h=128, nlayers=1,
    # #     vocab_size=21, seq_len=500,        
    # #     proj_glob_config = {'layer':nn.Linear, 'd_in':768, 'd_out':128},
    # #     proj_loc_config = {'layer':nn.Linear, 'd_in':500, 'd_out':128},
    # #     )
    # x = torch.randint(0, 21, (128, 500))
    # glob_feat = torch.rand((128, 500, 768))
    # loc_feat = torch.rand((128, 500, 500))
    # y = model(x,loc_feat=loc_feat)   
    y = model(AAI_feat,onehot_feat,blosum62_feat,PAAC_feat)  
    # y = model(AAI_feat)
    # y=model(x,ids,masks)
