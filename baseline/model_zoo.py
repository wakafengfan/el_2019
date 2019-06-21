import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import numpy as np
hidden_size = 768


class SubjectModel(BertPreTrainedModel):
    def __init__(self, config):
        super(SubjectModel, self).__init__(config)
        self.bert = BertModel(config)

        # cnn
        # self.bert_convs = nn.ModuleList([nn.Conv1d(in_channels=hidden_size,
        #                                            out_channels=hidden_size,
        #                                            kernel_size=k) for k in [2, 3, 4]
        #                                  ])
        # self.word2vec_convs = nn.ModuleList([nn.Conv1d(in_channels=200,
        #                                                out_channels=200,
        #                                                kernel_size=k) for k in [2, 3, 4]])

        # gru
        # self.gru = nn.GRU(input_size=200, hidden_size=200, bidirectional=True)

        self.linear1 = nn.Linear(in_features=hidden_size, out_features=1)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=1)
        self.apply(self.init_bert_weights)

    def forward(self, flag, device=None, tt=None, x1_ids=None, x1_segments=None, x1_mask=None, x2_ids=None, x2_segments=None, x2_mask=None):
        if flag == 'x1':

            # bert
            x1_encoder_layers, x1_pooled_out = self.bert(x1_ids, x1_segments, x1_mask, output_all_encoded_layers=False)

            # # bert + cnn
            # b,_,h = x1_encoder_layers.size()
            # x1_bert_conv = [F.relu(bert_conv(torch.cat([x1_encoder_layers.permute(0, 2, 1), torch.zeros((b,h,k-1),device=device)], dim=-1))).permute(0,2,1)
            #                 for k, bert_conv in enumerate(self.bert_convs, start=2)]  # [(b,s,h),...,]
            # x1_bert_conv = torch.cat(x1_bert_conv, dim=-1)  # [b,s,h*3]
            #
            # # word2vec + cnn
            # b,_,h = tt.size()
            # x1_wv_conv = [F.relu(wv_conv(torch.cat([tt.permute(0,2,1), torch.zeros((b,h,k-1),device=device)], dim=-1))).permute(0,2,1)
            #               for k,wv_conv in enumerate(self.word2vec_convs, start=2)]  # [(b,s,200)]
            # x1_wv_conv = torch.cat(x1_wv_conv, dim=-1)  #[b,s,200*3]
            #
            # # GRU
            # x1_wv_gru,_ = self.gru(tt)  # [b,s,200*2]
            #
            # x1 = torch.cat([x1_encoder_layers, x1_bert_conv, x1_wv_conv, x1_wv_gru], dim=-1)  # [b,s,h+h*3+200*3+200*2]


            ps1 = torch.sigmoid(self.linear1(x1_encoder_layers).squeeze(-1))
            ps2 = torch.sigmoid(self.linear2(x1_encoder_layers).squeeze(-1))

            return ps1, ps2, x1_encoder_layers, x1_pooled_out
        else:
            x2_encoder_layers, x2_pooled_out = self.bert(x2_ids, x2_segments, x2_mask, output_all_encoded_layers=False)
            return x2_encoder_layers, x2_pooled_out


class ObjectModel(nn.Module):
    def __init__(self):
        super(ObjectModel, self).__init__()
        W_bert = torch.empty(hidden_size + 1, hidden_size)
        nn.init.xavier_normal_(W_bert)
        self.W_bert = nn.Parameter(W_bert)

        # W_wv = torch.empty(200 + 1, 200)
        # nn.init.xavier_normal_(W_wv)
        # self.W_wv = nn.Parameter(W_wv)

        self.linear = nn.Linear(in_features=hidden_size * 3, out_features=1)

    def forward(self, x1, x1_h, x1_mask, y, x2, x2_h, x2_mask,tt,tt2):
        x1_mask = 1 - x1_mask.byte()
        x2_mask = 1 - x2_mask.byte()

        # x1_wv = torch.cat([tt, y.unsqueeze(2)], dim=-1)  # [b,s,201]
        # x1w_wv = torch.matmul(x1_wv, self.W_wv)
        # a_wv = torch.bmm(x1w_wv, tt2.permute(0,2,1))
        # a_wv.masked_fill_(x2_mask.unsqueeze(1), -1e-5)
        # a_wv = F.softmax(a_wv, dim=-1)
        # x12_wv = torch.bmm(a_wv, tt2)
        # x12_wv = F.max_pool1d(x12_wv.masked_fill(x1_mask.unsqueeze(2), -1e5).permute(0, 2, 1), kernel_size=x12_wv.size(1))
        # x12_wv = x12_wv.squeeze(2)  # [b,200]

        x1 = torch.cat([x1, y.unsqueeze(2)], dim=-1)  # [b,s,h] [b,s,1] -> [b,s,h+1]
        x1w = torch.matmul(x1, self.W_bert)
        a = torch.bmm(x1w, x2.permute(0, 2, 1))  # [b,s1,s2]
        a.masked_fill_(x2_mask.unsqueeze(1), -1e-5)  # [b,s2]->[b,1,s2]
        a = F.softmax(a, dim=-1)
        x12 = torch.bmm(a, x2)  # [b,s1,s2]*[b,s2,h]->[b,s1,h]
        x12 = F.max_pool1d(x12.masked_fill(x1_mask.unsqueeze(2), -1e5).permute(0, 2, 1), kernel_size=x12.size(1))
        x12 = x12.squeeze(2)

        h = torch.cat([x1_h, x2_h, x12], dim=1)  # [b,3*h+200]

        o = torch.sigmoid(self.linear(h))

        return o, x1_mask, x2_mask


