import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

hidden_size = 768


class SubjectModel(BertPreTrainedModel):
    def __init__(self, config):
        super(SubjectModel, self).__init__(config)
        self.bert = BertModel(config)
        self.linear1 = nn.Linear(in_features=hidden_size, out_features=1)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=1)
        self.apply(self.init_bert_weights)

    def forward(self, flag, x1_ids=None, x1_segments=None, x1_mask=None, x2_ids=None, x2_segments=None, x2_mask=None):
        if flag == 'x1':
            x1_encoder_layers, x1_pooled_out = self.bert(x1_ids, x1_segments, x1_mask, output_all_encoded_layers=False)

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

        W_wv = torch.empty(200 + 1, 200)
        nn.init.xavier_normal_(W_wv)
        self.W_wv = nn.Parameter(W_wv)

        self.linear_wv = nn.Linear(in_features=200, out_features=hidden_size)

        self.linear = nn.Linear(in_features=hidden_size * 4, out_features=1)

    def forward(self, x1, x1_h, x1_mask, y, x2, x2_h, x2_mask,tt,tt2):
        x1_mask = 1 - x1_mask.byte()
        x2_mask = 1 - x2_mask.byte()

        x1_wv = torch.cat([tt, y.unsqueeze(2)], dim=-1)  # [b,s,201]
        x1w_wv = torch.matmul(x1_wv, self.W_wv)
        a_wv = torch.bmm(x1w_wv, tt2.permute(0,2,1))
        a_wv.masked_fill_(x2_mask.unsqueeze(1), -1e-5)
        a_wv = F.softmax(a_wv, dim=-1)
        x12_wv = torch.bmm(a_wv, tt2)
        x12_wv = F.max_pool1d(x12_wv.masked_fill(x1_mask.unsqueeze(2), -1e5).permute(0, 2, 1), kernel_size=x12_wv.size(1))
        x12_wv = x12_wv.squeeze(2)  # [b,200]
        x12_wv = self.linear_wv(x12_wv)

        x1 = torch.cat([x1, y.unsqueeze(2)], dim=-1)  # [b,s,h] [b,s,1] -> [b,s,h+1]
        x1w = torch.matmul(x1, self.W_bert)
        a = torch.bmm(x1w, x2.permute(0, 2, 1))  # [b,s1,s2]
        a.masked_fill_(x2_mask.unsqueeze(1), -1e-5)  # [b,s2]->[b,1,s2]
        a = F.softmax(a, dim=-1)
        x12 = torch.bmm(a, x2)  # [b,s1,s2]*[b,s2,h]->[b,s1,h]
        x12 = F.max_pool1d(x12.masked_fill(x1_mask.unsqueeze(2), -1e5).permute(0, 2, 1), kernel_size=x12.size(1))
        x12 = x12.squeeze(2)

        h = torch.cat([x1_h, x2_h, x12, x12_wv], dim=1)  # [b,3*h]

        o = torch.sigmoid(self.linear(h))

        return o, x1_mask, x2_mask


