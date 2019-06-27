import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

hidden_size = 768


class SubjectModel(BertPreTrainedModel):
    def __init__(self, config):
        super(SubjectModel, self).__init__(config)

        # model_1 bert
        self.bert = BertModel(config)
        self.bert_l1 = nn.Linear(in_features=hidden_size, out_features=1)
        self.bert_l2 = nn.Linear(in_features=hidden_size, out_features=1)

        # # model_2 bert + cnn2
        # self.bert_ck2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=2)
        # self.bert_ck2_l1 = nn.Linear(in_features=hidden_size, out_features=1)
        # self.bert_ck2_l2 = nn.Linear(in_features=hidden_size, out_features=1)
        #
        # # model_2 bert + cnn3
        # self.bert_ck3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3)
        # self.bert_ck3_l1 = nn.Linear(in_features=hidden_size, out_features=1)
        # self.bert_ck3_l2 = nn.Linear(in_features=hidden_size, out_features=1)

        self.apply(self.init_bert_weights)

    def forward(self, device=None, x1_ids=None, x1_segments=None, x1_mask=None):
        # bert
        x1_encoder_layers, x1_pooled_out = self.bert(x1_ids, x1_segments, x1_mask, output_all_encoded_layers=False)
        ps1_bert = torch.sigmoid(self.bert_l1(x1_encoder_layers).squeeze(-1))
        ps2_bert = torch.sigmoid(self.bert_l2(x1_encoder_layers).squeeze(-1))

        # batch_size = x1_encoder_layers.size(0)
        #
        # # bert + cnn2
        # pad_tensor = torch.zeros((batch_size, hidden_size, 1), device=device)
        # x1_bert_ck2 = torch.tanh(self.bert_ck2(torch.cat([x1_encoder_layers.permute(0, 2, 1), pad_tensor], dim=-1))).permute(0, 2, 1)
        # ps1_bert_ck2 = torch.sigmoid(self.bert_ck2_l1(x1_bert_ck2).squeeze(-1))
        # ps2_bert_ck2 = torch.sigmoid(self.bert_ck2_l2(x1_bert_ck2).squeeze(-1))
        #
        # # bert + cnn3
        # pad_tensor = torch.zeros((batch_size, hidden_size, 2), device=device)
        # x1_bert_ck3 = torch.tanh(
        #     self.bert_ck3(torch.cat([x1_encoder_layers.permute(0, 2, 1), pad_tensor], dim=-1))).permute(0, 2, 1)
        # ps1_bert_ck3 = torch.sigmoid(self.bert_ck3_l1(x1_bert_ck3).squeeze(-1))
        # ps2_bert_ck3 = torch.sigmoid(self.bert_ck3_l2(x1_bert_ck3).squeeze(-1))
        #
        # ps1 = 0.4 * ps1_bert + 0.3 * ps1_bert_ck2 + 0.3 * ps1_bert_ck3
        # ps2 = 0.4 * ps2_bert + 0.3 * ps2_bert_ck2 + 0.3 * ps2_bert_ck3

        x1_mask = 1 - x1_mask.byte()

        return ps1_bert, ps2_bert, x1_mask


class ObjectModel(BertPreTrainedModel):
    def __init__(self, config):
        super(ObjectModel, self).__init__(config)
        self.bert = BertModel(config)

        self.linear = nn.Linear(in_features=hidden_size, out_features=1)

        self.apply(self.init_bert_weights)

    def forward(self, x_ids, x_seg, x_mask):
        output, _ = self.bert(x_ids, output_all_encoded_layers=False)  # [b,s,h]

        x_mask = 1-x_mask.byte()

        output.masked_fill_(x_mask.unsqueeze(-1), -1e-5)
        x = F.max_pool1d(output.permute(0,2,1), kernel_size=output.size(1)) # [b,h,1]
        x = x.squeeze(-1)

        o = torch.sigmoid(self.linear(x))

        return o


def focal_loss(y_p, y_t):
    gamma = 0.2
    loss = -y_t * (1-y_p)**gamma * torch.log(y_p) - (1-y_t) * y_p**gamma * torch.log(1-y_p)
    return loss.sum()
