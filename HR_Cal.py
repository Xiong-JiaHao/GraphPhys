import torch.nn as nn
from gcn_lib import get_act_layer
from torch.nn import Sequential as Seq


class HR_Cal(nn.Module):

    def __init__(self, opt):
        super(HR_Cal, self).__init__()

        # hr
        in_channle = opt.hr_cal_class_channle[0]
        channle = opt.hr_cal_class_channle
        dropout = opt.hr_cal_class_dropout_rate
        out_class = opt.hr_cal_out_class
        self.backbone_hr_class = nn.ModuleList([])
        for id in range(1, len(channle)):
            self.backbone_hr_class += [Seq(nn.Linear(in_channle, channle[id]),
                                           nn.Dropout(p=dropout[id - 1], inplace=False),
                                           get_act_layer(opt.hr_cal_class_act))]
            in_channle = channle[id]

        self.backbone_hr_class += Seq(nn.Linear(in_channle, out_class), nn.Softmax(dim=1))
        self.backbone_hr_class = Seq(*self.backbone_hr_class)

    def forward(self, x):  # x [B, 160]
        hr_class = x
        for i in range(len(self.backbone_hr_class)):
            hr_class = self.backbone_hr_class[i](hr_class)
        hr = hr_class.argmax(axis=1) + 40
        hr = hr.unsqueeze(1).float()
        return hr, hr_class


def hr_cal(opt):
    model = HR_Cal(opt)
    return model
