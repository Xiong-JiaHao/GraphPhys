import torch
from torch import nn
from PhysNet.PhysNet_Layer import physnet
from Denoise_Graph import denoise_graph
from HR_Cal import hr_cal


class Model_PhysNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.opt = OptInit()
        self.model_physnet = physnet()
        self.model_wave_graph = denoise_graph(self.opt)
        self.model_hr_cal = hr_cal(self.opt)

        # Initialize weights
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)

        self.apply(_init)

    def forward(self, x):  # [B, 3, 160, 128, 128]
        wave = self.model_physnet(x)  # [B, 3, 160, 8, 8]
        ecg = self.model_wave_graph(wave)

        hr, hr_class = self.model_hr_cal(ecg)
        return ecg, hr, hr_class

    def __str__(self):
        str = self.opt.__str__() + "\n"
        str = str + self.model_physnet.__str__() + "\n"
        str = str + self.model_wave_graph.__str__() + "\n"
        str = str + self.model_hr_cal.__str__() + "\n"
        return str

    def save_model(self, log):
        torch.save(self.state_dict(), log + '/' + 'model_hr_best.pkl')
        torch.save(self.model_physnet.state_dict(), log + '/' + 'model_physnet.pkl')
        torch.save(self.model_wave_graph.state_dict(), log + '/' + 'model_wave_graph_best.pkl')
        torch.save(self.model_hr_cal.state_dict(), log + '/' + 'model_hr_cal_best.pkl')


class OptInit:
    def __init__(self):
        # Wave Graph
        self.wave_gnn_k = 18  # neighbor num (default:18)
        self.wave_gnn_conv = 'avg_relative_conv'  # graph conv layer {edge, sage, gin, mr, avg_relative_conv}
        self.wave_gnn_act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
        self.wave_gnn_norm = 'batch'  # batch or instance normalization {batch, instance}
        self.wave_gnn_bias = True  # bias of conv layer True or False
        self.wave_gnn_epsilon = 0.2  # stochastic epsilon for gcn
        self.wave_gnn_use_stochastic = False  # stochastic for gcn, True or False
        self.wave_gnn_drop_path = 0.0
        self.wave_gnn_blocks = [1, 1]  # number of basic blocks in the backbone
        self.wave_last_dim = 4096
        self.wave_channels = [768, 192, 96, 24, 1]  # number of channels of deep features

        # HRcal
        self.hr_cal_class_channle = [160, 280]
        self.hr_cal_class_act = 'gelu'
        self.hr_cal_class_dropout_rate = [0.25]
        self.hr_cal_out_class = 140

    def __str__(self):
        attrs = vars(self)
        return ', '.join("%s: %s" % item for item in attrs.items())
