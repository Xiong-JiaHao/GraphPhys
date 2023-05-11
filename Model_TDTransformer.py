import torch
from torch import nn
from TDTransformer.TDTransformer_Layer import td_transformer
from Denoise_Graph import denoise_graph
from HR_Cal import hr_cal


class Model_TDTransformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.opt = OptInit()
        # self.model_feature_extraction = face_feature_extraction(self.opt)
        # self.model_graph = face_graph(self.opt)
        self.model_transformer = td_transformer(self.opt)
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

    def forward(self, x):
        # input_feature_extraction = x.permute(2, 0, 1, 3, 4)  # [160, B, 3, 128, 128]
        # output_grape = []
        # for x in input_feature_extraction:
        #     x = self.model_feature_extraction(x)  # [B, 96, 16, 16]
        #     # output_grape.append(self.model_graph(x))  # [B, 96, 16, 16]
        #     output_grape.append(x)  # [B, 96, 16, 16]

        wave = self.model_transformer(x)  # [B, 48, 160, 4, 4]
        ecg = self.model_wave_graph(wave)

        hr, hr_class = self.model_hr_cal(ecg)
        return ecg, hr, hr_class

    def __str__(self):
        str = self.opt.__str__() + "\n"
        # str = str + self.model_feature_extraction.__str__() + "\n"
        # str = str + self.model_graph.__str__() + "\n"
        str = str + self.model_transformer.__str__() + "\n"
        str = str + self.model_wave_graph.__str__() + "\n"
        str = str + self.model_hr_cal.__str__() + "\n"
        return str

    def save_model(self, log):
        torch.save(self.state_dict(), log + '/' + 'model_hr_best.pkl')
        torch.save(self.model_transformer.state_dict(), log + '/' + 'model_hr_transformer_best.pkl')
        torch.save(self.model_wave_graph.state_dict(), log + '/' + 'model_wave_graph_best.pkl')
        torch.save(self.model_hr_cal.state_dict(), log + '/' + 'model_hr_cal_best.pkl')


class OptInit:
    def __init__(self):
        # TDTransformer
        self.tdt_image_size = (160, 128, 128)
        self.tdt_patches = (4, 4, 4)
        self.tdt_dim = 96
        self.tdt_ff_dim = 144
        self.tdt_num_heads = 4
        self.tdt_num_layers = 12
        self.tdt_dropout_rate = 0.1
        self.tdt_theta = 0.7
        self.tdt_gra_sharp = 0.2

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
        self.wave_last_dim = 768
        self.wave_channels = [768, 192, 96, 24, 1]  # number of channels of deep features

        # HRcal
        self.hr_cal_class_channle = [160, 280]
        self.hr_cal_class_act = 'gelu'
        self.hr_cal_class_dropout_rate = [0.25]
        self.hr_cal_out_class = 140

    def __str__(self):
        attrs = vars(self)
        return ', '.join("%s: %s" % item for item in attrs.items())

