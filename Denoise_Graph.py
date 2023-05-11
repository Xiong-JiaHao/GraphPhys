import torch
import torch.nn as nn
from timm.models.layers import DropPath
from model.gcn_lib.torch_edge import DenseDilatedKnnGraph
from model.gcn_lib.torch_vertex import GraphConv2d
from model.gcn_lib.torch_nn import get_act_layer
from torch.nn import Sequential as Seq
from data import ClipFramesLen


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = get_act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x)
        # x = self.drop_path(x) + shortcut
        return x


class SignalGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """

    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0):
        super(SignalGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None, **kwargs):
        tmp = x
        y = None
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(SignalGraphConv2d, self).forward(x, edge_index, y)
        return tmp + x


class DenoiseGraph(torch.nn.Module):
    def __init__(self, opt):
        super(DenoiseGraph, self).__init__()
        k = opt.wave_gnn_k
        act = opt.wave_gnn_act
        norm = opt.wave_gnn_norm
        bias = opt.wave_gnn_bias
        epsilon = opt.wave_gnn_epsilon
        stochastic = opt.wave_gnn_use_stochastic
        conv = opt.wave_gnn_conv
        drop_path = opt.wave_gnn_drop_path
        blocks = opt.wave_gnn_blocks
        channels = opt.wave_channels
        self.last_dim = opt.wave_last_dim

        self.n_blocks = sum(blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        max_dilation = max(1, 49 // max(num_knn))

        idx = 0
        self.graph_backbone = nn.ModuleList([])
        start_dim = channels[0]
        self.change_dim_layout = None
        if start_dim != self.last_dim:
            self.change_dim_layout = nn.Linear(self.last_dim, start_dim, bias=True)
        for i in range(len(blocks)):
            for j in range(blocks[i]):
                self.graph_backbone += [
                    Seq(SignalGraphConv2d(ClipFramesLen, ClipFramesLen, num_knn[idx], min(idx // 4 + 1, max_dilation),
                                          conv, act, norm, bias, stochastic, epsilon),
                        FFN(ClipFramesLen, ClipFramesLen, act=act, drop_path=dpr[idx]))]
                idx += 1

        dim = channels[0]
        self.graph_backbone = Seq(*self.graph_backbone)
        self.backbone = nn.ModuleList([])
        for id in range(1, len(channels)):
            self.backbone += [Seq(nn.Conv1d(dim, channels[id], 1, stride=1, padding=0), get_act_layer(act))]
            dim = channels[id]

        self.backbone = Seq(*self.backbone)

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        batch_size = x.shape[0]
        features_last = x.permute(0, 2, 1, 3, 4)
        features_last = features_last.reshape(batch_size, 160, self.last_dim)  # x [B, 160, self.last_dim]
        if self.change_dim_layout is not None:
            features_last = self.change_dim_layout(features_last)
        ecg = features_last.unsqueeze(3)  # [B, 160, start_dim, 1]

        for i in range(len(self.graph_backbone)):
            ecg = self.graph_backbone[i](ecg)

        ecg = ecg.squeeze(3).permute(0, 2, 1)  # x [B, start_dim, 160]
        for i in range(len(self.backbone)):
            ecg = self.backbone[i](ecg)
        # [B, 1, 160]

        ecg = ecg.squeeze(1)  # x [B, 160]
        return ecg


def denoise_graph(opt):
    model = DenoiseGraph(opt)
    return model
