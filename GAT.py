import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class GATLayer(MessagePassing):

    def __init__(self, in_channels, out_channels, dropout=0.2, alpha=0.2):
        super().__init__(aggr="add")

        self.input_channels = in_channels
        self.output_channels = out_channels
        self.dropout = dropout
        self.alpha = alpha

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # Xavier Initialization of Weights
        self.W = nn.Linear(in_channels, out_channels)
        self.att = nn.Linear(2 * out_channels, 1)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att.weight)

    def forward(self, x, edge_index):
        Wh = self.W(x)
        out = self.propagate(edge_index, x=Wh)

        return out

    def message(self, edge_index_i, x_i, x_j, size_i):

        # x_cat = Whu||Whv
        x_cat = torch.cat([x_i, x_j], dim=-1)

        # attention = aT.(Whu||Whv)
        attention = self.att(x_cat)
        attention = self.leakyrelu(attention)

        # alpha = softmax(leakyReLU(aT.(Whu||Whv)))
        alpha = softmax(attention, edge_index_i, num_nodes=size_i)

        # Dropout for regularization
        alpha = F.dropout(alpha, self.dropout)

        # Final message passing
        message = alpha * x_j
        return message


class GATModel(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.gat1 = GATLayer(in_channels, hidden_channels, alpha=self.alpha)
        self.gat2 = GATLayer(hidden_channels,
                             hidden_channels,
                             alpha=self.alpha)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.leaky_relu(x, self.alpha)
        x = self.gat2(x, edge_index)
        x = F.leaky_relu(x, self.alpha)
        x = self.fc(x)
        return x
