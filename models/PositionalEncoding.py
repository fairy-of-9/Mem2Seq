import torch
import numpy as np
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super().__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    '''Sinusoid position encoding table'''

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            pos_anlge_vec = [position / np.power(10000, 2 * (hid_j // 2 / d_hid)) for hid_j in range(d_hid)]
            return pos_anlge_vec

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i + 1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

    # def vis_pos_vec(self, trg_dim, pos_vec):
    #     plt.pcolormesh(pos_vec, cmap='RdBu')
    #     plt.xlabel('Depth')
    #     plt.xlim((0, trg_dim))
    #     plt.ylabel('Postion')
    #     plt.colorbar()
    #     plt.show()