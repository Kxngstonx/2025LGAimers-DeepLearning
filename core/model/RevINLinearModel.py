import torch
import torch.nn as nn

class RLinear(nn.Module):
    def __init__(self, configs):
        super().__init__()

        configs['rev'] = True
        configs['individual'] = False
        self.pred_len = configs['pred_len']

        self.Linear = nn.ModuleList([
            nn.Linear(configs['seq_len'], configs['pred_len']) for _ in range(configs['enc_in'])
        ]) if configs['individual'] else nn.Linear(configs['seq_len'], configs['pred_len'])

        self.dropout = nn.Dropout(0.1)
        self.rev = RevIN(configs['enc_in']) if configs['rev'] else None
        self.individual = configs['individual']

    def forward(self, x):
        # x: [B, L, D]
        x = self.rev(x, 'norm') if self.rev else x
        x = self.dropout(x)
        if self.individual:
            pred = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm') if self.rev else pred

        return pred


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)

        elif mode == 'denorm':
            x = self._denormalize(x)

        else:
            raise NotImplementedError

        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean

        return x