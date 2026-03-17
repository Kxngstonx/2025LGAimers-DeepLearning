import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class PatchMixerLayer(nn.Module):
    def __init__(self, dim, a, kernel_size=8):
        super().__init__()
        self.Resnet = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim, a, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(a)
        )

    def forward(self, x):
        x = x + self.Resnet(x)  # x: [batch * n_val, patch_num, d_model]
        x = self.Conv_1x1(x)  # x: [batch * n_val, a, d_model]
        return x


class PatchMixer(nn.Module):
    """
    🎯 **kwargs 기반 PatchMixer 모델
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = Backbone(**kwargs)

    def forward(self, x):
        x = self.model(x)
        return x


class Backbone(nn.Module):
    """
    🔧 **kwargs 기반 PatchMixer Backbone
    """

    def __init__(self, **kwargs):
        super().__init__()

        # 🎯 **kwargs에서 파라미터 추출 (기본값 설정)
        self.nvals = kwargs.get('enc_in', kwargs.get('feature_size', 31))
        self.lookback = kwargs.get('seq_len', kwargs.get('window_size', 28))
        self.forecasting = kwargs.get('pred_len', kwargs.get('forecast_size', 7))
        self.patch_size = kwargs.get('patch_len', 16)
        self.stride = kwargs.get('stride', 8)
        self.kernel_size = kwargs.get('mixer_kernel_size', 8)
        self.d_model = kwargs.get('d_model', 256)
        self.dropout = kwargs.get('dropout', 0.1)
        self.head_dropout = kwargs.get('head_dropout', 0.1)
        self.depth = kwargs.get('e_layers', kwargs.get('num_layers', 3))

        # RevIN 파라미터
        self.revin = kwargs.get('revin', True)
        revin_affine = kwargs.get('revin_affine', kwargs.get('affine', True))
        revin_subtract_last = kwargs.get('revin_subtract_last', kwargs.get('subtract_last', False))

        # 패치 관련 계산
        self.PatchMixer_blocks = nn.ModuleList([])
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num = int((self.lookback - self.patch_size) / self.stride + 1) + 1

        # a 파라미터 설정
        self.a = kwargs.get('a', self.patch_num)
        if self.a < 1 or self.a > self.patch_num:
            self.a = self.patch_num

        # PatchMixer 블록들 생성
        for _ in range(self.depth):
            self.PatchMixer_blocks.append(
                PatchMixerLayer(
                    dim=self.patch_num,
                    a=self.a,
                    kernel_size=self.kernel_size
                )
            )

        # 레이어들 정의
        self.W_P = nn.Linear(self.patch_size, self.d_model)

        self.head0 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * self.d_model, self.forecasting),
            nn.Dropout(self.head_dropout)
        )

        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.a * self.d_model, int(self.forecasting * 2)),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(int(self.forecasting * 2), self.forecasting),
            nn.Dropout(self.head_dropout)
        )

        self.dropout = nn.Dropout(self.dropout)

        # RevIN 설정
        if self.revin:
            self.revin_layer = RevIN(
                self.nvals,
                affine=revin_affine,
                subtract_last=revin_subtract_last
            )

    def forward(self, x):
        bs = x.shape[0]
        nvars = x.shape[-1]

        if self.revin:
            x = self.revin_layer(x, 'norm')

        x = x.permute(0, 2, 1)  # x: [batch, n_val, seq_len]

        x_lookback = self.padding_patch_layer(x)
        x = x_lookback.unfold(dimension=-1, size=self.patch_size,
                              step=self.stride)  # x: [batch, n_val, patch_num, patch_size]

        x = self.W_P(x)  # x: [batch, n_val, patch_num, d_model]
        x = torch.reshape(x,
                          (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # x: [batch * n_val, patch_num, d_model]
        x = self.dropout(x)
        u = self.head0(x)

        for PatchMixer_block in self.PatchMixer_blocks:
            x = PatchMixer_block(x)

        x = self.head1(x)
        x = u + x
        x = torch.reshape(x, (bs, nvars, -1))  # x: [batch, n_val, pred_len]
        x = x.permute(0, 2, 1)

        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x