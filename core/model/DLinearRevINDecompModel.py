import torch
from sympy import false
from torch import nn


# --------------------------
# RevIN (per-instance, per-channel)
# --------------------------
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta  = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode="norm", stats=None):
        # x: (B, T, C)
        if mode == "norm":
            mean = x.mean(dim=1, keepdim=True)                    # (B,1,C)
            var  = x.var(dim=1, unbiased=False, keepdim=True)     # (B,1,C)
            std  = torch.sqrt(var + self.eps)
            x_n  = (x - mean) / std
            if self.affine:
                x_n = x_n * self.gamma + self.beta
            return x_n, (mean, std)
        elif mode == "denorm":
            assert stats is not None, "stats(mean,std) required for denorm"
            mean, std = stats
            if self.affine:
                x = (x - self.beta) / (self.gamma + 1e-8)
            return x * std + mean
        else:
            raise ValueError("mode must be 'norm' or 'denorm'")

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(torch.nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return moving_mean, residual

class DLinearRevIN(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.window_size = kwargs.get('window_size', 28)
        self.forecast_size = kwargs.get('forecast_size', 7)
        self.kernel_size = kwargs.get('kernel_size', 7)
        self.decomposition = series_decomp(self.kernel_size)
        self.individual = kwargs.get('individual', false)
        self.channels = kwargs.get('channels', 1)

        self.revin = RevIN(num_features=self.channels, eps=1e-5, affine=False)

        if self.individual:
            self.Linear_Seasonal = torch.nn.ModuleList()
            self.Linear_Trend = torch.nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Trend.append(torch.nn.Linear(self.window_size, self.forecast_size))
                self.Linear_Trend[i].weight = torch.nn.Parameter((1/self.window_size)*torch.ones([self.forecast_size, self.window_size]))
                self.Linear_Seasonal.append(torch.nn.Linear(self.window_size, self.forecast_size))
                self.Linear_Seasonal[i].weight = torch.nn.Parameter((1/self.window_size)*torch.ones([self.forecast_size, self.window_size]))
        else:
            self.Linear_Trend = torch.nn.Linear(self.window_size, self.forecast_size)
            self.Linear_Trend.weight = torch.nn.Parameter((1/self.window_size)*torch.ones([self.forecast_size, self.window_size]))
            self.Linear_Seasonal = torch.nn.Linear(self.window_size,  self.forecast_size)
            self.Linear_Seasonal.weight = torch.nn.Parameter((1/self.window_size)*torch.ones([self.forecast_size, self.window_size]))

    def forward(self, x):

        x, stats = self.revin(x, mode="norm")

        trend_init, seasonal_init = self.decomposition(x)
        trend_init, seasonal_init = trend_init.permute(0,2,1), seasonal_init.permute(0,2,1)

        if self.individual:
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.forecast_size], dtype=trend_init.dtype).to(trend_init.device)
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.forecast_size], dtype=seasonal_init.dtype).to(seasonal_init.device)
            for idx in range(self.channels):
                trend_output[:, idx, :] = self.Linear_Trend[idx](trend_init[:, idx, :])
                seasonal_output[:, idx, :] = self.Linear_Seasonal[idx](seasonal_init[:, idx, :])
        else:
            trend_output = self.Linear_Trend(trend_init)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
        x = seasonal_output + trend_output
        x = x.permute(0,2,1)

        x = self.revin(x, mode="denorm", stats=stats)


        return x