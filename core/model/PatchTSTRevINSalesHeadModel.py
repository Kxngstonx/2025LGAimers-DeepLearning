import torch
import torch.nn as nn


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


# --------------------------
# 간단한 Patch 임베더
# --------------------------
class PatchEmbed(nn.Module):
    """
    입력: (B, T, D_in)
    1) 시간축으로 patchify: (B, N, patch_len, D_in)
    2) flatten 후 Linear -> (B, N, d_model)
    """
    def __init__(self, d_in, d_model, patch_len=4, stride=1, dropout=0.0):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(d_in * patch_len, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D_in)
        B, T, D = x.shape
        # unfold along time: (B, N, patch_len, D)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # (B, N, L, D)
        B, N, L, D = patches.shape
        patches = patches.reshape(B, N, L * D)     # (B, N, L*D)
        tokens  = self.proj(patches)               # (B, N, d_model)
        return self.drop(tokens)                   # (B, N, d_model)


# --------------------------
# Positional Embedding (learnable)
# --------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        # x: (B, N, d_model)
        N = x.size(1)
        return x + self.pe[:, :N, :]


# --------------------------
# PatchTST-RevIN (sales-only head)
# --------------------------
class PatchTSTRevINSalesHead(nn.Module):
    """
    입력  : x (B, window_size, feature_size)
    출력  : y (B, forecast_size, feature_size)  # ← DLinear와 동일
            단, 예측값은 target_idx(=sales 채널)에만 채워짐.
    """
    def __init__(self, **kwargs):

        """
            Args:
                window_size: int = 28,
                forecast_size: int = 7,
                feature_size: int = 25,
                target_idx: int = 0,
                d_model: int = 256,
                n_heads: int = 8,
                num_layers: int = 2,
                patch_len: int = 4,
                stride: int = 1,
                dropout: float = 0.1,
                exo_proj: bool = True,     # (B,T,C)->Linear(C->d_model) 사전 투영
                exo_scale: float = 1.0,    # 예측 후 스케일 확대/축소 여지
                max_tokens: int = 512      # 최대 토큰 수(포지셔널 임베드 용)
        """

        super().__init__()
        self.window_size = kwargs.get('window_size', 28)
        self.forecast_size = kwargs.get('forecast_size', 7)
        self.channels = kwargs.get('feature_size', 25)
        self.target_idx = kwargs.get('target_idx', 0)
        self.d_model = kwargs.get('d_model', 256)
        self.n_heads = kwargs.get('n_heads', 8)
        self.num_layers = kwargs.get('num_layers', 2)
        self.patch_len = kwargs.get('patch_len', 4)
        self.stride = kwargs.get('stride', 1)
        self.dropout = kwargs.get('dropout', 0.1)
        self.exo_proj = kwargs.get('exo_proj', True)
        self.exo_scale = kwargs.get('exo_scale', 1.0)
        self.max_tokens = kwargs.get('max_tokens', 512)

        # 1) RevIN
        self.revin = RevIN(num_features=self.channels, eps=1e-5, affine=False)

        # 2) 시간별 feature -> 임베딩 (선택)
        self.pre_proj = nn.Linear(self.channels, self.d_model) if self.exo_proj else None
        d_in = self.d_model if self.exo_proj else self.channels

        # 3) Patch 임베딩
        self.patch_embed = PatchEmbed(d_in=d_in, d_model=self.d_model, patch_len=self.patch_len, stride=self.stride, dropout=self.dropout)

        # 토큰 수(=N) 계산
        # N = floor((W - patch_len)/stride) + 1
        self.num_tokens = (self.window_size - self.patch_len) // self.stride + 1
        assert self.num_tokens > 0, "patch_len/stride가 window_size보다 크지 않게 설정하세요."

        # 4) positional embedding + transformer encoder
        self.pos_emb = PositionalEmbedding(max_len=min(self.max_tokens, self.num_tokens), d_model=self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_model*4,
            dropout=self.dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # 5) sales-only forecasting head
        # 토큰을 평균 풀링 → (B,d_model) → (B,Tp)
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.forecast_size)
        )

    def forward(self, x):
        """
        x: (B, W, C)
        return: (B, Tp, C)  # target_idx 위치만 예측값
        """
        B, W, C = x.shape
        assert W == self.window_size and C == self.channels, "입력 shape가 설정과 다릅니다."

        # --- RevIN 정규화 ---
        x_n, stats = self.revin(x, mode="norm")          # (B,W,C)

        # --- (선택) exogenous projection ---
        if self.pre_proj is not None:
            h = self.pre_proj(x_n)                       # (B,W,d_model)
        else:
            h = x_n                                      # (B,W,C)

        # --- patchify & pos-embed ---
        tok = self.patch_embed(h)                        # (B,N,d_model)
        tok = self.pos_emb(tok)                          # (B,N,d_model)

        # --- Transformer encoder ---
        z = self.encoder(tok)                            # (B,N,d_model)

        # --- Token pooling(평균) & Head ---
        z_pool = z.mean(dim=1)                           # (B,d_model)
        y_norm = self.head(z_pool)                       # (B,Tp)   (정규화 공간의 sales)

        # --- RevIN 되돌리기(판매 채널만) ---
        mean, std = stats                                # (B,1,C), (B,1,C)
        mu_s  = mean[:, :, self.target_idx]              # (B,1)
        std_s = std[:, :, self.target_idx]               # (B,1)

        y = y_norm * std_s + mu_s                        # (B,Tp)
        y = y * self.exo_scale                           # 온도/스케일 여지

        # --- DLinear 호환 shape로 변환 ---
        out = x.new_zeros(B, self.forecast_size, self.channels)            # (B,Tp,C)
        out[:, :, self.target_idx] = y                   # sales 채널만 채움
        return out