# adapted from https://github.com/jik876/hifi-gan

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "speech-resynthesis"))
from models import CodeGenerator
sys.path.pop(0)


class CustomPermute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.new_dims = args

    def forward(self, x):
        return x.permute(self.new_dims)
    

class MelCodeGenerator(CodeGenerator):
    def __init__(self, h):
        super().__init__(h)
        self.dict = nn.Embedding(h.num_embeddings, h.embedding_dim)  # (200, 128)
        self.multispkr = h.get('multispkr', None)
        self.text_supervision = h.text_supervision

        embedder_dim = h.get("embedder_dim", None)
        if self.multispkr and not embedder_dim:
            self.spkr = nn.Embedding(h.get("num_speakers", 200), h["embedding_dim"])
        elif embedder_dim:
            self.spkr = nn.Linear(embedder_dim, h["embedding_dim"])

        self.layer = nn.Sequential(
            nn.ConvTranspose1d(h.embedding_dim, h.embedding_dim, kernel_size=4, stride=2, padding=1),  # up-samples
            nn.GELU(),
        )
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(h.embedding_dim, h.embedding_dim)  # no change in dims

        if self.text_supervision:
            self.layer_text = nn.Sequential(
                nn.Embedding(h.num_embeddings_text, h.embedding_dim_text),  # (4000, 589) -> https://ai.stackexchange.com/a/37168
                CustomPermute(0, 2, 1),
                nn.ConvTranspose1d(h.embedding_dim_text, h.embedding_dim_text, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                CustomPermute(0, 2, 1),
                nn.Dropout(0.1),
                nn.Linear(h.embedding_dim_text, h.embedding_dim_text),
                CustomPermute(0, 2, 1),
            )

        self.debug = bool(int(os.environ.get('DEBUG', 0)))

    def forward(self, **kwargs):
        # code = (B, T/2) 
        # mel = (B, 80, T)
        # spkr = (B, 256)

        x = kwargs['mel']

        code = self.dict(kwargs['code'])  # creates 128 dimensions (C) (embedding) for each code label (timestep) -> (B, T/2, 128)
        code = self.layer(code.permute(0,2,1)).permute(0,2,1)  # after permutes: (B, 128, T/2), (B, T, 128) -> up-samples (doubles) channels (in this case T)
        code = self.dropout(code)
        code = self.fc(code)
        code = code.permute(0,2,1)  # after: (B, 128, T)

        x = torch.cat([x, code], dim=1)  # mel and code have same no. timesteps, concats along the feature (C) dimension

        if self.text_supervision:
            t_label = self.layer_text(kwargs['t_label'])
            x = torch.cat([x, t_label], dim=1)

        if self.multispkr:
            spkr = self.spkr(kwargs['spkr'])  # speaker embedding 256 -> 128 dims (linear layer)
            spkr = self._upsample(spkr, x.shape[-1])  # up-sampled to same number of timesteps as mel + code
            x = torch.cat([x, spkr], dim=1)  # concat along the C dimension e.g. (B, C1, T) + (B, C2, T) = (B, C1 + C2, T), no. timesteps are the same

        # at this point, x = mel-spec + unit codes + spk embedding = 80 dims + 128 dims + 128 dims = 336 dims
        # 336 dims is input to conv_pre in CodeGenerator class

        for k, feat in kwargs.items():
            if k in ['spkr', 'code', 'mel', 't_label']:
                continue

            feat = self._upsample(feat, x.shape[-1])
            x = torch.cat([x, feat], dim=1)

        if self.debug:
            print('Input', x.shape)

        return super(CodeGenerator, self).forward(x)
