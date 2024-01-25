import os
from pathlib import Path

import torch
import numpy as np
import sentencepiece as spm

# 26 chars + 0-9 + apostrophe + space = 38 chars = 0-37 index, # = blank
CHARS = '#abcdefghijklmnopqrstuvwxyz0123456789 \''
DATA_PATH = Path(__file__).resolve().parents[0].joinpath('data')
LRS3_LANGUAGE_MODEL_PATH = str(DATA_PATH.joinpath('lrs2lrs3_lower.model'))
LRS3_VOCAB_PATH = str(DATA_PATH.joinpath('lrs2lrs3_lower.vocab'))


class SentenceProcessor:

    def __init__(self):
        self.char_level = bool(int(os.environ.get('CHAR_LEVEL', 0)))
        self.sp = spm.SentencePieceProcessor(model_file=LRS3_LANGUAGE_MODEL_PATH) if not self.char_level else None
        self.classes = self.get_classes()
        self.num_classes = self.sp.get_piece_size() if not self.char_level else len(self.classes)
        self.blank = 0

    def encode(self, text):
        if self.char_level:
            return np.array([CHARS.index(c) for c in text])

        return np.asarray(self.sp.encode(text))

    def decode(self, indices):
        if self.char_level:
            return ''.join([CHARS[l] for l in indices])

        return self.sp.decode(indices.tolist())

    def get_classes(self):
        if self.char_level:
            return CHARS

        with open(LRS3_VOCAB_PATH, encoding='utf-8') as f:
            return [l.strip().split('\t')[0] for l in f.readlines()]

    def is_valid(self, text):
        if self.char_level:
            return all([c in CHARS for c in text])
        
        return True

    def replace_repeated_indices(self, indices):
        # replace consecutive repeated tokens with blanks
        indices = [int(x) for x in indices]

        i = 0
        new_indices = []
        while i < len(indices) - 1:
            if indices[i] == 0 or indices[i] != indices[i + 1]:
                new_indices.append(indices[i])
                i += 1
                continue

            repeated_token = indices[i]
            new_indices.append(repeated_token)
            for j in range(i + 1, len(indices)):
                if indices[j] == repeated_token:
                    new_indices.append(0)
                else:
                    break

            i = j

        if len(new_indices) < len(indices):
            new_indices.append(indices[-1])

        assert len(new_indices) == len(indices)

        return new_indices


def load_pretrained_frontend(frontend, pretrained_path):
    pm = torch.load(pretrained_path, map_location='cpu')

    def copy(p, v):
        p.data.copy_(v)

    # stem
    copy(frontend.frontend3D[0].weight, pm['encoder.frontend.frontend3D.0.weight'])
    for attr in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']: 
        copy(getattr(frontend.frontend3D[1], attr), pm[f'encoder.frontend.frontend3D.1.{attr}'])

    # trunks
    for i in range(1, 5):  # layer 
        for j in range(2):  # block
            # conv_2d_1
            copy(
                getattr(frontend.trunk, f'layer{i}')[j].conv1.weight,
                pm[f'encoder.frontend.trunk.layer{i}.{j}.conv1.weight']
            )

            # batch_norm_2d_1
            for attr in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                copy(
                    getattr(getattr(frontend.trunk, f'layer{i}')[j].bn1, attr),
                    pm[f'encoder.frontend.trunk.layer{i}.{j}.bn1.{attr}']
                )

            # conv_2d_2
            copy(
                getattr(frontend.trunk, f'layer{i}')[j].conv2.weight,
                pm[f'encoder.frontend.trunk.layer{i}.{j}.conv2.weight']
            )

            # batch_norm_2d_1
            for attr in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                copy(
                    getattr(getattr(frontend.trunk, f'layer{i}')[j].bn2, attr),
                    pm[f'encoder.frontend.trunk.layer{i}.{j}.bn2.{attr}']
                )

            # occurs in first block of layers 2-4
            is_downsample = i in [2, 3, 4] and j == 0
            if is_downsample: 
                copy(
                    getattr(frontend.trunk, f'layer{i}')[j].downsample[0].weight,
                    pm[f'encoder.frontend.trunk.layer{i}.{j}.downsample.0.weight']
                )
                for attr in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                    copy(
                        getattr(getattr(frontend.trunk, f'layer{i}')[j].downsample[1], attr),
                        pm[f'encoder.frontend.trunk.layer{i}.{j}.downsample.1.{attr}']
                    )

    return frontend
