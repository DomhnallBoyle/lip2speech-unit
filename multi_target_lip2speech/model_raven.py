import sys,logging
import contextlib
from argparse import Namespace
from typing import Optional

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import FairseqEncoder, FairseqEncoderModel, register_model
from omegaconf import MISSING

DBG=True if len(sys.argv) == 1 else False

if DBG:
    pass
else:
    from avhubert.hubert_asr import HubertEncoderWrapper, AVHubertAsrConfig, AVHubertSeq2SeqConfig, Linear, Embedding
    from pathlib import Path
    # sys.path.insert(0, Path(__file__).resolve().parent.parent)
    from espnet.nets.pytorch_backend.transformer.encoder import Encoder
    # sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath('raven')))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath('raven')))
    from raven._espnet.nets.pytorch_backend.transformer.encoder import Encoder as RAVENTransformerEncoder
    sys.path.pop(0)
    sys.path.pop(0)
    from .model import MultiTargetRAVENEncoderModelConfig, MLP, TextClassifier
    from .helpers import SentenceProcessor

logger = logging.getLogger(__name__)


@register_model("multi_target_raven", dataclass=MultiTargetRAVENEncoderModelConfig)
class MultiTargetRAVENEncoderModel(FairseqEncoderModel):
    def __init__(self, encoder, tgt_dict, cfg, conformer):
        super().__init__(encoder)
        self.cfg = cfg
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.conformer = conformer
        self.init()

    def init(self):
        # load auto-avsr model, only the encoder part i.e. ResNet + Conformer
        if not self.cfg.raven_checkpoint_path:
            return
        
        state = torch.load(self.cfg.raven_checkpoint_path, map_location='cpu')
        state_keys = list(state.keys())
        for k in state_keys:
            if any([name in k for name in ['decoder', 'ctc']]):
                del state[k]

        self.encoder.load_state_dict(state)

        # freeze auto-avsr encoder params during training
        for param in self.encoder.parameters():
            param.requires_grad = False

        logger.info('RAVEN PRETRAINED FRONTEND LOADED')

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        raven_encoder = RAVENEncoder(cfg)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        cfg.decoder_embed_dim = len(tgt_dict)

        conformer = None
        if cfg.use_conformer:
            conformer = Conformer(cfg)

        return cls(raven_encoder, tgt_dict, cfg, conformer)

    def forward(self, **kwargs):
        with torch.no_grad():
            output = self.encoder(**kwargs)

        if self.cfg.use_conformer:
            output = self.conformer(
                source=output['encoder_out'].repeat_interleave(2, dim=0),
                padding_mask=output['encoder_padding_mask'].repeat_interleave(2, dim=1),
                spk_emb=kwargs['spk_emb']
            )

        output['encoder_out'] = output['encoder_out'].transpose(0,1).contiguous()

        return output

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class RAVENEncoder(FairseqEncoder):
    def __init__(self, cfg):
        super().__init__(None)

        self.encoder = RAVENTransformerEncoder(
            idim=cfg.encoder_idim,
            attention_dim=cfg.encoder_attention_dim, # adim,
            attention_heads=cfg.encoder_attention_heads, # aheads,
            linear_units=cfg.encoder_linear_units, # eunits,
            num_blocks=cfg.encoder_num_blocks, #elayers,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            frontend="conv3d",
            input_layer="vanilla_linear",
            macaron_style=False,
            encoder_attn_layer_type="rel_mha",
            use_cnn_module=False,
            zero_triu=False,
            cnn_module_kernel=31,
            relu_type="swish",
            a_upsample_ratio=1,
            layerscale=True,
            init_values=0.1,
            ff_bn_pre=True,
            post_norm=False,
            gamma_zero=False,
            gamma_init=0.1,
            mask_init_type=None,
            drop_path=0.1
        )

    def forward(self, source, padding_mask, spk_emb):
        # padding_mask = N x 100
        # spk_emb = N x 256

        video = source['video'].squeeze(1)  # [B, T, H, W]
        masks = ~padding_mask.unsqueeze(-2)  # [B, 1, 100]

        x, masks = self.encoder(video, masks=masks)  # x = [B, T, 768]

        return {
            'encoder_out': x.transpose(0, 1),  # [T, B, C]
            'encoder_padding_mask': padding_mask,
            'padding_mask': padding_mask
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out


class Conformer(FairseqEncoder):
    def __init__(self, cfg, tgt_dict=None):
        super().__init__(None)

        self.encoder = Encoder(
            idim=-1,
            attention_dim=cfg.conformer_embed_dim, # adim
            attention_heads=cfg.conformer_attention_heads, # aheads
            linear_units=cfg.conformer_ffn_embed_dim, # eunits
            num_blocks=cfg.conformer_layers, #elayers
            dropout_rate=cfg.conformer_dropout, # dropout_rate
            positional_dropout_rate=cfg.conformer_dropout, # dropout_rate
            attention_dropout_rate=cfg.conformer_attention_dropout, # transformer_attn_dropout_rate
            input_layer="conv3d", # transformer_input_layer
            normalize_before=cfg.conformer_layer_norm_first,
            macaron_style=1, # macaron_style
            encoder_attn_layer_type="rel_mha", # transformer_encoder_attn_layer_type
            use_cnn_module=1, # use_cnn_module
            zero_triu=False, # zero_triu
            cnn_module_kernel=31, # cnn_module_kernel
            relu_type="swish", # relu_type
            a_upsample_ratio=1, # a_upsample_ratio,
        )
        self.encoder.frontend = None

        d = cfg.conformer_embed_dim

        self.sentence_processor = SentenceProcessor() if cfg.text_supervision else None
        self.text_classifier = TextClassifier(input_dim=d, num_classes=self.sentence_processor.num_classes) if cfg.text_supervision else None

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.num_updates = 0

        self.proj_in = Linear(cfg.encoder_attention_dim, d)

        if tgt_dict is not None:
            # self.proj_out = Linear(d, len(tgt_dict))
            self.proj_out = MLP(
                d, [d, d, len(tgt_dict)], cfg.final_dropout, nn.GELU
            )
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj_out = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj_out = None

        self.mel_conv = nn.Sequential(
            nn.Conv1d(in_channels=d+256,out_channels=d,kernel_size=3,stride=1,padding=1),
            nn.Dropout(cfg.final_dropout),
            nn.GELU(),
            nn.Conv1d(in_channels=d,out_channels=d,kernel_size=3,stride=1,padding=1),
            nn.Dropout(cfg.final_dropout),
            nn.GELU(),
            nn.Conv1d(in_channels=d,out_channels=d,kernel_size=3,stride=1,padding=1),
            nn.Dropout(cfg.final_dropout),
            nn.GELU(),
        )
        self.mel_proj = Linear(d, 160)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, spk_emb=None, tbc=True, **kwargs):

        x = source

        if tbc:
            # T x B x C -> B x T x C
            x = x.transpose(0, 1)

        x = self.proj_in(x)

        x, masks = self.encoder.forward_after_frontend(
            x,
            masks = ~padding_mask.unsqueeze(-2),
        )

        padding_mask = ~masks.squeeze(-2)

        if spk_emb is not None:
            assert spk_emb.size(-1) == 256
            spk_x = torch.cat([spk_emb.unsqueeze(1).repeat(1,x.size(1),1), x], dim=-1)
        else:
            spk_x = x

        encoder_out_mel = self.mel_proj(self.mel_conv(spk_x.transpose(1,2)).transpose(1,2))

        B, T, D = encoder_out_mel.shape
        encoder_out_mel = encoder_out_mel.reshape(B, T, D//2, 2).transpose(-1,-2).reshape(B, T*2, D//2)

        if tbc:
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj_out:
            unit_proj = self.proj_out(x)

        out = {
            "encoder_out": unit_proj,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
            "encoder_out_mel": encoder_out_mel,
        }

        if self.text_classifier:
            out['encoder_out_text'] = self.text_classifier(x)

        return out

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict
