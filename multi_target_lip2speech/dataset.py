import itertools
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from python_speech_features import logfbank
from scipy.io import wavfile

DBG=True if len(sys.argv) == 1 else False

if DBG:
    import utils_aug as custom_utils
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
        stream=sys.stdout,
    )
    from ..avhubert.hubert_dataset import load_audio_visual, load_label, load_label_offset, verify_label_lengths, AVHubertDataset
else:
    from . import utils_aug as custom_utils
    from avhubert.hubert_dataset import load_audio_visual, load_label, load_label_offset, verify_label_lengths, AVHubertDataset
    from .helpers import SentenceProcessor

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from helpers import load_groundtruth_data

logger = logging.getLogger(__name__)


class MultiTargetDataset(AVHubertDataset):
    def __init__(
            self,
            manifest_path: str,
            gt_path: str,
            sample_rate: float,
            label_paths: List[str],
            label_rates: Union[List[float], float],  # -1 for sequence labels
            pad_list: List[str],
            eos_list: List[str],
            label_processors: Optional[List[Any]] = None,
            max_keep_sample_size: Optional[int] = None,
            min_keep_sample_size: Optional[int] = None,
            max_sample_size: Optional[int] = None,
            shuffle: bool = True,
            pad_audio: bool = False,
            normalize: bool = False,
            store_labels: bool = True,
            random_crop: bool = False,
            single_target: bool = False,
            stack_order_audio: int=1,
            skip_verify: bool=False,
            image_mean: float=0,
            image_std: float=1,
            image_crop_size: int=88,
            image_aug: bool=False,
            modalities: Optional[List[str]]=None,
            is_s2s=False,
            noise_fn=None,
            noise_prob=0,
            noise_snr=0,
            noise_num=1,
            time_mask: bool = False,
            random_erase: bool = False,
            text_supervision: bool = False,
            grayscale_transform: bool = False
    ):
        # load gt if available
        self.gt_path = gt_path
        self.text_supervision = text_supervision and Path(gt_path).exists()
        self.sentence_processor = SentenceProcessor() if self.text_supervision else None
        self.gt_d = self.load_gt_data(gt_path=gt_path) if self.text_supervision else None
        self.grayscale_transform = grayscale_transform

        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, int)
            else label_rates
        )
        self.modalities = set(modalities)
        self.audio_root, self.names, inds, tot, self.sizes = load_audio_visual(
            manifest_path, 
            max_keep_sample_size, 
            min_keep_sample_size, 
            frame_rate=sample_rate, 
            label_paths=label_paths, 
            label_rates=self.label_rates,
            gt_d=self.gt_d
        )
        self.sample_rate = sample_rate
        self.stack_order_audio = stack_order_audio
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.store_labels = store_labels
        self.is_s2s = is_s2s
        self.noise_wav, self.noise_prob, self.noise_snr, self.noise_num = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else [], noise_prob, noise_snr, noise_num

        # assert self.single_target == (self.label_rates[0] == -1), f"single target should be equivalent to sequence label (label_rate==-1)"
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        assert (
            label_processors is None
            or len(label_processors) == self.num_labels
        )
        if not skip_verify:
            for label_path, label_rate in zip(label_paths, self.label_rates):
                verify_label_lengths(self.sizes, self.sample_rate, label_path, label_rate, inds, tot)
        else:
            logger.info(f"Skip label alignment verifying")

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        if image_aug:
            if self.grayscale_transform:
                transforms = [
                    custom_utils.Normalize(0.0, 255.0),
                    custom_utils.RandomCrop((image_crop_size, image_crop_size)),
                    custom_utils.Grayscale(),
                    custom_utils.HorizontalFlip(0.5),
                    custom_utils.Normalize(image_mean, image_std) 
                ]
            else:
                transforms = [
                    custom_utils.Normalize(0.0, 255.0),
                    custom_utils.RandomCrop((image_crop_size, image_crop_size)),
                    custom_utils.HorizontalFlip(0.5),  # flip around y-axis
                    custom_utils.Normalize(image_mean, image_std) 
                ]

            transforms += [custom_utils.RandomErase(0.5)] if random_erase else []
            transforms += [custom_utils.TimeMask()] if time_mask else []
            self.transform = custom_utils.Compose(transforms)
        else:
            if self.grayscale_transform:
                transforms = [
                    custom_utils.Normalize(0.0, 255.0),
                    custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                    custom_utils.Grayscale(),
                    custom_utils.Normalize(image_mean, image_std)
                ]
            else:
                transforms = [
                    custom_utils.Normalize(0.0, 255.0),
                    custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                    custom_utils.Normalize(image_mean, image_std)
                ]
            self.transform = custom_utils.Compose(transforms)

        logger.info(f"image transform: {self.transform}")

        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}, "
            f"seqs2seq data={self.is_s2s},")
        logger.info(
            f"Noise wav: {noise_fn}->{len(self.noise_wav)} wav, Prob: {self.noise_prob}, SNR: {self.noise_snr}, Number of mixture: {self.noise_num}"
        )

    def load_gt_data(self, gt_path):
        gt_df = load_groundtruth_data(gt_path, skip_lines=True)[0]
        
        return {
            row['Video Name']: row['Phrase'] for _, row in gt_df.iterrows() 
            if self.sentence_processor.is_valid(row['Phrase'])
        }
    
    def load_text_labels(self, video_path):
        _id = Path(video_path).stem
        try:
            text = self.gt_d[_id]
        except KeyError as e:
            logger.info(f'ID not found: {self.gt_path}, {_id}')
            raise e

        return self.sentence_processor.encode(text)

    def load_additional_feature(self, mix_name):
        video_fn, audio_fn = mix_name

        mel_fn = os.path.join(self.audio_root, video_fn).replace('/video/', '/mel/')[:-4]+'.npy'
        if os.path.exists(mel_fn):
            mel = np.load(mel_fn)
        else:
            raise FileNotFoundError(f"{mel_fn} does not exist")

        spk_emb_fn = os.path.join(self.audio_root, video_fn).replace('/video/', '/spk_emb/')[:-4]+'.npy'
        if os.path.exists(spk_emb_fn):
            spk_emb = np.load(spk_emb_fn)
        else:
            raise FileNotFoundError(f"{spk_emb_fn} does not exist")

        return mel, spk_emb

    def load_video(self, audio_name):
        if not self.grayscale_transform:
            return super().load_video(audio_name)  # loads in grayscale by default

        # read video in colour (BGR), convert to grayscale later
        feats = custom_utils.load_video(os.path.join(self.audio_root, audio_name), grayscale=False)
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        
        return feats

    def __getitem__(self, index):
        sample = super().__getitem__(index)  # speech units loaded in base class
        mel, spk_emb = self.load_additional_feature(self.names[index])

        mel = torch.from_numpy(mel.astype(np.float32))
        spk_emb = torch.from_numpy(spk_emb.astype(np.float32))

        sample["mel"] = mel
        sample["spk_emb"] = spk_emb
        sample['names'] = self.names[index]

        if self.text_supervision:
            text_labels = self.load_text_labels(video_path=self.names[index][0])
            sample['text_labels'] = torch.from_numpy(text_labels.astype(np.int))

        return sample

    def collater(self, samples):
        batch = super().collater(samples)

        max_mel_len = max(len(s["mel"]) for s in samples)
        batch["mel"] = torch.stack([torch.nn.functional.pad(s["mel"], [0, 0, 0, max_mel_len - len(s["mel"])]) for s in samples])

        batch["net_input"]["spk_emb"] = torch.stack([s["spk_emb"] for s in samples])

        if self.text_supervision:
            # 'text_labels' are 1-dimensional, no need for padding
            batch['text_labels'] = torch.from_numpy(np.concatenate([s['text_labels'] for s in samples])).int()
            batch['text_labels_lengths'] = torch.Tensor([s['text_labels'].shape[0] for s in samples]).int()

            assert batch['text_labels'].shape[0] == batch['text_labels_lengths'].sum()

        return batch
