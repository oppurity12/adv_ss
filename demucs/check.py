import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import torch as th
from torch import distributed, nn
from torch.utils.data import ConcatDataset, DataLoader
from torch.nn.parallel.distributed import DistributedDataParallel

from .augment import FlipChannels, FlipSign, Remix, Scale, Shift
from .compressed import get_compressed_datasets
from .model import Demucs
from .parser import get_name, get_parser
from .raw import Rawset
from .repitch import RepitchedWrapper
from .pretrained import load_pretrained, SOURCES
from .tasnet import ConvTasNet
from .test import evaluate
from .train import train_model, validate_model
from .utils import (human_seconds, load_model, save_model, get_state,
                    save_state, sizeof_fmt, get_quantizer)
from .wav import get_wav_datasets, get_musdb_wav_datasets



def main():
    parser = get_parser()
    args = parser.parse_args()
    name = get_name(parser, args)
    print(f"Experiment {name}")

    if args.musdb is None and args.rank == 0:
        print(
            "You must provide the path to the MusDB dataset with the --musdb flag. "
            "To download the MusDB dataset, see https://sigsep.github.io/datasets/musdb.html.",
            file=sys.stderr)
        sys.exit(1)

    eval_folder = args.evals / name
    eval_folder.mkdir(exist_ok=True, parents=True)
    args.logs.mkdir(exist_ok=True)
    metrics_path = args.logs / f"{name}.json"
    eval_folder.mkdir(exist_ok=True, parents=True)
    args.checkpoints.mkdir(exist_ok=True, parents=True)
    args.models.mkdir(exist_ok=True, parents=True)

    if args.device is None:
        device = "cpu"
        if th.cuda.is_available():
            device = "cuda"
    else:
        device = args.device

    th.manual_seed(args.seed)
    # Prevents too many threads to be started when running `museval` as it can be quite
    # inefficient on NUMA architectures.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if args.world_size > 1:
        if device != "cuda" and args.rank == 0:
            print("Error: distributed training is only available with cuda device", file=sys.stderr)
            sys.exit(1)
        th.cuda.set_device(args.rank % th.cuda.device_count())
        distributed.init_process_group(backend="nccl",
                                       init_method="tcp://" + args.master,
                                       rank=args.rank,
                                       world_size=args.world_size)

    checkpoint = args.checkpoints / f"{name}.th"
    checkpoint_tmp = args.checkpoints / f"{name}.th.tmp"
    if args.restart and checkpoint.exists() and args.rank == 0:
        checkpoint.unlink()

    if args.test or args.test_pretrained:
        args.epochs = 1
        args.repeat = 0
        if args.test:
            model = load_model(args.models / args.test)
        else:
            model = load_pretrained(args.test_pretrained)
    elif args.tasnet:
        model = ConvTasNet(audio_channels=args.audio_channels,
                           samplerate=args.samplerate, X=args.X,
                           segment_length=4 * args.samples,
                           sources=SOURCES)
    else:
        model = Demucs(
            audio_channels=args.audio_channels,
            channels=args.channels,
            context=args.context,
            depth=args.depth,
            glu=args.glu,
            growth=args.growth,
            kernel_size=args.kernel_size,
            lstm_layers=args.lstm_layers,
            rescale=args.rescale,
            rewrite=args.rewrite,
            stride=args.conv_stride,
            resample=args.resample,
            normalize=args.normalize,
            samplerate=args.samplerate,
            segment_length=4 * args.samples,
            sources=SOURCES,
        )
    model.to(device)
    
    # augment = [Shift(args.data_stride)]
    augment = [Shift(args.data_stride)]

    # if args.augment:
    #     augment += [FlipSign(), FlipChannels(), Scale(),
    #                 Remix(group_size=args.remix_group_size)]
    augment = nn.Sequential(*augment).to(device)
    print("Agumentation pipeline:", augment)


    # Setting number of samples so that all convolution windows are full.
    # Prevents hard to debug mistake with the prediction being shifted compared
    # to the input mixture.
    samples = model.valid_length(args.samples)
    print(f"Number of training samples adjusted to {samples}")
    samples = samples + args.data_stride
    if args.repitch:
        # We need a bit more audio samples, to account for potential
        # tempo change.
        samples = math.ceil(samples / (1 - 0.01 * args.max_tempo))
    args.metadata.mkdir(exist_ok=True, parents=True)
    train_set, valid_set = get_compressed_datasets(args, samples)


    print("Train set and valid set sizes", len(train_set), len(valid_set))

    data_loader = DataLoader(train_set, batch_size=3)
    it = iter(data_loader)
    print(next(it).shape)


if __name__ == '__main__':
  main()