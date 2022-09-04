# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import sys
from concurrent import futures

import musdb
import museval
import torch as th
import tqdm
from scipy.io import wavfile
from torch import distributed
from zmq import device

from .audio import convert_audio
from .utils import apply_model, pgd_attack_demucs, pgd_attack_tasnet
from .attack import PGD


from torchmetrics import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio, SignalNoiseRatio

import torch
import gc


def evaluate(model,
             musdb_path,
             eval_folder,
             workers=2,
             device="cpu",
             rank=0,
             save=False,
             shifts=0,
             split=False,
             overlap=0.25,
             is_wav=False,
             world_size=1,
             writer=None,
             ):
    """
    Evaluate model using museval. Run the model
    on a single GPU, the bottleneck being the call to museval.
    """

    output_dir = eval_folder / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    json_folder = eval_folder / "results/test"
    json_folder.mkdir(exist_ok=True, parents=True)

    # we load tracks from the original musdb set
    test_set = musdb.DB(musdb_path, subsets=["test"], is_wav=is_wav)
    src_rate = 44100  # hardcoded for now...
    
    for p in model.parameters():
        p.requires_grad = False
        p.grad = None
    
    SNR = SignalNoiseRatio().to(device)
    SDR = SignalDistortionRatio().to(device)
    SI_SDR = ScaleInvariantSignalDistortionRatio().to(device)

    pendings = []
    total_result = {}
    total_middele_result = {}

    for name in model.sources:
        total_result[f"total {name} sdr"] = 0
        total_result[f"total {name} si_sdr"] = 0

        total_middele_result[f"total middle {name} sdr"] = 0
        total_middele_result[f"total middle {name} si_sdr"] = 0


    for index in tqdm.tqdm(range(rank, len(test_set), world_size), file=sys.stdout):
        track = test_set.tracks[index]

        out = json_folder / f"{track.name}.json.gz"
        if out.exists():
            continue

        mix = th.from_numpy(track.audio).t().float()
        ref = mix.mean(dim=0)  # mono mixture
        mix = (mix - ref.mean()) / ref.std()
        mix = convert_audio(mix, src_rate, model.samplerate, model.audio_channels)

        
        estimates = apply_model(model, mix.to(device),
                                shifts=shifts, split=split, overlap=overlap)

        estimates = estimates * ref.std() + ref.mean()

        references = th.stack(
            [th.from_numpy(track.targets[name].audio).t() for name in model.sources])
        references = convert_audio(references, src_rate,
                                    model.samplerate, model.audio_channels)
        
        references_ = references.to(device)

        for idx, target in enumerate(model.sources):
            results = {}
            # results[f'{target} snr'] = SNR(estimates[idx], references_[idx]).item()
            audio_length = estimates.size(-1)
            middle = audio_length // 2
            start = middle - (model.samplerate * 15)
            end = middle + (model.samplerate * 15)

            # 전체 음성
            sdr = SDR(estimates[idx], references_[idx]).item()
            si_sdr = SI_SDR(estimates[idx], references_[idx]).item()

            # 중간 30초 음성
            # print(f"start-end, {start} - {end}")
            # middle_sdr = SDR(estimates[idx,:,start:end], references_[idx,:,start:end]).item()
            # middle_si_sdr = SI_SDR(estimates[idx,:,start:end], references_[idx,:,start:end]).item()
            # print(f"middle_sdr: {middle_sdr}")

            results[f'{target} sdr'] = sdr
            results[f'{target} si_sdr'] = si_sdr

            # results[f"{target} middle sdr"] = middle_sdr
            # results[f"{target} middle si_sdr"] = middle_si_sdr

            # if sdr < 0 or si_sdr < 0:
            #     continue

            total_result[f'total {target} sdr'] += sdr / len(test_set)
            total_result[f'total {target} si_sdr'] += si_sdr / len(test_set)

            # total_middele_result[f'total middle {target} sdr'] += middle_sdr / len(test_set)
            # total_middele_result[f'total middle {target} si_sdr'] += middle_si_sdr / len(test_set)

            # print(results)

            writer.add_scalars(f"track_name: {track.name} target: {target}", results)

            del sdr, si_sdr
        
        references = references.transpose(1, 2).numpy()
        estimates = estimates.transpose(1, 2)
        estimates = estimates.cpu().numpy()
        # win = int(30. * model.samplerate)
        # hop = int(15. * model.samplerate)
        if save:
            folder = eval_folder / "wav/test" / track.name
            folder.mkdir(exist_ok=True, parents=True)
            for name, estimate in zip(model.sources, estimates):
                wavfile.write(str(folder / (name + ".wav")), model.samplerate, estimate)
        

        # sdr, sir, isr, sar = museval.evaluate(references, estimates, win=win, hop=hop)
        # track_store = museval.TrackStore(win=44100, hop=44100, track_name=track.name)
        # for idx, target in enumerate(model.sources):
        #     values = {
        #         "SDR": sdr[idx].tolist(),
        #         "SIR": sir[idx].tolist(),
        #         "ISR": isr[idx].tolist(),
        #         "SAR": sar[idx].tolist()
        #     }

        #     track_store.add_target(target_name=target, values=values)
        #     json_path = json_folder / f"{track.name}.json.gz"
        #     gzip.open(json_path, "w").write(track_store.json.encode('utf-8'))


        # if workers:
        #     pendings.append((track.name, pool.submit(
        #         museval.evaluate, references, estimates, win=win, hop=hop)))
        # else:
        # pendings.append((track.name, museval.evaluate(
        #     references, estimates, win=win, hop=hop)))

        del references, mix, estimates, references_

        gc.collect()
        torch.cuda.empty_cache()
    
    writer.add_scalars("total_result", total_result)
    # writer.add_scalars("total_middle_result", total_middele_result)


    # for track_name, pending in tqdm.tqdm(pendings, file=sys.stdout):
    #     sdr, isr, sir, sar = pending
    #     track_store = museval.TrackStore(win=44100, hop=44100, track_name=track_name)
    #     for idx, target in enumerate(model.sources):
    #         values = {
    #             "SDR": sdr[idx].tolist(),
    #             "SIR": sir[idx].tolist(),
    #             "ISR": isr[idx].tolist(),
    #             "SAR": sar[idx].tolist()
    #         }
            
    #         track_store.add_target(target_name=target, values=values)
    #         json_path = json_folder / f"{track_name}.json.gz"
    #         gzip.open(json_path, "w").write(track_store.json.encode('utf-8'))


    if world_size > 1:
        distributed.barrier()


def ds_sdr(reference, src, adv_src):
    SDR = SignalDistortionRatio().to(src.device)
    return SDR(reference, src) - SDR(reference, adv_src)

def di_sdr(src, adv_src):
    device_ = adv_src.device
    SDR = SignalDistortionRatio().to(device_)
    return SDR(src.to(device_), adv_src)


def attack_evaluate(model,
             musdb_path,
             eval_folder,
             workers=2,
             device="cpu",
             rank=0,
             save=False,
             shifts=0,
             split=False,
             overlap=0.25,
             is_wav=False,
             world_size=1,
             writer=None,
             pretrained='demucs'
             ):
    """
    Evaluate model using museval. Run the model
    on a single GPU, the bottleneck being the call to museval.
    """

    output_dir = eval_folder / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    json_folder = eval_folder / "results/test"
    json_folder.mkdir(exist_ok=True, parents=True)

    # we load tracks from the original musdb set
    test_set = musdb.DB(musdb_path, subsets=["test"], is_wav=is_wav)
    src_rate = 44100  # hardcoded for now...

    SDR = SignalDistortionRatio().to(device)

    total_result = {}
    total_middele_result = {}

    for name in model.sources:
        total_result[f"total {name} sdr"] = 0
        total_result[f"total {name} si_sdr"] = 0

        total_middele_result[f"total middle {name} sdr"] = 0
        total_middele_result[f"total middle {name} si_sdr"] = 0


    for index in tqdm.tqdm(range(rank, len(test_set), world_size), file=sys.stdout):
        track = test_set.tracks[index]
        print(f"track_name: {track.name}")

        out = json_folder / f"{track.name}.json.gz"
        if out.exists():
            continue

        mix = th.from_numpy(track.audio).t().float()
        ref = mix.mean(dim=0)  # mono mixture
        mix = (mix - ref.mean()) / ref.std()
        mix = convert_audio(mix, src_rate, model.samplerate, model.audio_channels)

        if pretrained == 'demucs':
            estimates, adv_estimates, adv_audio  = pgd_attack_demucs(model, mix.to(device),
                                    shifts=shifts, split=split, overlap=overlap, steps=40, eps=0.05)
        else:
            estimates, adv_estimates, adv_audio  = pgd_attack_tasnet(model, mix.to(device),
                                    shifts=shifts, split=split, overlap=overlap, steps=40, eps=0.05)
        
        di_sdr_ = di_sdr(mix, adv_audio).item()
        writer.add_scalar(f"track_name: {track.name} di_sdr", di_sdr_)

        del di_sdr_

        mix = mix * ref.std() + ref.mean()
        mix = mix.transpose(0, 1)
        mix = mix.cpu().numpy()

        adv_audio = adv_audio * ref.std() + ref.mean()
        adv_audio = adv_audio.transpose(0, 1)
        adv_audio = adv_audio.cpu().numpy()

        folder = eval_folder / "wav/test" / track.name
        folder.mkdir(exist_ok=True, parents=True)

        wavfile.write(str(folder / ("normal.wav")), model.samplerate, mix)
        wavfile.write(str(folder / ("adversarial.wav")), model.samplerate, adv_audio)
        
        del adv_audio

        estimates = estimates * ref.std() + ref.mean()
        adv_estimates = adv_estimates * ref.std() + ref.mean()

        references = th.stack(
            [th.from_numpy(track.targets[name].audio).t() for name in model.sources])
        references = convert_audio(references, src_rate,
                                    model.samplerate, model.audio_channels)
        
        references_ = references.to(device)

        for idx, target in enumerate(model.sources):
            results = {}
            sdr = SDR(estimates[idx], references_[idx]).item()
            adv_sdr = SDR(adv_estimates[idx], references_[idx]).item()

            ds_sdr_ =  ds_sdr(references_[idx], estimates[idx], adv_estimates[idx]).item()

            results[f'{target} sdr'] = sdr
            results[f'{target} adv_sdr'] = adv_sdr

            results[f'{target} ds_sdr'] = ds_sdr_

            total_result[f'total {target} sdr'] += sdr / len(test_set)

            writer.add_scalars(f"track_name: {track.name} target: {target}", results)

            del sdr, adv_sdr, ds_sdr_
        
        references = references.transpose(1, 2).numpy()
        estimates = estimates.transpose(1, 2)
        estimates = estimates.cpu().numpy()
        
        adv_estimates = adv_estimates.transpose(1, 2)
        adv_estimates = adv_estimates.cpu().numpy()

        # win = int(30. * model.samplerate)
        # hop = int(15. * model.samplerate)
        if save and index == 0:
            for name, estimate in zip(model.sources, estimates):
                wavfile.write(str(folder / (name + ".wav")), model.samplerate, estimate)
            
            for name, estimate in zip(model.sources, adv_estimates):
                wavfile.write(str(folder / ("adv_" + name + ".wav")), model.samplerate, estimate)

        # sdr, sir, isr, sar = museval.evaluate(references, estimates, win=win, hop=hop)
        # track_store = museval.TrackStore(win=44100, hop=44100, track_name=track.name)
        # for idx, target in enumerate(model.sources):
        #     values = {
        #         "SDR": sdr[idx].tolist(),
        #         "SIR": sir[idx].tolist(),
        #         "ISR": isr[idx].tolist(),
        #         "SAR": sar[idx].tolist()
        #     }

        #     track_store.add_target(target_name=target, values=values)
        #     json_path = json_folder / f"{track.name}.json.gz"
        #     gzip.open(json_path, "w").write(track_store.json.encode('utf-8'))


        # if workers:
        #     pendings.append((track.name, pool.submit(
        #         museval.evaluate, references, estimates, win=win, hop=hop)))
        # else:
        # pendings.append((track.name, museval.evaluate(
        #     references, estimates, win=win, hop=hop)))

        del ref, references, references_, mix, estimates, adv_estimates

        gc.collect()
        torch.cuda.empty_cache()
        
        if index > 5:
            break

    
    writer.add_scalars("total_result", total_result)

    # writer.add_scalars("total_middle_result", total_middele_result)


    # for track_name, pending in tqdm.tqdm(pendings, file=sys.stdout):
    #     sdr, isr, sir, sar = pending
    #     track_store = museval.TrackStore(win=44100, hop=44100, track_name=track_name)
    #     for idx, target in enumerate(model.sources):
    #         values = {
    #             "SDR": sdr[idx].tolist(),
    #             "SIR": sir[idx].tolist(),
    #             "ISR": isr[idx].tolist(),
    #             "SAR": sar[idx].tolist()
    #         }
            
    #         track_store.add_target(target_name=target, values=values)
    #         json_path = json_folder / f"{track_name}.json.gz"
    #         gzip.open(json_path, "w").write(track_store.json.encode('utf-8'))


    if world_size > 1:
        distributed.barrier()