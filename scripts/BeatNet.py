
# this is the batch test module using a pytorch batchloader instance to load the data and obtain the performance over the large datasets
import os

import torch
import numpy as np
from madmom.features import DBNDownBeatTrackingProcessor
from deterministic_1D import deterministic_1D
from particle_filtering_1D_2 import particle_filter_1D
from particle_filtering_cascade2 import particle_filter_cascade2
# import timeit
from log_spect import LOG_SPECT
import librosa


class BeatNet:
    def __init__(self, model, inference_model):
        self.sample_rate = 22050
        log_spec_sample_rate = self.sample_rate
        log_spec_hop_length = int(20 * 0.001 * log_spec_sample_rate)  # = 441
        log_spec_win_length = int(64 * 0.001 * log_spec_sample_rate)  # = 441
        self.proc = LOG_SPECT(sample_rate=log_spec_sample_rate, win_length=log_spec_win_length,
                             hop_size=log_spec_hop_length, n_bands=[24])

        if model == 1:  # GTZAN out trained model
            self.model = torch.load('models/model-1.pt', map_location=torch.device('cpu'))  #
        elif model == 2:  # Ballroom out trained model
            self.model = torch.load('models/model-2.pt', map_location=torch.device('cpu'))  #
        elif model == 3:  # Rock_corpus out trained model
            self.model = torch.load('models/model-3.pt', map_location=torch.device('cpu'))  #
        else:
            raise RuntimeError(f'Failed to open the trained model: {model}')
        self.inference_model = inference_model
        if self.inference_model == "DT":
            self.estimator = deterministic_1D(beats_per_bar=[], fps=50, plot=True)
        elif self.inference_model == "PF":                 # Particle Filter instance
            self.estimator = particle_filter_1D(beats_per_bar=[], fps=50, plot=True)
        elif self.inference_model == "PF2":                 # Particle Filter instance
            self.estimator = particle_filter_cascade2(beats_per_bar=[], fps=50, plot=False)
        elif self.inference_model == "DBN":                # HMM instance
            self.estimator = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=50)
        else:
            raise RuntimeError('inference_model can be either "PF", "DBN" or "DT"')

    def process(self, audio_path):
        # start = timeit.default_timer()
        with torch.no_grad():
            if isinstance(audio_path, str):
                audio, _ = librosa.load(audio_path, sr=self.sample_rate)  # reading file
            else:
                audio = audio_path
            feats = self.proc.process_audio(audio).T
            feats = torch.from_numpy(feats)
            feats = feats.unsqueeze(0)
            preds = self.model(feats)[0]
            preds = self.model.final_pred(preds)
            preds = preds.detach().numpy()
            preds = np.transpose(preds[:2, :])

            if self.inference_model == "DT" or self.inference_model == "PF" or self.inference_model == "PF2":   # Online _ causal
                data = self.estimator.process(preds)
            elif self.inference_model == "DBN":    # offline _ none-causal
                data = self.estimator(preds)

            # stop = timeit.default_timer()
        downs = data[:, 0][data[:, 1] == 1]
        beats = data[:, 0]
        return beats, downs
        # return data


beatnet = BeatNet(1,'DT')
# count = '00003'
# # beatnet2 = BeatNet(3,'PF2')
beats, downs = beatnet.process("C:\datasets\GTZAN/audio/blues/blues.00001.wav")
# y, sr = librosa.load(f"C:\datasets\GTZAN/audio/reggae/reggae.{count}.wav", sr=22050)
#
# beats, downs,x = beatnet.process(y)
# # beats, downs,x = beatnet.process("C:/datasets/GTZAN/123.mp3")111111
# # beats2, downs2 = beatnet2.process("C:/datasets/GTZAN/123.mp3")
# y, sr = librosa.load(f"C:\datasets\GTZAN/audio/reggae/reggae.{count}.wav", sr=22050)
# tempo, beats3 = librosa.beat.beat_track(y=y, sr=sr)
# y_beats = librosa.clicks(frames=beats*50, sr=sr, hop_length=441, length=len(y))
# y_downs = librosa.clicks(frames=downs*50, sr=sr, hop_length=441, click_freq=500.0, length=len(y))
# y = y+y_beats+y_downs
#
# from matplotlib import pyplot as plt
# import soundfile as sf
# sf.write(f"C:/datasets/GTZAN/reggae.{count}-out.wav", y, sr, 'PCM_24')
print("done")

# from madmom.features.downbeats import RNNDownBeatProcessor as Downn
#
# proc = Downn()
# preds = proc("C:/datasets/GTZAN/test-imagine.mp3")