"""
This is the user script of the causal deterministic jump-back reward inference model on the proposed 1D state space.

The model first takes the waveform to the spectral domain and then feeds them into one of the pre-trained BeatNet models to obtain beat/downbeat activations.
Finally, the activations are used in a jump-reward inference model to infer beats, downbeats, tempo, and meter. 

system input: Raw audio waveform 

System output: a vector with beats, downbeats, local tempo, and local meter columns, respectively.  shape (num_beats, 4).


References: 

M. Heydari, M. McCallum, A. Ehmann and Z. Duan, "A NOVEL 1D STATE SPACE FOR EFFICIENT MUSIC RHYTHMIC ANALYSIS", In Proc. IEEE Int. Conf. Acoust. Speech
Signal Process. (ICASSP), 2022. #(Submitted)

M.  Heydari,  F.  Cwitkowitz,  and  Z.  Duan,    “BeatNet:CRNN and particle filtering for online joint beat down-beat and meter tracking,” inProc. of the 22th Intl. 
Conf.on Music Information Retrieval (ISMIR), 2021.

"""
import os
import torch
import numpy as np
from Inference_1D import deterministic_1D
from log_spect import LOG_SPECT
import librosa
import sys
from model import BDA

class joint_inference:
    def __init__(self, model, plot=False):
        self.sample_rate = 22050
        log_spec_sample_rate = self.sample_rate
        log_spec_hop_length = int(20 * 0.001 * log_spec_sample_rate)  
        log_spec_win_length = int(64 * 0.001 * log_spec_sample_rate) 
        self.proc = LOG_SPECT(sample_rate=log_spec_sample_rate, win_length=log_spec_win_length,
                             hop_size=log_spec_hop_length, n_bands=[24])
        script_dir = os.path.dirname(__file__)
        #assiging a BeatNet CRNN instance to extract joint beat and downbeat activations
        self.model = BDA(272, 150, 2, 'cpu')   #Beat Downbeat Activation estimator
        #loading the pre-trained BeatNet CRNN weigths
        if model == 1:  # GTZAN out trained model
            self.model.load_state_dict(torch.load(os.path.join(script_dir, 'models/model_1_weights.pt')), strict=False)
        elif model == 2:  # Ballroom out trained model
            self.model.load_state_dict(torch.load(os.path.join(script_dir, 'models/model_2_weights.pt')), strict=False)
        elif model == 3:  # Rock_corpus out trained model
            self.model.load_state_dict(torch.load(os.path.join(script_dir, 'models/model_3_weights.pt')), strict=False)
        else:
            raise RuntimeError(f'Failed to open the trained model: {model}')
        self.model.eval()
        self.estimator = deterministic_1D(beats_per_bar=[], fps=50, plot=plot)

    def process(self, audio_path):
        with torch.no_grad():
            if isinstance(audio_path, str):
            	audio, _ = librosa.load(audio_path, sr=self.sample_rate)  # reading the data
            else:
            	audio = audio_path
            feats = self.proc.process_audio(audio).T
            feats = torch.from_numpy(feats)
            feats = feats.unsqueeze(0)
            preds = self.model(feats)[0]  # extracting the activations by passing the feature through the NN
            preds = self.model.final_pred(preds)
            preds = preds.detach().numpy()
            preds = np.transpose(preds[:2, :])
            output = self.estimator.process(preds)
        return output

# Usage:
# estimator = joint_inference(1,plot=True)
# output = estimator.process("C:/datasets/testdata/123.mp3")
from BeatNet import BeatNet