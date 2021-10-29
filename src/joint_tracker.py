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
from joint-tracker import deterministic_1D
import sys
from BeatNet.BeatNet import BeatNet

# instance = BeatNet(1)
# Output = instance.process("C:/datasets/testdata/123.mp3", 'PF', plot=True)
# print("hi")

class joint_inference:
    def __init__(self, model, plot=False):
        activation_estimator = BeatNet(model)
        self.plot = plot
        self.inferer = deterministic_1D(beats_per_bar=[], fps=50, plot=plot)

    def process(self, audio_path):
            preds =  BeatNet.activation_extractor(audio_path)
            output = self.inferer.process(preds)
        return output
#
# # Usage:
# # estimator = joint_inference(1,plot=True)
# # output = estimator.process("C:/datasets/testdata/123.mp3")
# from BeatNet import BeatNet
