"""
This is the user script of the causal deterministic jump-back reward inference model on the proposed 1D state space.

The model first takes the waveform to the spectral domain and then feeds them into one of the pre-trained BeatNet models to obtain beat/downbeat activations.
Finally, the activations are used in a jump-reward inference model to infer beats, downbeats, tempo, and meter. 

system input: Raw audio waveform 

System output: a vector with beats, downbeats, local tempo, and local meter columns, respectively.  shape (num_beats, 4).


References: 

M. Heydari, M. McCallum, A. Ehmann and Z. Duan, "A NOVEL 1D STATE SPACE FOR EFFICIENT MUSIC RHYTHMIC ANALYSIS", In Proc. IEEE Int. Conf. Acoust. Speech
Signal Process. (ICASSP), 2022. #(Submitted)

M.  Heydari,  F.  Cwitkowitz,  and  Z.  Duan,    “BeatNet:CRNN and particle filtering for online joint beat down-beat and meter tracking,” in Proc. of the 22th Intl.
Conf.on Music Information Retrieval (ISMIR), 2021.

"""
from Inference_1D import inference_1D
from BeatNet.BeatNet import BeatNet


class joint_inference():
    def __init__(self, model, plot=False):
        self.activation_estimator = BeatNet(model)
        self.estimator = inference_1D(beats_per_bar=[], fps=50, plot=plot)

    def process(self, audio_path):
        preds = self.activation_estimator.activation_extractor(
            audio_path)  # extracting the activations using the BeatNet nueral network
        output = self.estimator.process(
            preds)  # infering online joing beat, downbeat, tempo and meter using the 1D state space and jump back reward technique
        return output


# Usage example:
# estimator = joint_inference(1, plot=False)
# output = estimator.process("C:/datasets/testdata/123.mp3")
# print(output)
