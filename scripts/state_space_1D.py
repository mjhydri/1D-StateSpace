import numpy as np


class state_space_1D:
    '''
        This class creates 1D state spaces for different music rhythmic hierarchies.
        
        M. Heydari, M. McCallum, A. Ehmann and Z. Duan, "A NOVEL 1D STATE SPACE FOR EFFICIENT MUSIC RHYTHMIC ANALYSIS", In Proc. IEEE Int. Conf. Acoust. Speech
        Signal Process. (ICASSP), 2022. #(Submitted)
    '''

    def __init__(self, min_interval, max_interval):
        self.min_interval = int(np.round(min_interval))
        self.max_interval = int(np.round(max_interval))
        self.first_states = np.array([0])
        self.last_states = np.array([self.max_interval-1])
        self.num_states = self.max_interval
        self.state_intervals = np.array([max_interval] * max_interval)
        self.state_positions = np.linspace(0, 1,  self.num_states, endpoint=False)


class beat_state_space_1D(state_space_1D):

    def __init__(self, alpha=0.01, tempo=None, fps=None, min_interval=None, max_interval=None):
        super().__init__(min_interval, max_interval)
        self.jump_weights = np.concatenate((np.zeros(self.min_interval), np.array([alpha] * (self.max_interval - self.min_interval)), ))
        if tempo and fps:
            self.jump_weights[round(60. * fps / tempo) - self.min_interval] = 1 - alpha


class downbeat_state_space_1D(state_space_1D):

    def __init__(self, alpha=0.1, meter=None, min_beats_per_bar=None, max_beats_per_bar=None):
        super().__init__(min_beats_per_bar, max_beats_per_bar)
        self.jump_weights = np.concatenate((np.zeros(self.min_interval-1), np.array([alpha] * (self.max_interval - self.min_interval+1)), ))
        if meter:
            self.jump_weights[meter[0] - self.min_interval+1] = 1 - alpha
        pass

