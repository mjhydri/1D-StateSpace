import numpy as np
import matplotlib.pyplot as plt
from state_space_1D import beat_state_space_1D, downbeat_state_space_1D
import joint_tracker


class BDObservationModel:
    """
    Observation model for beat and downbeat tracking with particle filtering.
    
    Based on the first character of this parameter, each (down-)beat period gets split into (down-)beat states
    "B" stands for border model which classifies 1/(observation lambda) fraction of states as downbeat states and
    the rest as the beat states (if it is used for downbeat tracking state space) or the same fraction of states
    as beat states and the rest as the none beat states (if it is used for beat tracking state space).
    "N" model assigns a constant number of the beginning states as downbeat states and the rest as beat states
     or beginning states as beat and the rest as none-beat states
    "G" model is a smooth Gaussian transition (soft border) between downbeat/beat or beat/none-beat states


    Parameters
    ----------
        state_space : :class:`BarStateSpace` instance
        BarStateSpace instance.
        observation_lambda : str
        
    References:
    ----------
        
        M. Heydari and Z. Duan, “Don’t look back: An online
        beat tracking method using RNN and enhanced particle filtering,” in In Proc. IEEE Int. Conf. Acoust. Speech
        Signal Process. (ICASSP), 2021.


    """

    def __init__(self, state_space, observation_lambda):

        if observation_lambda[0] == 'B':   # Observation model for the rigid boundaries
            observation_lambda = int(observation_lambda[1:])
            # compute observation pointers
            # always point to the non-beat densities
            pointers = np.zeros(state_space.num_states, dtype=np.uint32)
            # unless they are in the beat range of the state space
            border = 1. / observation_lambda
            pointers[state_space.state_positions % 1 < border] = 1
            # the downbeat (i.e. the first beat range) points to density column 2
            pointers[state_space.state_positions < border] = 2
            # instantiate a ObservationModel with the pointers
            self.pointers = pointers

        elif observation_lambda[0] == 'N':  # Observation model for the fixed number of states as beats boundaries
            observation_lambda = int(observation_lambda[1:])
            # computing observation pointers
            # always point to the non-beat densities
            pointers = np.zeros(state_space.num_states, dtype=np.uint32)
            # unless they are in the beat range of the state space
            for i in range(observation_lambda):
                border = np.asarray(state_space.first_states) + i
                pointers[border[1:]] = 1
                # the downbeat (i.e. the first beat range) points to density column 2
                pointers[border[0]] = 2
                # instantiate a ObservationModel with the pointers
            self.pointers = pointers

        elif observation_lambda[0] == 'G':  # Observation model for gaussian boundaries
            observation_lambda = float(observation_lambda[1:])
            pointers = np.zeros((state_space.num_beats + 1, state_space.num_states))
            for i in range(state_space.num_beats + 1):
                pointers[i] = gaussian(state_space.state_positions, i, observation_lambda)
            pointers[0] = pointers[0] + pointers[-1]
            pointers[1] = np.sum(pointers[1:-1], axis=0)
            pointers = pointers[:2]
            self.pointers = pointers


def gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu) / sig, 2.) / 2)  # /(np.sqrt(2.*np.pi)*sig)

# Beat state space observations 
def densities(observations, observation_model, state_model):
    new_obs = np.zeros(state_model.num_states, float)
    if len(np.shape(observation_model.pointers)) != 2:  # For Border line and fixed number observation distribution
        new_obs[np.argwhere(
            observation_model.pointers == 2)] = observations
        new_obs[np.argwhere(
            observation_model.pointers == 0)] = 0.03
    elif len(np.shape(observation_model.pointers)) == 2:  # For Gaussian observation distribution
        new_obs = observation_model.pointers[
                      0] * observations
        new_obs[new_obs < 0.005] = 0.03
    return new_obs


# Downbeat state space observations 
def densities2(observations, observation_model, state_model):
    new_obs = np.zeros(state_model.num_states, float)
    if len(np.shape(observation_model.pointers)) != 2:  # For Border line and fixed number observation distribution
        new_obs[np.argwhere(
            observation_model.pointers == 2)] = observations[1]
        new_obs[np.argwhere(
            observation_model.pointers == 0)] = 0.00002
    elif len(np.shape(observation_model.pointers)) == 2:  # For Gaussian observation distribution
        new_obs = observation_model.pointers[0] * observations
        new_obs[new_obs < 0.005] = 0.03
    # return the observations
    return new_obs


# def densities_down(observations, beats_per_bar):
#     new_obs = np.zeros(beats_per_bar, float)
#     new_obs[0] = observations[1]  # downbeat
#     new_obs[1:] = observations[0]  # beat
#     return new_obs


class deterministic_1D: 
    '''
        Running the jump-reward inference for the 1D state spaces over the given activation functions to infer music rhythmic information jointly.

        Parameters
        ----------
        activations : numpy array, shape (num_frames, 2)
            Activation function with probabilities corresponding to beats
            and downbeats given in the first and second column, respectively.

        Returns
        ----------
        beats, downbeats, local tempo, local meter : numpy array, shape (num_beats, 4)
            "1" at the second column indicates the downbeats.  

        References:
        ----------    

    '''
    MIN_BPM = 55.
    MAX_BPM = 200.  # 215.
    NUM_TEMPI = 300
    LAMBDA_B = 0.01  # beat transition lambda
    Lambda_D = 0.01  # downbeat transition lambda
    OBSERVATION_LAMBDA = "B56"
    fps = 50
    T = 1 / fps
    MIN_BEAT_PER_BAR = 1
    MAX_BEAT_PER_BAR = 4
    OFFSET = 4  # The point of time after which the inference model starts to work. Can be zero!
    IG_THRESHOLD = 0.4  # Information Gate threshold

    def __init__(self, beats_per_bar=[], min_bpm=MIN_BPM, max_bpm=MAX_BPM, offset=OFFSET,
                 min_bpb=MIN_BEAT_PER_BAR, max_bpb=MAX_BEAT_PER_BAR, ig_threshold=IG_THRESHOLD,
                 lambda_b=LAMBDA_B, lambda_d=Lambda_D, observation_lambda=OBSERVATION_LAMBDA,
                 fps=None, plot=False, **kwargs):
        self.beats_per_bar = beats_per_bar
        self.fps = fps
        self.observation_lambda = observation_lambda
        self.offset = offset
        self.ig_threshold = ig_threshold
        self.plot = plot
        beats_per_bar = np.array(beats_per_bar, ndmin=1)

        # convert timing information to construct a beat state space
        min_interval = round(60. * fps / max_bpm)
        max_interval = round(60. * fps / min_bpm)

        # State spaces and observation models initialization
        self.st = beat_state_space_1D(alpha=lambda_b, tempo=None, fps=None, min_interval=min_interval,
                                      max_interval=max_interval, )  # beat tracking state space
        self.st2 = downbeat_state_space_1D(alpha=lambda_d, meter=self.beats_per_bar, min_beats_per_bar=min_bpb,
                                           max_beats_per_bar=max_bpb)  # downbeat tracking state space
        self.om = BDObservationModel(self.st, observation_lambda)  # beat tracking observation model
        self.om2 = BDObservationModel(self.st2, "B60")  # downbeat tracking observation model
        pass

    def process(self, activations, **kwargs):
        T = 1 / self.fps
        font = {'color': 'green', 'weight': 'normal', 'size': 12}
        counter = 0
        if self.plot:
            fig = plt.figure(figsize=(1800 / 96, 900 / 96), dpi=96)
            subplot1 = fig.add_subplot(411)
            subplot2 = fig.add_subplot(412)
            subplot3 = fig.add_subplot(413)
            subplot4 = fig.add_subplot(414)
            subplot1.title.set_text('Beat tracking inference diagram')
            subplot2.title.set_text('Beat states jumping back weigths')
            subplot3.title.set_text('Downbeat tracking inference diagram')
            subplot4.title.set_text('Downbeat states jumping back weigths')
            fig.tight_layout()

        # output vector initialization for beat, downbeat, tempo and meter
        output = np.zeros((1, 4), dtype=float)
        # Beat and downbeat belief state initialization
        beat_distribution = np.ones(self.st.num_states) * 0.8
        beat_distribution[5] = 1 # this is just a flag to show the initial transitions
        down_distribution = np.ones(self.st2.num_states) * 0.8
        # local tempo and meter initialization
        local_tempo = 0
        meter = 0

        activations = activations[int(self.offset / T):]
        both_activations = activations.copy()
        activations = np.max(activations, axis=1)
        activations[activations < self.ig_threshold] = 0.03

        for i in range(len(activations)):  # loop through all frames to infer beats/downbeats
            counter += 1
            #   beat detection

            #   beat transition   (motion)
            local_beat = ''
            if np.max(self.st.jump_weights) > 1:
                self.st.jump_weights = 0.7 * self.st.jump_weights / np.max(self.st.jump_weights)
            b_weight = self.st.jump_weights.copy()
            beat_jump_rewards1 = -beat_distribution * b_weight  # calculating the transition rewards
            b_weight[b_weight < 0.7] = 0  # Thresholding the jump backs
            beat_distribution1 = sum(beat_distribution * b_weight)  # jump back
            beat_distribution2 = np.roll((beat_distribution * (1 - b_weight)), 1)  # move forward
            beat_distribution2[0] += beat_distribution1
            beat_distribution = beat_distribution2

            #  Beat correction
            if activations[i] > self.ig_threshold:  # beat correction is done only when there is a meaningful activation
                obs = densities(activations[i], self.om, self.st)
                beat_distribution_old = beat_distribution.copy()
                beat_distribution = beat_distribution_old * obs
                beat_max = np.argmax(beat_distribution)
                beat_jump_rewards2 = beat_distribution - beat_distribution_old  # beat correction rewards calculation
                beat_jump_rewards = beat_jump_rewards2
                beat_jump_rewards[:self.st.min_interval - 1] = 0
                if np.max(-beat_jump_rewards) != 0:
                    beat_jump_rewards = -4 * beat_jump_rewards / np.max(-beat_jump_rewards)
                    self.st.jump_weights += beat_jump_rewards
                local_tempo = round(self.fps * 60 / (np.argmax(self.st.jump_weights) + 1))
            else:
                beat_jump_rewards1[:self.st.min_interval - 1] = 0
                self.st.jump_weights += 2 * beat_jump_rewards1
                self.st.jump_weights[:self.st.min_interval - 1] = 0
                beat_max = np.argmax(beat_distribution)

            #  downbeat detection
            if (beat_max < (
                    int(.07 / T)) + 1) and (counter * T + self.offset) - output[-1][
                0] > .45 * T * self.st.min_interval:  # here the local tempo (:np.argmax(self.st.jump_weights)+1): can be used as the criteria rather than the minimum tempo
                local_beat = 'NoooOOoooW!'

                #   downbeat transition   (motion)
                if np.max(self.st2.jump_weights) > 1:
                    self.st2.jump_weights = 0.2 * self.st2.jump_weights / np.max(self.st2.jump_weights)
                d_weight = self.st2.jump_weights.copy()
                down_jump_rewards1 = - down_distribution * d_weight
                d_weight[d_weight < 0.2] = 0
                down_distribution1 = sum(down_distribution * d_weight)  # jump back
                down_distribution2 = np.roll((down_distribution * (1 - d_weight)), 1)  # move forward
                down_distribution2[0] += down_distribution1
                down_distribution = down_distribution2

                #  Downbeat correction
                if both_activations[i][1] > 0.00002:
                    obs2 = densities2(both_activations[i], self.om2, self.st2)
                    down_distribution_old = down_distribution.copy()
                    down_distribution = down_distribution_old * obs2
                    down_max = np.argmax(down_distribution)
                    down_jump_rewards2 = down_distribution - down_distribution_old  # downbeat correction rewards calculation
                    down_jump_rewards = down_jump_rewards2
                    down_jump_rewards[:self.st2.max_interval - 1] = 0
                    if np.max(-down_jump_rewards) != 0:
                        down_jump_rewards = -0.3 * down_jump_rewards / np.max(-down_jump_rewards)
                        self.st2.jump_weights = self.st2.jump_weights + down_jump_rewards
                    meter = np.argmax(self.st2.jump_weights) + 1
                else:
                    down_jump_rewards1[:self.st2.min_interval - 1] = 0
                    self.st2.jump_weights += 2 * down_jump_rewards1
                    self.st2.jump_weights[:self.st2.min_interval - 1] = 0
                    down_max = np.argmax(down_distribution)

                #  Beat vs Downbeat distinguishment
                if down_max == self.st2.first_states[0]:
                    output = np.append(output, [[counter * T + self.offset, 1, local_tempo, meter]], axis=0)
                    last_detected = "Downbeat"
                else:
                    output = np.append(output, [[counter * T + self.offset, 2, local_tempo, meter]], axis=0)
                    last_detected = "Beat"
                if self.plot:
                    subplot3.cla()
                    subplot4.cla()
                    down_distribution = down_distribution / np.max(down_distribution)
                    subplot3.bar(np.arange(self.st2.num_states), down_distribution, color='maroon', width=0.4,
                                 alpha=0.2)
                    subplot3.bar(0, both_activations[i][1], color='green', width=0.4, alpha=0.3)
                    # subplot3.bar(np.arange(1, self.st2.num_states), both_activations[i][0], color='yellow', width=0.4, alpha=0.3)
                    subplot4.bar(np.arange(self.st2.num_states), self.st2.jump_weights, color='maroon', width=0.4,
                                 alpha=0.2)
                    subplot4.set_ylim([0, 1])
                    subplot3.title.set_text('Downbeat tracking 1D inference model')
                    subplot4.title.set_text(f'Downbeat states jumping back weigths')
                    subplot4.text(1, -0.26, f'The type of the last detected event: {last_detected}',
                                  horizontalalignment='right', verticalalignment='center', transform=subplot2.transAxes,
                                  fontdict=font)
                    subplot4.text(1, -1.63, f'Local time signature = {meter}/4 ', horizontalalignment='right',
                                  verticalalignment='center', transform=subplot2.transAxes, fontdict=font)
                    position2 = down_max
                    subplot3.axvline(x=position2)

            if self.plot:  # activates this when you want to plot the performance
                if counter % 1 == 0:  # Choosing how often to plot
                    print(counter)
                    subplot1.cla()
                    subplot2.cla()
                    beat_distribution = beat_distribution / np.max(beat_distribution)
                    subplot1.bar(np.arange(self.st.num_states), beat_distribution, color='maroon', width=0.4, alpha=0.2)
                    subplot1.bar(0, activations[i], color='green', width=0.4, alpha=0.3)
                    # subplot2.bar(np.arange(self.st.num_states), np.concatenate((np.zeros(self.st.min_interval),self.st.jump_weights)), color='maroon', width=0.4, alpha=0.2)
                    subplot2.bar(np.arange(self.st.num_states), self.st.jump_weights, color='maroon', width=0.4,
                                 alpha=0.2)
                    subplot2.set_ylim([0, 1])
                    subplot1.title.set_text('Beat tracking 1D inference model')
                    subplot2.title.set_text("Beat states jumping back weigths")
                    subplot1.text(1, 2.48, f'Beat moment: {local_beat} ', horizontalalignment='right',
                                  verticalalignment='top', transform=subplot2.transAxes, fontdict=font)
                    subplot2.text(1, 1.12, f'Local tempo: {local_tempo} (BPM)', horizontalalignment='right',
                                  verticalalignment='top', transform=subplot2.transAxes, fontdict=font)
                    position = beat_max
                    subplot1.axvline(x=position)
                    # plt.show()
                    plt.pause(0.05)
                    subplot1.clear()

        return output[1:]
