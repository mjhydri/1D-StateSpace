# A Novel 1D State Space for Efficient Music Rhythmic Analysis 

An implementation of the probablistic jump reward semi_Markov inference model for music rhythmic analysis leveraging the proposed 1D state space. 

[![PyPI](https://img.shields.io/pypi/v/jump-reward-inference.svg)](https://pypi.org/project/jump-reward-inference/)
[![CC BY 4.0][cc-by-shield]][cc-by]
[![Downloads](https://pepy.tech/badge/jump-reward-inference)](https://pepy.tech/project/jump-reward-inference)

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-novel-1d-state-space-for-efficient-music/online-beat-tracking-on-gtzan)](https://paperswithcode.com/sota/online-beat-tracking-on-gtzan?p=a-novel-1d-state-space-for-efficient-music)

This repository contains the source code and demo videos of a joint music rhythmic analyzer system using the 1D state space and jump reward technique proposed in ICASSP-2022. This implementation includes music beat, downbeat, tempo, and meter tracking jointly and in a causal fashion. 

*[arXiv 2111.00704](https://arxiv.org/abs/2111.00704)*

The model first takes the waveform to the spectral domain and then feeds them into one of the pre-trained BeatNet models to obtain beat/downbeat activations.
Finally, the activations are used in a jump-reward inference model to infer beats, downbeats, tempo, and meter. 


System Input:
-------------
Raw audio waveform 

System Output:
--------------
A vector including beats, downbeats, local tempo, and local meter columns, respectively and with the following shape: numpy_array(num_beats, 4).

Installation Command:
---------------------
Approach #1: Installing binaries from the pypi website:
```
pip install jump-reward-inference
```

Approach #2: Installing directly from the Git repository:
```
pip install git+https://github.com/mjhydri/1D-StateSpace
```
* Note that by using either of the approaches all dependencies and required packages get installed automatically except Pyaudio that connot be installed that way. Pyaudio is a python binding for Portaudio to handle audio streaming. 
 
If Pyaudio is not installed in your machine, download an appropriate version for your machine from *[here](https://www.lfd.uci.edu/~gohlke/pythonlibs/)*. Then, navigate to the file location through commandline and use the following command to install the wheel file locally:
```
pip install <Pyaudio_file_name.whl>   
```
Usage Example:
--------------
```
estimator = joint_inference(1, plot=True) 

output = estimator.process("music file directory")
```
Video Demos:
------------

This section demonstrates the system performance for several music genres. Each demo comprises four plots that are described as follows:  

* The first plot: 1D state space for music beat and tempo tracking. Each bar represents the posterior probability of the corresponding state at each time frame.
* The second plot: The jump-back reward vector for the corresponding beat states. 
* The third plot:1D state space for music downbeat and meter tracking.
* The fourth plot: The jump-back reward vector for the corresponding downbeat states. 


1: Music Genre: Pop
  
[![Easy song](https://img.youtube.com/vi/YXGzvLe6bSQ/0.jpg)](https://youtu.be/YXGzvLe6bSQ)
  


2: Music Genre: Country
  
[![Easy song](https://img.youtube.com/vi/-9Lwirn6YAI/0.jpg)](https://youtu.be/-9Lwirn6YAI)

  

3: Music Genre: Reggae
  
[![Easy song](https://img.youtube.com/vi/VnDBmXWemPI/0.jpg)](https://youtu.be/VnDBmXWemPI)



4: Music Genre: Blues
  
[![Easy song](https://img.youtube.com/vi/CcUe3P0Y9BM/0.jpg)](https://youtu.be/CcUe3P0Y9BM)
  


5: Music Genre: Classical
  
  [![Easy song](https://img.youtube.com/vi/fl2ErbGrbyo/0.jpg)](https://youtu.be/fl2ErbGrbyo)
  

Demos Discussion:
-----------------
1- As demo videos suggest, the system infers multiple music rhythmic parameters, including music beat, downbeat, tempo and meter jointly and in an online fashion using very compact 1D state spaces and jump back reward technique. The system works suitably for different music genres. However, the process is relatively more straightforward for some genres such as pop and country due to the rich percussive content, solid attacks, and simpler rhythmic structures. In contrast, it is more challenging for genres with poor percussive profile, longer attack times, and more complex rhythmic structures such as classical music. 

2- Since both neural networks and inference models are designed for online/real-time applications, the causalilty constrains are applied and future data is not accessible. It makes the jumpback weigths weaker initially and become stronger over time. 

3- Given longer listening time is required to infer higher hierarchies, i.e., downbeat and meter, within the very early few seconds, downbeat results are less confident than lower hierarchies, i.e., beat and tempo, however, they get accurate after observing a bar period.     

Acknowledgement
---------------
Many thanks to the Pandora/SiriusXM Inc. research team for making it legal to publish the project's source code. To load the raw audio and input features extraction [Librosa](https://github.com/librosa/librosa) and [Madmom](https://github.com/CPJKU/madmom) libraries are ustilzed respectively. Many thanks for their great jobs. This work has been partially supported by the National Science Foundation grant 1846184.

References:
-----------
M. Heydari, M. McCallum, A. Ehmann and Z. Duan, "A Novel 1D State Space for Efficient Music Rhythmic Analysis", In Proc. IEEE Int. Conf. Acoust. Speech
Signal Process. (ICASSP), 2022. #(Recently Submitted)

M.  Heydari,  F.  Cwitkowitz,  and  Z.  Duan,    “BeatNet:CRNN and particle filtering for online joint beat down-beat and meter tracking,” in Proc. of the 22th Intl. 
Conf.on Music Information Retrieval (ISMIR), 2021.

M. Heydari and Z. Duan, “Don’t Look Back: An online beat  tracking  method  using  RNN  and  enhanced  particle filtering,”  in Proc. IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP), 2021.
