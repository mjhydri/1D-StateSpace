# 1D-StateSpace

This repository contains the source code and demo videos of a joint music rhythmic analyzer system using the 1D state space and jump reward technique proposed in ICASSP-2022. This implementation includes music beat, downbeat, tempo, and meter tracking jointly and in a causal fashion. 

The model first takes the waveform to the spectral domain and then feeds them into one of the pre-trained BeatNet models to obtain beat/downbeat activations.
Finally, the activations are used in a jump-reward inference model to infer beats, downbeats, tempo, and meter. 

** Note: The source code and the user package will be uploaded soon.

System Input:
-------------
Raw audio waveform 

System Output:
--------------
A vector including beats, downbeats, local tempo, and local meter columns, respectively and with the following shape: numpy_array(num_beats, 4).

References:
-----------
M. Heydari, M. McCallum, A. Ehmann and Z. Duan, "A Novel 1D State Space for Efficient Music Rhythmic Analysis", In Proc. IEEE Int. Conf. Acoust. Speech
Signal Process. (ICASSP), 2022. #(Submitted)

M.  Heydari,  F.  Cwitkowitz,  and  Z.  Duan,    “BeatNet:CRNN and particle filtering for online joint beat down-beat and meter tracking,” in Proc. of the 22th Intl. 
Conf.on Music Information Retrieval (ISMIR), 2021.

M. Heydari and Z. Duan, “Don’t Look Back: An online beat  tracking  method  using  RNN  and  enhanced  particle filtering,”  in Proc. IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP), 2021.

Installation Command:
---------------------
pip install git+https://github.com/mjhydri/1D-StateSpace

Usage Example:
--------------
estimator = joint inference(1, plot=True) 

output = estimator.process("music file directory")

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


