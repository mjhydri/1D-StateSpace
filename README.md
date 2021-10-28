# 1D-StateSpace

This repository contains the source code and demo videos of a joint music rhythmic analyzer system using the 1D state space and jump reward technique proposed in ICASSP-2022. This implementation includes music beat, downbeat, tempo, and meter tracking jointly and in a causal fashion. 

The model first takes the waveform to the spectral domain and then feeds them into one of the pre-trained BeatNet models to obtain beat/downbeat activations.
Finally, the activations are used in a jump-reward inference model to infer beats, downbeats, tempo, and meter. 

** Note: The source code and the user package will be uploaded on January 21, 2022.

System Input:
-------------
Raw audio waveform 

System Output:
--------------
A vector including beats, downbeats, local tempo, and local meter columns, respectively.  shape (num_beats, 4).

References:
-----------
M. Heydari, M. McCallum, A. Ehmann and Z. Duan, "A NOVEL 1D STATE SPACE FOR EFFICIENT MUSIC RHYTHMIC ANALYSIS", In Proc. IEEE Int. Conf. Acoust. Speech
Signal Process. (ICASSP), 2022. #(Submitted)

M.  Heydari,  F.  Cwitkowitz,  and  Z.  Duan,    “BeatNet:CRNN and particle filtering for online joint beat down-beat and meter tracking,” in Proc. of the 22th Intl. 
Conf.on Music Information Retrieval (ISMIR), 2021.

Installation Command:
---------------------
pip install git+https://github.com/mjhydri/1D-StateSpace

Usage Example:
--------------
estimator = joint inference(1) 

output =estimator.process("music file directory")

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
1- As demo videos suggest, for some genres such as pop and country, the process is relatively more straightforward due to the rich percussive content, solid attacks, and simpler rhythmic structures. However, it is more challenging for genres with poor percussive profile, longer attack times, and more complex rhythmic structures such as Classical music. 

2- Since both neural networks and inference models are designed for online/real-time applications, the system is causal, and future data is not accessible. It makes the belief state and the actual jump weight weak initially and become stronger over time. 

3- Given longer listening time is required to infer higher hierarchies, i.e., downbeat and meter, within the first few seconds, downbeat results are less confident than lower hierarchies, i.e., beat and tempo.   


