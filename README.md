# DeepReinforcementLearning

It was originally inspired by Stanford's Reinforcement Learning course (CS234). The assignments were a part of my CS735: Deep Reinforcement Learning Course at the University of South Carolina. A brief description and objectives of the assignments are included.

# Assignment 1: 
# Assignment 2:
1.1 Optimum reward in Test ENV
1.2 Tabular Q-learning
1.3 Q-Learning with Linear and Non-linear Function approximation ( a variant of DQN implementation [Original Paper])
1.4 DQN on Atari (Pong-v0)
Since Pong-v0 is quite backdated, I needed to be extra careful while setting the environment.
The following helped:

!pip install gym==0.22.0
!pip install "gym[atari, accept-rom-license]"
!pip install pyglet
!pip uninstall ale-py
!pip install ale-py

I tried to run DQN on Atari (Pong_v0) in colab, but that attempt was met with unusual errors, and then I had to switch to cluster gpu. Training time approximately 1hr15min for 500k steps.


<img width="372" alt="a2git" src="https://user-images.githubusercontent.com/47276166/229419331-6e20a96b-99c5-4304-8e3d-5dffb4a4b1e7.PNG">
