# pong_experiments
Experimenting with different inputs to policy gradient learning for Pong

Code adapted from [Andrej Karpathy](https://github.com/gameofdimension/policy-gradient-pong)

-----------------

Summary of findings from these first few experiments:
- Initialization is extremely important. Running the same experiment with different random initial weight parameters (even with the same number of hidden units) performs drastically differently in the first 1000 episodes, depending on whether or not we got a favorable initialization. In fact, there are cases where one initialization results in getting stuck in a terrible place (winning no games at all), and another initialization of the same experiment performs well in the beginning and seems to get better. (See rel_bonus_70* files for example)


 — Note: all initializations were “Xavier” initializations - More hidden neurons seem to lead to increased performance (not completely sure on this, but it seems to be the case from the few experiments I ran, at least up to 100) - Using absolute positions of objects with no memory performs *worse* (at least with 10, 50, 100 hidden neurons) than the difference of frames approach given by Karpathy, at least in the first 1000 episodes. 
 
 
 - Capped pseudo-rewards: I tried giving the agent a small capped reward for keeping the ball within the top and bottom position of its paddle when the ball was approaching, with a higher reward when the ball is closer to the paddle. I split these rewards into 4 zones.  
 — _What seems to work and helped convergence_: Giving the agent a reward in the closest zone (which meant the agent was pretty much guaranteed to return the ball). BUT, if the agent had a bad initialization of weights, it never even got to the point of collecting this pseudo-reward. I'm leaving the promising initializations running for now, to see if they ultimately converge to good play.  
 — _What doesn’t work_: Giving the agent a very small reward for lining up the paddle with the ball when the ball is far away. I thought that this may encourage the agent to keep the ball lined up with the paddle as the ball got closer, but this is not the case. It would just repeat the behavior that gave it its very small reward. 
 
 - Next steps:
 
— Going to look at [Curriculum Learning](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf) (hopefully to get some insight to improve training stability)  
— Going to try to think about how pseudo-rewards or intrinsic rewards should be done. Will probably read through the [Curiosity driven exploration.](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w5/papers/Pathak_Curiosity-Driven_Exploration_by_CVPR_2017_paper.pdf)



