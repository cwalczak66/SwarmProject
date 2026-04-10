# MaxCal-Derived Swarm Control

This project extends recent research on a theoretical decentralized framework called Maximum Caliber (MaxCal), which is used to design swarm behaviors from high-level goals while robots rely only on local information. Our approach derives the local control rule of each robot directly from the principle of MaxCal, which generalises the maximum-entropy principle from static state representations to trajectories and sequential processes. By imposing two macroscopic constraints and observing the coverage age on every cell of a discretised arena and an information age on every robot, we can optimize the MaxCal equation and obtain a stochastic transition kernel that the robots execute locally resulting in an oscillating behavior between the two tasks. With only the coverage constraint active, we analytically recover the predicted coverage of the arena and therefore the expected mixing rate of the underlying Markov chain. We aim to visually show that when two opposing behaviors, such as clustering and spreading, create a natural tradeoff, MaxCal can derive local behavior rules that achieve an optimal balance between them.

## How to run the code:
Here we can put information on what to run, 
`pip install -r requirements.txt`
