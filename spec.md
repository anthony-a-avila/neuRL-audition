We study whether ideas about how the brain works, especially the prefrontal cortex, can help reinforcement learning agents handle new situations better. 

This semester we're building test environments that measure that ability more carefully than the ones people use now, then trying brain-inspired designs against them.
What we're asking you to do

You'll write a few dozen lines of Python. There are no neural networks involved, so numpy and matplotlib are all you need. Everything runs on any laptop in a few seconds.

You do not need prior RL experience. If this is new to you, read chapters 1 and 6 of Sutton and Barto's Reinforcement Learning: An Introduction first. It's free at incompleteideas.net (it’s enough background for either option). 


Option 2. Schultz, Dayan and Montague (1997), "A Neural Substrate of Prediction and Reward" (Science).

This paper showed that dopamine neurons in the brain seem to compute the same error signal that TD learning uses. 

It's the reason our project is a reasonable thing to attempt at all.

The experiment: a cue appears, and a reward follows shortly after. 

Run a TD learner on this and watch its error signal. 

Three things should happen. 

Early on, the error spikes when the reward arrives. 

After learning, the spike moves to the cue instead. 

And if you then withhold the reward, the error goes negative at the moment the reward should have come.

Recreate the figure showing those three cases.

Please send us the following:

- A figure. Label the axes. Run it with a few different random seeds so you can show how much the result moves around.

- A writeup (one page at most). Tell us what you found, where your version came out different from the paper's and what you think caused that, what you tried that didn't work, and one thing you'd try next.

- We also want your code as a GitHub link or a zip file. Mihir will read it.

- A short video explanation of what you did. Do NOT read off of a script.

