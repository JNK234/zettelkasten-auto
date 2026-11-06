
Created: 05-08-2025 00:54

Status: #mature  

Tags: [[markov-property]] [[reward]] [[q-value-function]]

# Value Function

**Value functions** quantify the "goodness" of a state or a state-action pair under a particular policy. They represent the **expected cumulative reward** an agent will receive in the future. This can be represented using the following Return term. 

**Return ($G_t$)**: The **discounted sum of rewards** from a given time step $t$ up to the horizon $H$. This is a specific sequence of rewards that might be obtained. $$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{H-1} r_{t+H-1}$$
**Discount Factor ($\gamma$)**: A value between 0 and 1, applied to future rewards. It determines the relative importance of immediate rewards versus future rewards.
    - A **large discount factor (closer to 1)** means that **long-term rewards are much more influential** than short-term rewards. If $\gamma = 1$, all future rewards are valued equally.
    - A **small discount factor (closer to 0)** means the agent is more "myopic" and primarily values **immediate rewards**, with distant future rewards having very little influence.

But the return is not feasible to compute. So we formulate Value Functions that approximate the return or expected return. 

**V-Value (State-Value Function), $V^\pi_h(s)$**: This is the **expected return** if the agent starts from state $s$ at step $h$ and subsequently follows policy $\pi$ for all future steps until the end of the episode. This includes the Markov Property.
$$V^\pi_h(s) = E^\pi \left[ \sum_{h'=h}^H r_{h'}(S_{h'}, A_{h'}) \mid S_h = s \right]$$The expectation is taken over the future state-action sequence, which is stochastic due to both the environment's transitions and the policy's action choices (if stochastic).

# References