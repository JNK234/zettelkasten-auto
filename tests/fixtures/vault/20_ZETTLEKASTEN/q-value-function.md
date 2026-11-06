
Created: 05-08-2025 01:05

Status: #mature  

Tags: [[value-function]] [[reward]]

# Q-Value Function

Q-Value Function is Value function with action included in calculating the expected cumulative reward. 

**Q-Value (State-Action Value Function), $Q^\pi_h(s,a)$**: This is the expected return if the agent starts from state $s$ at step $h$, takes a specific action $a$, and then subsequently follows policy $\pi$ for all future steps. $$Q^\pi_h(s,a) = E^\pi \left[ \sum_{h'=h}^H r_{h'}(S_{h'}, A_{h'}) \mid S_h = s, A_h = a \right]$$
**Relationship between V-value and Q-value**: The V-value of a state $s$ under policy $\pi$ can be expressed as the expected Q-value of taking an action from $s$ according to $\pi$ i.e. 
$$V^\pi_h(s) = E_{A_h \sim \pi_h(\cdot|s)} [Q^\pi_h(s,A_h)]$$ Or, more specifically, summing over actions if the policy is stochastic: $$V^\pi_h(s) = \sum_{a \in A} \pi_h(a|s) Q^\pi_h(s,a)$$


# References


