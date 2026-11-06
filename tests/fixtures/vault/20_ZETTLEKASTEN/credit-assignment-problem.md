
Created: 12-07-2025 22:46

Status: #mature  

Tags: [[20-zettlekastenreinforcement-learning]] [[reward]] [[reward-prediction-error-rpe]] [[reinforcement-learning]]

# Credit Assignment Problem

A significant challenge in RL is the **credit assignment problem**. If a sequence of events leads to a failure, it's not immediately clear which specific decision in that sequence was responsible for the negative outcome, as rewards can be delayed. 

For example, a student receiving a bad grade at the end of a course isn't due to checking the grade, but likely to earlier actions like poor exam performance. The decision that leads to a good or bad outcome might not be immediately labeled with a high or low reward; the reward might only manifest much later. This means the agent cannot simply "copy" the data it collects, but must use the success/failure labels to infer how to improve its behavior

# References

1. [[Reinforcement Learning - Comprehensive Introduction]]