
Created: 03-04-2026 23:38

Status: #baby 

Tags: [[prediction-error]] [[sequential-planning]] [[feedback-loops]]

# Error Accumulation in Long-Horizon Closed-Loop Prediction

## Definition
Error accumulation in long-horizon tasks occurs when sequential prediction errors from each step compound over time, degrading trajectory accuracy exponentially as task length increases. Closed-loop re-prediction (re-planning at each step based on new observations) reduces but cannot eliminate this drift.

## How It Works
In long-horizon tasks composed of many sequential subtasks, each step's trajectory prediction introduces error. This error shifts the true system state away from the predicted state. On the next step, the trajectory model receives an observation that is out-of-distribution relative to the clean, predicted trajectories in training data. The model produces a slightly incorrect prediction for this step as well. Critically, this second error is compounded by the first: the gripper position is already off-target from the previous step's error, and the new prediction is made from this corrupted state. Closed-loop re-prediction—re-planning at each timestep based on the current observation—helps by resetting the prediction to current ground truth, but it does not eliminate drift. If the model is statistically biased or tends to undershoot corrections, even closed-loop re-prediction will not recover from compounding drift. The LIBERO-Long benchmark (39.33% success) versus short-horizon tasks demonstrates this effect empirically: long-horizon performance drops substantially due to accumulated prediction error.

## Key Points
- Each step's prediction error becomes a biased input distribution for the next step's model, creating non-stationary uncertainty
- Closed-loop re-prediction resets some sources of error but does not address systematic bias or model limitations on out-of-distribution inputs
- Long-horizon performance degrades predictably and substantially; this is not a sign of poor model capacity but an inherent property of sequential prediction
- The feedback loop of increasing error can accelerate: off-target gripper positions lead to even more out-of-distribution observations, causing larger prediction errors
- Task length becomes a critical variable; breaking tasks into shorter subgoals or using hierarchical planning could mitigate accumulation

## When to Use
Applicable to any sequential decision-making system that chains predictions, including robot planning, video prediction, and reinforcement learning with learned models.



# See Also

- [[action-chunking-with-receding-horizon-control]]
- [[receding-horizon-control]]
- [[memoryless-reactive-control-cascading-failures]]
- [[multi-step-compounding-error-in-sequential-tasks]]
- [[keyframe-abstraction-for-sequential-manipulation]]

# References

1. [[2401.00025 - ATM Any-point Trajectory]]
