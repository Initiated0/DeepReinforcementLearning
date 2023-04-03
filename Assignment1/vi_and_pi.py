### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	policy: np.array[nS]
		The policy to evaluate. Maps states to actions.
	tol: float
		Terminate policy evaluation when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns
	-------
	value_function: np.ndarray[nS]
		The value function of the given policy, where value_function[s] is
		the value of state s
	"""

	value_function = np.zeros(nS)

	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	flag = True
	while flag:

		mx_val = 0.0

		for state in range(nS):
			previous_value_function = value_function[state]
			# print("polciyEvaluation: ", previous_value_fucntion)
			val = 0.0
			for state_policy in P[state][policy[state]]:
				transitionProbability = state_policy[0]
				nextState = state_policy[1]
				currentReward = state_policy[2]
				val += transitionProbability * (currentReward + gamma * value_function[nextState])
				value_function[state] = val
			mx_val = max(mx_val, abs(value_function[state] - previous_value_function))

		if mx_val < tol:
			flag = False

	return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new_policy: np.ndarray[nS]
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""

	new_policy = np.zeros(nS, dtype='int')

	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	for state in range(nS):
		next_value_function = np.zeros(nA)
		for action in range(nA):
			for state_policy in P[state][action]:
				transitionProbability = state_policy[0]
				nextState = state_policy[1]
				currentReward = state_policy[2]
				next_value_function[action] += transitionProbability * (currentReward + gamma * value_from_policy[nextState])
		new_policy[state] = np.argmax(next_value_function)

	return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
	"""Runs policy iteration.

	You should call the policy_evaluation() and policy_improvement() methods to
	implement this method.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		tol parameter used in policy_evaluation()
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

	value_function = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)

	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	flag = True
	count = 0
	while flag:

		mx_val = 0.0
		count += 1

		for state in range(nS):
			previous_value_function = value_function[state]
			value_function = policy_evaluation(P, nS, nA, policy)
			policy = policy_improvement(P, nS, nA, value_function, policy)
			mx_val = max(mx_val, abs(value_function[state] - previous_value_function))

		if mx_val < tol:
			flag = False

	return value_function, policy, count


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		Terminate value iteration when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

	value_function = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	flag = True
	count = 0
	while flag:
		count += 1
		mx_val = 0.0

		for state in range(nS):
			previous_value_function = value_function[state]
			next_value_function = np.zeros(nA)
			for action in range(nA):
				for state_policy in P[state][action]:
					transitionProbability = state_policy[0]
					nextState = state_policy[1]
					currentReward = state_policy[2]
					next_value_function[action] += transitionProbability * (currentReward + gamma * value_function[nextState])

			value_function[state] = max(next_value_function)
			policy[state] = np.argmax(next_value_function)
			mx_val = max(mx_val, abs(value_function[state] - previous_value_function))

		if mx_val < tol:
			flag = False

	return value_function, policy, count

def render_single(env, policy, max_steps=100):
  """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

  episode_reward = 0
  ob = env.reset()
  for t in range(max_steps):
    env.render()
    time.sleep(0.25)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    if done:
      break
  env.render();
  if not done:
    print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
  else:
  	print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":

	# comment/uncomment these lines to switch between deterministic/stochastic environments
	#env = gym.make("Deterministic-4x4-FrozenLake-v0")
	env = gym.make("Stochastic-4x4-FrozenLake-v0")
	print("transitionProbability, P :", env.P)
	print("#ofStates :", env.nS)
	print("#ofActions :", env.nA)

	print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

	V_pi, p_pi, pi_count = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_pi, 100)

	print("value function: ", V_pi)
	print("Policy: ", p_pi)
	print("Policy iteration count: ", pi_count)


	print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)

	V_vi, p_vi, vi_count = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_vi, 100)

	print("value function: ", V_vi)
	print("Policy: ", p_vi)
	print("Value iteration count: ", vi_count)


################################################
#  Results of Deterministic-4x4-FrozenLake-v0  #
################################################

# Policy Iteration:

# Episode reward: 1.000000
# value function:  [0.59  0.656 0.729 0.656 0.656 0.    0.81  0.    0.729 0.81  0.9   0.
#  0.    0.9   1.    0.   ]
# Policy:  [1 2 1 0 1 0 1 0 2 1 1 0 0 2 2 0]
# Policy iteration count:  1
# ----------------------------------------------------------------------------------------
# Value Iteration:

# Episode reward: 1.000000
# value function:  [0.59  0.656 0.729 0.656 0.656 0.    0.81  0.    0.729 0.81  0.9   0.
#  0.    0.9   1.    0.   ]
# Policy:  [1 2 1 0 1 0 1 0 2 1 1 0 0 2 2 0]
# Value iteration count:  7

################################################
#   Results of Stochastic-4x4-FrozenLake-v0    #
################################################

# Policy Iteration:

# Episode reward: 1.000000
# value function:  [0.021 0.021 0.039 0.019 0.03  0.    0.071 0.    0.072 0.156 0.197 0.
#  0.    0.251 0.431 0.   ]
# Policy:  [1 3 0 3 0 0 0 0 3 1 0 0 0 2 1 0]
# Policy iteration count:  2

# ----------------------------------------------------------------------------------------

# Value Iteration:

# Episode reward: 0.000000
# value function:  [0.064 0.058 0.072 0.054 0.088 0.    0.111 0.    0.143 0.246 0.299 0.
#  0.    0.379 0.639 0.   ]
# Policy:  [0 3 0 3 0 0 0 0 3 1 0 0 0 2 1 0]
# Value iteration count:  23