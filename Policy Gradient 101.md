# Policy Gradient 101
1. [Policy Gradient 101](#policy-gradient-101)
    - [Overview](#overview)
    - [Policy Definition and Mathematical Formulation](#policy-definition-and-mathematical-formulation)
      - State Vector and Logits
      - Softmax Function
      - Log-Probability & Gradient Derivation
2. [Implementation Examples](#implementation-examples)
    - [Bare Metal NumPy Implementation](#bare-metal-numpy-implementation)
    - [PyTorch Implementation](#pytorch-implementation)
    - [TRL Implementation](#trl-implementation)
3. [Q & A](#q--a)
    - [What is Gradient Descent in Supervised Learning?](#what-is-gradient-descent-in-supervised-learning)
    - [What is the Policy Gradient Theorem in Reinforcement Learning?](#what-is-the-policy-gradient-theorem-in-reinforcement-learning)
    - [Are These Just Stochastic Gradient Methods?](#are-these-just-stochastic-gradient-methods)
    - [Are Loss and Reward the Same?](#are-loss-and-reward-the-same)
    - [What Is Logistic Regression and Why "Logistic"?](#what-is-logistic-regression-and-why-logistic)
    - [What Is the Softmax Function?](#what-is-the-softmax-function)
    - [What’s the Chain Rule (in Calculus)?](#whats-the-chain-rule-in-calculus)
    - [How to Compute \(\nabla_\Theta \log p_a\) in Policy Gradient?](#how-to-compute-nabla_theta-log-p_a-in-policy-gradient)
    - [Final Comparison: Supervised vs. Policy Gradient Updates](#final-comparison-supervised-vs-policy-gradient-updates)



The following example demonstrates how to implement a simple policy gradient update using NumPy, entirely from scratch—that is, without relying on automatic differentiation libraries such as PyTorch. The policy in this case is a linear model that outputs **logits**, which are then transformed into a probability distribution via the softmax function. An action is sampled from this distribution, and the gradient of the log-probability of that action is computed manually.

We define the policy as follows. Given an input state vector   

$$
\mathbf{x} \in \mathbb{R}^{d},
$$  

the policy computes logits  

$$
\mathbf{z} = \Theta^\top \mathbf{x},
$$  

where $\Theta \in \mathbb{R}^{d \times k}$ is the parameter matrix, and $k$ is the number of actions.

The probability of selecting action $a$ is given by the softmax:

$$
p_a = \frac{\exp(z_a)}{\sum_{j=1}^k \exp(z_j)}.
$$

The log-probability of action $a$ is:

$$
\log p_a = z_a - \log\left(\sum_{j=1}^k \exp(z_j)\right).
$$

To compute the gradient of $\log p_a$ with respect to $\Theta$, we derive the partial derivatives for each column $j$ of $\Theta$:

$$
\frac{\partial \log p_a}{\partial \Theta_{\cdot,j}} = \mathbf{x} \left( \mathbb{1}_{j=a} - p_j \right),
$$

where $\mathbb{1}_{j=a}$ is the indicator function that is 1 if $j = a$, and 0 otherwise.

Equivalently, the full gradient matrix can be expressed compactly as:

$$
\nabla_{\Theta} \log p_a = \mathbf{x} \cdot (\mathbf{e}_a - \mathbf{p})^\top,
$$

where $\mathbf{e}_a$ is the one-hot vector corresponding to action $a$, and $\mathbf{p} \in \mathbb{R}^k$ is the vector of action probabilities.

Here is the raw NumPy implementation of this computation:


# Bare Metal Numpy 
gradient update 

```python
import numpy as np

def softmax(z):
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

class Policy:
    def __init__(self, input_dim, num_actions):
        # Initialize theta randomly: shape (input_dim, num_actions)
        self.theta = np.random.randn(input_dim, num_actions)
    
    def get_probs(self, x):
        # Compute logits: shape (num_actions,)
        logits = np.dot(x, self.theta)
        return softmax(logits)
    
    def sample_action(self, x):
        # Compute probabilities and sample an action according to them.
        probs = self.get_probs(x)
        action = np.random.choice(len(probs), p=probs)
        return action, probs
    
    def grad_log_prob(self, x, action, probs):
        # Compute the gradient of log P(action|x) with respect to theta.
        # The gradient is a matrix of the same shape as theta.
        # For each action j, the gradient is: x * (1_{j==action} - probs[j])
        one_hot = np.zeros_like(probs)
        one_hot[action] = 1.0
        # Outer product: shape (input_dim, num_actions)
        grad = np.outer(x, one_hot - probs)
        return grad

# Hyperparameters
input_dim = 4     # dimensionality of state x
num_actions = 2   # number of actions
learning_rate = 0.01

# Initialize policy
policy = Policy(input_dim, num_actions)

# Simulate a single step in an episode.
# In a real scenario, these would be collected by interacting with the environment.
x = np.random.randn(input_dim)   # example state
action, probs = policy.sample_action(x)
A = 1.0  # example advantage (typically computed from returns)

# Compute the gradient of log-probability for the chosen action.
grad_logp = policy.grad_log_prob(x, action, probs)

# Policy gradient update (gradient ascent on expected return)
# Update: theta <- theta + learning_rate * A * grad_logp
policy.theta += learning_rate * A * grad_logp

# For verification, let's print the updated theta.
print("Updated theta:\n", policy.theta)
```

# Pytorch 
```python
import torch
import torch.nn.functional as F

# Hyperparameters
d = 4   # input dimension (for the state space)
k = 3   # number of actions

# Dummy input state (shape: [d])
x = torch.randn(d, requires_grad=False)

# Initialize policy parameters, theta ∈ ℝ^(d × k)
theta = torch.nn.Parameter(torch.randn(d, k))  # shape: [d, k]

# Forward pass: compute logits and softmax probabilities
logits = x @ theta  # shape: [k]
probs = F.softmax(logits, dim=0)  # shape: [k]

# Sample an action from the policy
m = torch.distributions.Categorical(probs)
action = m.sample()  # integer in [0, k-1]

# Compute log-prob of that action
log_prob = m.log_prob(action)  # scalar

# Define a dummy reward
reward = 1.0

# Compute the policy gradient loss: -log p(a) * reward
loss = -log_prob * reward

# Backprop: compute ∇_theta of the objective
loss.backward()

# Print gradient (this is ∇_theta log p(a) * reward)
print("Gradient stored in theta.grad:")
print(theta.grad)

```

# TRL
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig

# Load a small pretrained language model (e.g. GPT-2)
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Ensure a pad token is defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define PPO configuration for TRL.
# Here we choose hyperparameters suitable for our toy example.
ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=1,
    mini_batch_size=1,
    ppo_epochs=4,
    init_kl_coef=0.01,
    # Additional hyperparameters can be set here as needed.
)

# Initialize the PPOTrainer with our model, tokenizer, and config.
ppo_trainer = PPOTrainer(ppo_config, model, tokenizer)

# Define a simple dummy reward function.
# For this toy example, we give a reward of 1 if the generated response
# contains the word "reward" (case-insensitive), otherwise 0.
def compute_reward(response_text):
    return 1.0 if "reward" in response_text.lower() else 0.0

# Define a simple prompt.
prompt = "This is a test. Generate a sentence that includes the word:"

# Prepare a batch (here, a list with a single prompt)
queries = [prompt]

# Use TRL's generate method to produce a response from the model.
# (Under the hood, TRL will use the current policy to sample tokens.)
response_tokens = ppo_trainer.generate(queries)
# Decode the generated tokens to obtain a human-readable string.
response_text = tokenizer.decode(response_tokens[0], skip_special_tokens=True)
print("Generated response:", response_text)

# Compute reward based on the generated response.
reward = compute_reward(response_text)
# For TRL, rewards are provided as a list (one per prompt in the batch).
rewards = [reward]
print("Reward:", rewards[0])

# Perform a PPO update step using TRL.
# This updates the model’s parameters using the prompt, generated response, and reward signal.
ppo_trainer.step(queries, response_tokens, rewards)

print("PPO update step completed.")

```


# Q & A 
---

## 1. What is Gradient Descent in Supervised Learning?

**Answer:**  
An optimization method to **minimize** a loss function $J(\Theta)$:

$$
\Theta \leftarrow \Theta - \eta \nabla_\Theta J(\Theta)
$$

where $\eta$ is the learning rate.

---

## 2. What is the Policy Gradient Theorem in Reinforcement Learning?

**Answer:**  
The gradient of expected return $J(\Theta)$ is:


```math
\nabla_\Theta J(\Theta) = \mathbb{E}_{(s, a) \sim \pi_\Theta} \left[ A(s, a) \nabla_\Theta \log \pi_\Theta(a \mid s) \right]
```

Policy update via gradient **ascent**:

$$
\Theta \leftarrow \Theta + \eta \, \mathbb{E}\left[A(s, a) \nabla_\Theta \log \pi_\Theta(a \mid s)\right]
$$

---

## 3. Are These Just Stochastic Gradient Methods?

**Answer:**  
Yes. Both use **stochastic gradient estimates**, differing in objective:

- **Supervised:** minimize loss  
- **RL:** maximize reward

---

## 4. Are Loss and Reward the Same?

**Answer:**  
No, they are conceptually **opposite**:

- **Loss**: Measures prediction error. You **minimize** it.  
- **Reward**: Measures success of actions. You **maximize** it.

You can define loss as negative reward, but they are conceptually distinct.

---

## 5. What Is Logistic Regression and Why "Logistic"?

**Answer:**  
Logistic regression uses the **sigmoid (logistic)** function to model probabilities:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = x^T \theta
$$

The log-odds (logit) is:

$$
\log\left( \frac{p}{1 - p} \right) = x^T \theta
$$

---

## 6. What Is the Softmax Function?

**Answer:**  
Softmax maps logits to a probability distribution over $k$ classes:

$$
p_j = \frac{e^{z_j}}{\sum_{l=1}^{k} e^{z_l}}
$$

When $k = 2$, softmax reduces to the sigmoid function.

---

## 7. What’s the Chain Rule (in Calculus)?

**Answer:**  
If $h(x) = f(g(x))$, then:

$$
h'(x) = f'(g(x)) \cdot g'(x)
$$

**Example:**  
Let $f(u) = e^u$, $g(x) = \log x$, then:

$$
h(x) = f(g(x)) = e^{\log x} = x, \quad h'(x) = 1
$$

---

## 8. How to Compute $\nabla_\Theta \log p_a$ in Policy Gradient?

**Answer:**  
For softmax policy where:

- $x \in \mathbb{R}^d$: state vector  
- $\Theta \in \mathbb{R}^{d \times k}$: policy parameter matrix  
- $z = x^T \Theta$: logits  
- $p = \text{softmax}(z)$: action probabilities  

The gradient of log-probability for action $a$ is:

$$
\nabla_\Theta \log p_a = x \cdot (e_a - p)^T
$$

Where $e_a$ is the one-hot vector for action $a$.  
Element-wise:

$$
\frac{\partial \log p_a}{\partial \Theta_{i,j}} = x_i \cdot (\mathbb{1}\{j = a\} - p_j)
$$

---

## 9. Final Comparison: Supervised vs. Policy Gradient Updates

| Type                    | Update Rule                                                                                                    | Objective       |
|-------------------------|------------------------------------------------------------------------------------------------------------------|-----------------|
| Supervised Learning     | $\Theta \leftarrow \Theta - \eta \nabla_\Theta J(\Theta)$                                                     | Minimize loss   |
| Policy Gradient (RL)    | $\Theta \leftarrow \Theta + \eta \, \mathbb{E}[A(s,a) \nabla_\Theta \log \pi_\Theta(a \mid s)]$              | Maximize reward |

---


### Q: What is logarithmic derivative identity?

**A:**
The logarithmic derivative identity states that if \( f(x) \) is a differentiable function and \( f(x) \neq 0 \), then

$$
\frac{d}{dx} \ln |f(x)| = \frac{f'(x)}{f(x)}.
$$

This follows directly from the chain rule. Essentially, it tells us that the derivative of the logarithm of a function is the ratio of the derivative of the function to the function itself. This identity is particularly useful when dealing with products or quotients of functions, as taking the logarithm converts multiplication into addition, thereby simplifying the differentiation process.

---

### Q: \( d_\theta f_\theta(x) \)

**A:**
For a function \( f_\theta(x) \) that depends on a parameter \(\theta\) (with \( f_\theta(x) \neq 0 \)), the logarithmic derivative with respect to \(\theta\) is given by

$$
\frac{d}{d\theta} \ln f_\theta(x) = \frac{d_\theta f_\theta(x)}{f_\theta(x)}.
$$

This identity follows directly from the chain rule. It shows that if you differentiate the logarithm of the function \( f_\theta(x) \) with respect to \(\theta\), you obtain the derivative of \( f_\theta(x) \) with respect to \(\theta\) divided by the function itself. Equivalently, you can express the derivative of the original function as

$$
d_\theta f_\theta(x) = f_\theta(x) \cdot \frac{d}{d\theta} \ln f_\theta(x).
$$

This relationship is especially useful in many areas such as statistics and optimization, where working with the logarithm of a function (for instance, a likelihood function) simplifies the differentiation process.

---

### Q: Does this have anything to do with expectation or integral?

**A:**
Yes, it does—especially in contexts where \( f_\theta(x) \) is a probability density or a parameter-dependent function integrated over some variable.

#### Expectation and Integration Connection

Consider a function \( f_\theta(x) \) that is a probability density function (pdf) with respect to \( x \). By definition, it satisfies

$$
\int f_\theta(x) \, dx = 1.
$$

If you differentiate this identity with respect to \(\theta\) (and under appropriate conditions allowing the interchange of differentiation and integration), you get

$$
\frac{d}{d\theta}\int f_\theta(x) \, dx = \int \frac{d}{d\theta} f_\theta(x) \, dx = 0.
$$

Now, using the logarithmic derivative identity

$$
\frac{d}{d\theta} f_\theta(x) = f_\theta(x) \cdot \frac{d}{d\theta}\ln f_\theta(x),
$$

the equation becomes

$$
\int f_\theta(x) \cdot \frac{d}{d\theta}\ln f_\theta(x) \, dx = 0.
$$

This shows that the expectation of the logarithmic derivative (often called the score function in statistics) with respect to the density \( f_\theta(x) \) is zero:

$$
\mathbb{E}\left[\frac{d}{d\theta}\ln f_\theta(x)\right] = 0.
$$

#### Why This Matters

- **Statistical Estimation:** In maximum likelihood estimation, the score function is the derivative of the log-likelihood with respect to the parameter \(\theta\). The fact that its expectation is zero under regular conditions is fundamental in proving properties of estimators (like unbiasedness) and in deriving the Cramér-Rao lower bound.
- **Differentiation Under the Integral Sign:** The process above is a common application of differentiating under the integral sign, which is a powerful technique in analysis and probability.

Thus, the logarithmic derivative identity is deeply connected with both expectation and integration when dealing with parameter-dependent functions or probability densities.

