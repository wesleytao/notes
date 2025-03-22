# Policy Gradient 101

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
