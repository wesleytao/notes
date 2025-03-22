# Cheat Sheet

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
