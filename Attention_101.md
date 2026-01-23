# Attention

$$
\text{SelfAttn}(X) = \underbrace{\text{softmax}\left(\frac{(X_{n \times d} W^Q_{d \times d_k})(X_{n \times d} W^K_{d \times d_k})^\top_{d_k \times n}}{\sqrt{d_k}} + M_{n \times n}\right)}_{A_{n \times n}} (X_{n \times d} W^V_{d \times d_v})_{n \times d_v} = O_{n \times d_v}
$$


