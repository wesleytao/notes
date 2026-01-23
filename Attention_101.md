# Attention 

$$
\text{SelfAttn}(X)
=
\underbrace{
\text{softmax}\!\left(
\frac{
\left(X_{n\times d_{\text{model}}}\, W^{Q}_{d_{\text{model}}\times d_k}\right)
\left(X_{n\times d_{\text{model}}}\, W^{K}_{d_{\text{model}}\times d_k}\right)^{\!\top}_{d_k\times n}
}{
\sqrt{d_k}
}
+ M_{n\times n}
\right)
}_{A_{n\times n}}
\left(X_{n\times d_{\text{model}}}\, W^{V}_{d_{\text{model}}\times d_v}\right)_{n\times d_v}
=
O_{n\times d_v}
$$
