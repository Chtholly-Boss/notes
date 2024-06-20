# Fast Fourier Transform
## Lemmas
![divide_lemma](./figure/fft/divide_lemma.png)
Based on this,We can compute $F_{2n}$ from $F_{n}$
![divide_res](./figure/fft/divide_res.png)
And we can get a chain:
![c-t-factor](./figure/fft/c_t_factorization.png)
## The Cooley-Tukey Framework
![cooley-tukey](./figure/fft/cooley_tukey_framework.png)
And Now we see carefully on each step.
- $x \leftarrow P_n x$
a bit-reversal permutation algorithm should be apply here.

- $x \leftarrow A_q x$
weight (pre)computations

- $x \leftarrow A_q x$
a choice of loop orderings