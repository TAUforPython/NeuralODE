# Neural ODE & Hamiltonian Neural ODE 

A comprehensive implementation and comparison of **Standard Neural ODEs** and **Hamiltonian Neural ODEs** using [heyoka.py](https://bluescarni.github.io/heyoka.py/) — a high-performance Taylor integration library with automatic differentiation and SIMD batching support.

This repository is inspired by and based on concepts from:
- [opinti/hamiltonian-neural-ode](https://github.com/opinti/hamiltonian-neural-ode) — Hamiltonian Neural ODE experiments
- [Greydanus et al., "Hamiltonian Neural Networks"](https://arxiv.org/abs/1906.01563) — Original HNN paper
- [P-SympNet paper](https://arxiv.org/abs/2203.16573) — Symplectic neural networks for Hamiltonian systems

---

## Overview

| Approach | Structure | Preserves Energy | Interpretable |
|----------|-----------|------------------|---------------|
| **Standard Neural ODE** | Unconstrained black-box `f_θ(x, t)` | ❌ No | ❌ No |
| **Hamiltonian Neural ODE** | Learned scalar `H_θ(q, p)` with symplectic gradients | ✅ Yes | ✅ Yes |

---

## Mathematical Background

**Neural ODE**: `dx/dt = f_θ(x(t), t)`

**Hamiltonian Neural ODE**: Learn `H_θ(q,p)`, then derive:
```
dq/dt = +∇_p H_θ(q,p)
dp/dt = -∇_q H_θ(q,p)
```

This guarantees energy conservation, volume preservation (Liouville's theorem), and time-reversibility.

---

## Double Mass-Spring System

A canonical 4D Hamiltonian system:

**Hamiltonian**: `H = p₁²/2m₁ + p₂²/2m₂ + ½k₁q₁² + ½k₂(q₂-q₁)²`

| Parameter | Value |
|-----------|-------|
| `k₁` | 10.0 |
| `k₂` | 7.5 |
| `m₁, m₂` | 1.0 |
| IC | `[0.15, 0.6, 0.5, 0.5]` |

---

## Key Differences

| Aspect | Standard NODE | Hamiltonian NODE |
|--------|---------------|------------------|
| Network output | Vector field `f_θ ∈ ℝ⁴` | Scalar `H_θ ∈ ℝ` |
| Parameters | ~4× more | ~4× fewer |
| Energy drift | Unbounded | Bounded |
| Phase space volume | Not preserved | Preserved |
| Long-term stability | Poor | Excellent |

---

### References

1. Greydanus et al. (2019) — [Hamiltonian Neural Networks](https://arxiv.org/abs/1906.01563)
2. Chen et al. (2018) — [Neural ODEs](https://papers.nips.cc/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html)
3. Biscani & Izzo (2020) — [heyoka](https://github.com/bluescarni/heyoka)
4. P-SympNet — [Symplectic neural networks](https://arxiv.org/abs/2203.16573)

---

*Built with heyoka.py — Taylor integration meets automatic differentiation.*
