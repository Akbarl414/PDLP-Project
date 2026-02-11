# Restarted PDHG (PDLP) Solver  
C++ implementation with HiGHS integration

This project implements a sparse **Primal–Dual Hybrid Gradient (PDHG)** solver for linear programs in standard form. It supports optional diagonal scaling, feasibility-based stopping, and benchmarking against **HiGHS** using `.mps` input files.

The goal of the project is to experiment with restarted PDHG variants and scaling strategies on real LP instances.

---

## Problem Form

The solver handles LPs in standard form:

$$
\min_{x \ge 0}  c^T x 
\quad \text{s.t.} \quad 
Ax = b
$$

The corresponding dual problem is:

$$
\max_y  b^T y 
\quad \text{s.t.} \quad 
A^T y \le c
$$

---

## Algorithm

Each iteration performs the standard PDHG updates:

### Primal step

$$
x^{k+1} = \max\left(0,\; x^k + \tau (A^T y^k - c)\right)
$$

### Dual step

$$
y^{k+1} = y^k - \sigma A(2x^{k+1} - x^k) + \sigma b
$$

Projection enforces $x \ge 0$.

---

## Step Size Strategy

If not provided by the user, the solver computes:

$$
\text{step-size} = \frac{1}{\|A\|_2}
$$

where $\|A\|_2$ is estimated via a sparse power method.

A balance parameter is computed as:

$$
\text{step-balance} = \frac{\|c\|_2}{\|b\|_2}
$$

Then:

$$
\tau = \frac{\text{step-size}}{\text{step-balance}},
\quad
\sigma = \text{step-size} \cdot \text{step-balance}
$$

Optionally, both can be flattened so that $\tau = \sigma$.

---

## Termination Criteria

The solver runs until all of the following hold:

- **Primal feasibility**

$$
\|Ax - b\|_2 \le \varepsilon_p
$$

- **Dual feasibility**

- **Duality gap**

$$
|b^T y - c^T x|
\le
\varepsilon \left(1 + |c^T x| + |b^T y|\right)
$$

An iteration cap acts as a safeguard.

---

## Scaling (Optional)

To improve conditioning, the solver supports:

- **Ruiz diagonal rescaling**
- **Chambolle–Pock diagonal preconditioning**
- Alternate row/column scaling variants

Scaling modifies $A$, $b$, and $c$ before solving.  
Solutions are automatically unscaled before final feasibility checks.

---

## Sparse Implementation

The matrix $A$ is stored in column-compressed format:

- `start` — column pointers  
- `index` — row indices  
- `value` — nonzero values  

All matrix-vector operations use this sparse structure.

---

## Build

Requires:

- C++17
- HiGHS

Example:

```bash
g++ -O3 -std=c++17 main.cpp ProjectFunctions.cpp \
-I/path/to/highs/include \
-L/path/to/highs/lib -lhighs \
-o pdlp

```

## Run

Basic usage:

    ./pdlp <model>

Example:

    ./pdlp afiro

This loads:

    <model>.mps

(Default model is `avgas` if none is provided.)

---

### Full Command Format

    ./pdlp <model> <step_size> <debug> <rescale> <iter_cap> <chamPock> <ruiz_iter_cap> <flatten_steps> <alternate_scaling>

All arguments after `<model>` are optional.

---

### Argument Breakdown

| Argument | Description |
|----------|-------------|
| `argv[1]` | Model name (without `.mps`) |
| `argv[2]` | Step size (`0` = auto-compute) |
| `argv[3]` | `"debug"` to enable periodic logging |
| `argv[4]` | `"rescale"` to enable scaling |
| `argv[5]` | Iteration cap (e.g. `2500000`) |
| `argv[6]` | Chambolle–Pock scaling (`1` = on, `0` = off) |
| `argv[7]` | Ruiz rescaling iteration cap |
| `argv[8]` | Flatten step sizes (`1` = τ = σ) |
| `argv[9]` | Use alternate scaling variants (`1` = on) |

---

### Example Runs

**Default solve**

    ./pdlp afiro

**Auto step size + scaling + CP enabled**

    ./pdlp afiro 0 none rescale 2500000 1 10 0 0

**Custom step size, no scaling**

    ./pdlp afiro 0.01

**Flattened step sizes with alternate scaling**

    ./pdlp afiro 0 debug rescale 2500000 1 10 1 1

---

## Output

The solver prints:

- Objective value  
- Iteration count  
- Duality gap  
- Feasibility metrics  
- Runtime  

It also appends results to a CSV file for benchmarking.

---

## HiGHS Comparison

After PDHG finishes, the same model is solved with HiGHS.

The relative objective difference is reported as:

$$|z_{\text{HiGHS}} - z_{\text{PDLP}}| / \max(1, |z_{\text{HiGHS}}|)$$

This acts as a correctness check.

---

## Notes

- `.mps` file paths are currently hardcoded in `main.cpp`.
- The solver is intended for experimentation and benchmarking.
- Adaptive step-size logic is partially implemented and can be extended.


