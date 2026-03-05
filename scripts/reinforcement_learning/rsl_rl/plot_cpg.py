"""Simulate the CPG network with GELU vs TANH and compare side by side."""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# === CPG parameters (from go1_cpg_train_mdpo.py) ===
W_rec = torch.tensor([
    [0.00, 0.54, -6.30],
    [6.30, 0.00, -2.16],
    [0.36, 1.80,  0.00],
])
W_in = torch.tensor([2.25, 0.0, 0.0])
bias = torch.tensor([0.0, -0.2, -0.3]) # control duty cycle
g_adapt = torch.tensor([1.8, 1.5, 0.0])

fixed_dn = 0.88
cpg_dt = 0.3
cpg_substeps = 1
n_steps = 500


def simulate_cpg(activation_fn, activation_name):
    """Run CPG simulation with a given activation function."""
    torch.manual_seed(42)
    cpg_x = torch.randn(4, 3) * 0.5
    cpg_a = torch.zeros(4, 3)

    history_x, history_r, history_clock = [], [], []

    for step in range(n_steps):
        ext_input = W_in * fixed_dn / 4
        for _ in range(cpg_substeps):
            r = activation_fn(cpg_x)
            rec_input = fixed_dn * torch.einsum("ln,mn->lm", r, W_rec)
            dxdt = -cpg_x + rec_input + ext_input + bias # - cpg_a
            # dadt = -cpg_a + g_adapt * r
            cpg_x = cpg_x + cpg_dt * dxdt
            # cpg_a = cpg_a + cpg_dt * dadt

        new_r = activation_fn(cpg_x)
        clock = new_r[:, :2].reshape(-1)

        history_x.append(cpg_x.clone())
        history_r.append(new_r.clone())
        history_clock.append(clock.clone())

    return (
        torch.stack(history_x).numpy(),
        torch.stack(history_r).numpy(),
        torch.stack(history_clock).numpy(),
    )


# === Simulate both ===
X_gelu, R_gelu, C_gelu = simulate_cpg(F.gelu, "GELU")
X_tanh, R_tanh, C_tanh = simulate_cpg(torch.tanh, "Tanh")
t = np.arange(n_steps) * cpg_dt

leg_names = ["FL", "FR", "RL", "RR"]
neuron_names = ["E1", "E2", "I"]
colors = ["#e41a1c", "#377eb8", "#4daf4a"]
clock_colors = ["#e41a1c", "#ff7f00", "#377eb8", "#984ea3"]

# === Plot: 4 rows × 2 columns (GELU left, Tanh right) ===
fig, axes = plt.subplots(4, 2, figsize=(18, 14), sharex=True)
fig.suptitle("CPG Dynamics: GELU vs Tanh", fontsize=14, fontweight="bold")

for col, (name, X, R, C) in enumerate([
    ("GELU", X_gelu, R_gelu, C_gelu),
    ("Tanh", X_tanh, R_tanh, C_tanh),
]):
    # Row 0: cpg_x
    ax = axes[0, col]
    for leg in range(4):
        for n in range(3):
            ls = ["-", "--"][leg // 2]
            alpha = 1.0 if leg < 2 else 0.6
            ax.plot(t, X[:, leg, n], ls=ls, alpha=alpha, color=colors[n],
                    label=f"{neuron_names[n]}" if (leg == 0 and col == 0) else None)
    ax.set_ylabel("cpg_x") if col == 0 else None
    ax.set_title(f"{name}")
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.legend(loc="upper right", fontsize=8)

    # Row 1: r (firing rates)
    ax = axes[1, col]
    for leg in range(4):
        for n in range(3):
            ls = ["-", "--"][leg // 2]
            alpha = 1.0 if leg < 2 else 0.6
            ax.plot(t, R[:, leg, n], ls=ls, alpha=alpha, color=colors[n])
    ax.set_ylabel(f"r = {name.lower()}(cpg_x)") if col == 0 else None
    ax.grid(True, alpha=0.3)

    # Row 2: Clock (8-dim)
    ax = axes[2, col]
    for leg in range(4):
        ax.plot(t, C[:, leg * 2], "-", color=clock_colors[leg], label=f"{leg_names[leg]}_E1" if col == 0 else None)
        ax.plot(t, C[:, leg * 2 + 1], "--", color=clock_colors[leg], label=f"{leg_names[leg]}_E2" if col == 0 else None)
    ax.set_ylabel("Clock (8-dim)") if col == 0 else None
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.legend(loc="upper right", ncol=4, fontsize=7)

    # Row 3: Diagonal pairs (trot check)
    ax = axes[3, col]
    ax.plot(t, C[:, 0], "-", color="#e41a1c", lw=2, label="FL_E1" if col == 0 else None)
    ax.plot(t, C[:, 6], "--", color="#e41a1c", lw=2, label="RR_E1" if col == 0 else None)
    ax.plot(t, C[:, 2], "-", color="#377eb8", lw=2, label="FR_E1" if col == 0 else None)
    ax.plot(t, C[:, 4], "--", color="#377eb8", lw=2, label="RL_E1" if col == 0 else None)
    ax.set_ylabel("Diagonal pairs") if col == 0 else None
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.legend(loc="upper right", ncol=4, fontsize=8)

# Add activation function comparison inset
ax_inset = fig.add_axes([0.42, 0.92, 0.16, 0.06])
x_range = torch.linspace(-3, 3, 200)
ax_inset.plot(x_range.numpy(), F.gelu(x_range).numpy(), "-", color="#e41a1c", lw=1.5, label="GELU")
ax_inset.plot(x_range.numpy(), torch.tanh(x_range).numpy(), "-", color="#377eb8", lw=1.5, label="Tanh")
ax_inset.axhline(0, color="gray", lw=0.5)
ax_inset.axvline(0, color="gray", lw=0.5)
ax_inset.legend(fontsize=7, loc="lower right")
ax_inset.set_title("Activation functions", fontsize=8)
ax_inset.tick_params(labelsize=6)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("cpg_gelu_vs_tanh.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to cpg_gelu_vs_tanh.png")
