import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------
# Robot parameters
# -----------------------------
L1, L2 = 0.2, 0.2

# Workspace limits for sliders
pos_limits = {
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "z": (-0.5, -0.05)
}

# Desired foot positions (local frame)
legs = {
    "FR": [0.0, 0.0, -0.3],
    "FL": [0.0,  0.0, -0.3],
    "RR": [0.0, 0.0, -0.3],
    "RL": [0.0,  0.0, -0.3]
}

# Store defaults
defaults = {k: v[:] for k, v in legs.items()}

# Base positions
base_pos = {
    "FR": np.array([0.3, -0.2, 0.0]),
    "FL": np.array([0.3,  0.2, 0.0]),
    "RR": np.array([-0.3, -0.2, 0.0]),
    "RL": np.array([-0.3,  0.2, 0.0])
}

# Body shape
body = np.array([
    [0.3, -0.2, 0],
    [0.3, 0.2, 0],
    [-0.3, 0.2, 0],
    [-0.3, -0.2, 0],
    [0.3, -0.2, 0]
])

# -----------------------------
# Inverse Kinematics
# -----------------------------
def ik(x, y, z):
    t1 = np.arctan2(y, -z)

    R = np.sqrt(y**2 + z**2)
    if R < 1e-6:
        R = 1e-6

    D = x**2 + R**2
    cos_t3 = (D - L1**2 - L2**2) / (2 * L1 * L2)
    cos_t3 = np.clip(cos_t3, -1.0, 1.0)

    t3 = np.arccos(cos_t3)

    alpha = np.arctan2(x, R)
    beta  = np.arctan2(L2*np.sin(t3), L1 + L2*np.cos(t3))

    t2 = alpha - beta

    return t1, t2, t3

# -----------------------------
# Forward Kinematics
# -----------------------------
def fk(t1, t2, t3):
    X = L1*np.sin(t2) + L2*np.sin(t2+t3)
    R = L1*np.cos(t2) + L2*np.cos(t2+t3)
    Y = R*np.sin(t1)
    Z = -R*np.cos(t1)
    return np.array([X, Y, Z])

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

def draw():
    ax.clear()
    ax.plot(body[:,0], body[:,1], body[:,2], linewidth=3)

    # -----------------------------
    # STORE ANGLES
    # -----------------------------
    angles_dict = {}

    for leg, (x,y,z) in legs.items():
        base = base_pos[leg]

        # IK solve
        t1, t2, t3 = ik(x, y, z)

        # store (convert to degrees)
        angles_dict[leg] = np.degrees([t1, t2, t3])

        # Knee
        knee = base + np.array([
            L1*np.sin(t2),
            L1*np.cos(t2)*np.sin(t1),
            -L1*np.cos(t2)*np.cos(t1)
        ])

        # Foot
        foot = base + fk(t1,t2,t3)

        # Draw
        ax.plot([base[0], knee[0]], [base[1], knee[1]], [base[2], knee[2]], marker='o')
        ax.plot([knee[0], foot[0]], [knee[1], foot[1]], [knee[2], foot[2]], marker='o')

    # -----------------------------
    # INFO BOX 
    # -----------------------------
    info_text = "         θ1     θ2     θ3\n"
    for leg in ["FR","FL","RR","RL"]:
        t1, t2, t3 = angles_dict[leg]
        info_text += f"{leg}: {t1:6.1f} {t2:6.1f} {t3:6.1f}\n"

    ax.text2D(
        0.02, 0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8)
    )

    ax.set_xlim([-0.6,0.6])
    ax.set_ylim([-0.6,0.6])
    ax.set_zlim([-0.6,0.6])

# -----------------------------
# TKINTER UI
# -----------------------------
root = tk.Tk()
root.title("Quadruped IK Simulator")
root.geometry("1100x650")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

control = tk.Frame(root)
control.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

def update(leg, i, val):
    legs[leg][i] = float(val)
    draw()
    canvas.draw_idle()

sliders = []

# Sliders
for leg in legs:
    f = tk.LabelFrame(control, text=leg)
    f.pack(fill="x", pady=6)

    for i, name in enumerate(["X","Y","Z"]):
        s = tk.Scale(
            f,
            from_=pos_limits[name.lower()][0],
            to=pos_limits[name.lower()][1],
            resolution=0.01,
            orient=tk.HORIZONTAL,
            label=name,
            command=lambda v, l=leg, j=i: update(l,j,v)
        )
        s.set(legs[leg][i])
        s.pack(side="left", expand=True, fill="x")

        sliders.append((s, leg, i))

# Reset
def reset():
    for l in legs:
        legs[l] = defaults[l][:]
    for s, l, i in sliders:
        s.set(legs[l][i])
    draw()
    canvas.draw()

tk.Button(control, text="Reset", command=reset).pack(fill="x", pady=10)

draw()
canvas.draw()
root.mainloop()