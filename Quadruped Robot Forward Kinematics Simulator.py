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

limits = {
    "theta1": (-np.pi/6, np.pi/6),
    "theta2": (-np.pi/2, np.pi/6),
    "theta3": (0, np.pi)
}

legs = {
    "FR": [0.0, -0.5, 1.0],
    "FL": [0.0, -0.5, 1.0],
    "RR": [0.0, -0.5, 1.0],
    "RL": [0.0, -0.5, 1.0]
}

# store defaults
defaults = {k: v[:] for k, v in legs.items()}

base_pos = {
    "FR": np.array([0.3, -0.2, 0.0]),
    "FL": np.array([0.3, 0.2, 0.0]),
    "RR": np.array([-0.3, -0.2, 0.0]),
    "RL": np.array([-0.3, 0.2, 0.0])
}

body = np.array([
    [0.3, -0.2, 0],
    [0.3, 0.2, 0],
    [-0.3, 0.2, 0],
    [-0.3, -0.2, 0],
    [0.3, -0.2, 0]
])

# -----------------------------
# FK
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

    for leg, (t1,t2,t3) in legs.items():
        base = base_pos[leg]

        knee = base + np.array([
            L1*np.sin(t2),
            L1*np.cos(t2)*np.sin(t1),
            -L1*np.cos(t2)*np.cos(t1)
        ])

        foot = base + fk(t1,t2,t3)

        ax.plot([base[0], knee[0]], [base[1], knee[1]], [base[2], knee[2]], marker='o')
        ax.plot([knee[0], foot[0]], [knee[1], foot[1]], [knee[2], foot[2]], marker='o')

    ax.set_xlim([-0.6,0.6])
    ax.set_ylim([-0.6,0.6])
    ax.set_zlim([-0.6,0.6])

# -----------------------------
# TKINTER UI
# -----------------------------
root = tk.Tk()
root.title("Quadruped FK Simulator")
root.geometry("1100x650")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

control = tk.Frame(root)
control.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

# -----------------------------
# Update
# -----------------------------
def update(leg, i, val):
    legs[leg][i] = float(val)
    draw()
    canvas.draw_idle()

# -----------------------------
# Store sliders
# -----------------------------
sliders = []

# -----------------------------
# Sliders
# -----------------------------
for leg in legs:
    f = tk.LabelFrame(control, text=leg)
    f.pack(fill="x", pady=6)

    for i, name in enumerate(["θ1","θ2","θ3"]):
        s = tk.Scale(
            f,
            from_=limits[f"theta{i+1}"][0],
            to=limits[f"theta{i+1}"][1],
            resolution=0.01,
            orient=tk.HORIZONTAL,
            label=name,
            command=lambda v, l=leg, j=i: update(l,j,v)
        )
        s.set(legs[leg][i])
        s.pack(side="left", expand=True, fill="x")

        sliders.append((s, leg, i))  # tiny tracking

# -----------------------------
# Reset
# -----------------------------
def reset():
    # 1. Force the data dictionary back to default values
    for l in legs: legs[l] = defaults[l][:]
    # 2. Sync the sliders (this ensures the UI matches the data)
    for s, l, i in sliders: s.set(legs[l][i])
    # 3. Final 'snap' to redraw the robot in the correct pose
    draw();
    canvas.draw()
# -----------------------------
tk.Button(control, text="Reset", command=reset).pack(fill="x", pady=10)

# -----------------------------
draw()
canvas.draw()
root.mainloop()