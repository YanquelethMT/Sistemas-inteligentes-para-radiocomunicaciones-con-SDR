import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

N = 50
n = np.arange(N)
x = 0.7 * np.random.randn(N)

L = 6           # longitud del segmento
R = 2           # traslape
step = L - R

starts = [0, step, N - L]
starts = sorted(set(starts))
segments = [(s, s + L - 1) for s in starts]

fig = plt.figure(figsize=(13, 8))
ax = plt.gca()
ax.set_xlim(-1, N + 1)
ax.set_ylim(-8, 4)
ax.axis("off")

ax.annotate("", xy=(N + 0.8, 0), xytext=(-0.8, 0),
            arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(-0.8, 3.6), xytext=(-0.8, 0),
            arrowprops=dict(arrowstyle="->", lw=2))

for i, xi in zip(n, x):
    ax.plot([i, i], [0, xi], lw=2, alpha=0.75)
    ax.plot(i, xi, marker="o", ms=6, alpha=0.9)

for dx in [10.1, 11.2, 12.3]:
    ax.plot(dx, 0.45, marker="o", ms=8)

ax.text(-0.15, -0.55, "0", fontsize=18, fontfamily="serif")
ax.text(0.1, 3.0, r"$r_i(n)$", fontsize=28)
ax.text(N - 0.2, -0.6, "Q", fontsize=24, fontfamily="serif")

y_top = -3.2
y_bottom = -5.6
y_fft = -7.2

for idx, (s, e) in enumerate(segments, start=1):
    mid = 0.5 * (s + e)

    ax.plot([s, s], [y_top - 0.15, y_bottom + 0.15], "--", lw=2, dashes=(4, 3))
    ax.plot([e, e], [y_top - 0.15, y_bottom + 0.15], "--", lw=2, dashes=(4, 3))

    ax.annotate("", xy=(e, y_top), xytext=(s, y_top),
                arrowprops=dict(arrowstyle="<->", lw=2))
    ax.text(mid - 0.9, y_top + 0.35, rf"$r_{{i,{idx}}}(n)$", fontsize=22)

    ax.plot([mid, mid], [y_top, y_bottom], lw=2.5, alpha=0.7)

    ax.annotate("", xy=(mid, y_fft + 0.15), xytext=(mid, y_bottom),
                arrowprops=dict(arrowstyle="->", lw=2.5))
    ax.text(mid - 0.9, (y_bottom + y_fft) / 2 - 0.1, "FFT", fontsize=24, fontfamily="serif")
    ax.text(mid - 1.2, y_fft - 0.45, rf"$P_{{r_{{i,{idx}}}}}(k)$", fontsize=22)

s1, e1 = segments[0]
s2, e2 = segments[1]
ax.annotate("", xy=(s2, y_top - 0.55), xytext=(e1, y_top - 0.55),
            arrowprops=dict(arrowstyle="<->", lw=2))
ax.text((s2 + e1) / 2 - 0.75, y_top - 1.2, r"$L-R$", fontsize=24)

ax.annotate("", xy=(e1, y_bottom + 0.2), xytext=(s1, y_bottom + 0.2),
            arrowprops=dict(arrowstyle="<->", lw=2))
ax.text((s1 + e1) / 2 - 1.25, y_bottom - 0.45, r"longitud $L$", fontsize=24)

for dx in [11.2, 12.3, 13.4]:
    ax.plot(dx, -4.35, marker="o", ms=8)
for dx in [10.8, 11.9, 13.0]:
    ax.plot(dx, -7.35, marker="o", ms=8)

plt.tight_layout()
plt.savefig("welch_diagrama_python.png", dpi=200, bbox_inches="tight")
plt.show()
