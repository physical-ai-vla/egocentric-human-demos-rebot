"""3D view of random samples (spec 33): current pose T_t (origin of the anchor frame) and the k = 1, 4, 8, 16 targets,
all drawn as A_k = inv(T_t) T(t + k dt) from ONE origin -- a sequential-delta bug would show all k collapsed near the origin."""
import numpy as np

from ..geometry.transforms import pose9_to_T


def plot_samples(cart20_list, titles, out_png, ks=(1, 4, 8, 16), axis_len=0.01):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(cart20_list); fig = plt.figure(figsize=(5 * n, 5))
    for i, (c, ttl) in enumerate(zip(cart20_list, titles)):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        for o, col in ((0, "tab:blue"), (10, "tab:red")):
            P = np.array([pose9_to_T(c[k - 1, o:o + 9])[:3, 3] for k in range(1, 17)])
            ax.plot([0, *P[:, 0]], [0, *P[:, 1]], [0, *P[:, 2]], color=col, alpha=0.4)
            for k in ks:
                T = pose9_to_T(c[k - 1, o:o + 9]); p = T[:3, 3]
                ax.plot([0, p[0]], [0, p[1]], [0, p[2]], color=col, ls=":", lw=0.8)
                for a, ac in zip(range(3), ("r", "g", "b")):
                    e = p + T[:3, a] * axis_len; ax.plot([p[0], e[0]], [p[1], e[1]], [p[2], e[2]], color=ac, lw=1)
                ax.text(*p, f"k{k}", fontsize=7, color=col)
        ax.scatter([0], [0], [0], color="k", s=20); ax.set_title(ttl, fontsize=8)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.suptitle("CART20 targets from the current pose T_t (blue = left, red = right; RGB = target axes)")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
