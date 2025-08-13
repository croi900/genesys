#!/usr/bin/env python3
"""Compare PRyMordial t_of_T with and without the Genesys NP contributions.

We call PRyMclass twice:
1.  Standard GR background (no arguments).
2.  With New-Physics energy/pressure functions coming from a solved Genesys
    model (here M1V0).  Flags are set so that the NP species is assumed to
    be in equilibrium with the plasma (NP_e_flag = True), which is the same
    configuration used in PotentialModel.compute_bbn().

Outputs: time_temperature_prym_compare.png / .pdf
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import PRyM.PRyM_init as PRyMini
import PRyM.PRyM_main as PRyMmain

from models.m1v0 import M1V0
from models.m2v0 import M2V0
from models.m3v0 import M3V0


def get_t_of_T(model):
    mdl = model(*model.get_initials())
    assert mdl.valid, "Genesys model invalid"

    rho_NP_func = mdl.temp.rho_w
    p_NP_func = mdl.temp.p_w
    drho_dT_func = mdl.temp.drho_dt_w

    PRyMini.NP_e_flag = True

    PRyMini.NP_thermo_flag = False
    PRyMini.xi_NP = 1.0
    prym_NP = PRyMmain.PRyMclass(rho_NP_func, p_NP_func, drho_dT_func)
    t_NP = prym_NP.t_of_T(T_grid)
    return t_NP


def find_optimal_zoom_interval(T_grid, t_GR, t_NP2, t_NP3):
    """
    Find the optimal interval where M2V0 and M3V0 merge and become indistinguishable.
    Automatically calculates appropriate width based on where these models converge.

    Parameters:
    -----------
    T_grid : array
        Temperature grid
    t_GR, t_NP2, t_NP3 : arrays
        Time arrays for GR and NP models

    Returns:
    --------
    T_center : float
        Center temperature for the zoom window
    T_min, T_max : float
        Temperature bounds for the zoom window
    """

    rel_diff_m2_m3 = np.abs(t_NP2 - t_NP3) / ((t_NP2 + t_NP3) / 2)

    bbn_mask = (T_grid >= 0.1) & (T_grid <= 10.0)
    bbn_indices = np.where(bbn_mask)[0]

    if len(bbn_indices) == 0:
        T_center = 1.0
        min_diff_idx_global = len(T_grid) // 2
    else:
        min_diff_idx_local = np.argmin(rel_diff_m2_m3[bbn_mask])
        min_diff_idx_global = bbn_indices[min_diff_idx_local]
        T_center = T_grid[min_diff_idx_global]

    log_center = np.log10(T_center)
    log_width_fixed = 0.15
    T_min = 10 ** (log_center - log_width_fixed / 2)
    T_max = 10 ** (log_center + log_width_fixed / 2)

    T_min = max(T_min, T_grid.min())
    T_max = min(T_max, T_grid.max())

    log_width_final = np.log10(T_max) - np.log10(T_min)
    linear_width = T_max - T_min
    m2_m3_diff_at_center = rel_diff_m2_m3[min_diff_idx_global]

    print(
        f"M2V0/M3V0 merge point: T = {T_min:.4f} to {T_max:.4f} MeV (center: {T_center:.4f} MeV)"
    )
    print(
        f"Log width: {log_width_final:.3f} decades, Linear width: {linear_width:.4f} MeV"
    )
    print(f"M2V0/M3V0 relative difference at center: {m2_m3_diff_at_center:.2e}")

    return T_center, T_min, T_max


T_min, T_max = 1e-4, 10.0
T_grid = np.logspace(np.log10(T_min), np.log10(T_max), 60000)


prym_GR = PRyMmain.PRyMclass()

t_GR = prym_GR.t_of_T(T_grid)
t_NP1 = get_t_of_T(M1V0)
t_NP2 = get_t_of_T(M2V0)
t_NP3 = get_t_of_T(M3V0)


fig, ax_main = plt.subplots(1, 1, figsize=(8, 6))


ax_main.loglog(T_grid, t_GR, "k-", lw=2.2, label="PRyM GR")
ax_main.loglog(T_grid, t_NP1, "r-", lw=2.2, label="PRyM w/ NP (M1V0)")
ax_main.loglog(T_grid, t_NP2, "b-", lw=2.2, label="PRyM w/ NP (M2V0)")
ax_main.loglog(T_grid, t_NP3, "g-", lw=2.2, label="PRyM w/ NP (M3V0)")
ax_main.set_xlabel("Temperature  T  [MeV]")
ax_main.set_ylabel("Time  t  [s]")
ax_main.set_title("t(T) GR vs Weyl models")
ax_main.grid(True, which="both", ls=":", alpha=0.3)

ax_main.legend(loc="upper right")


T_freezeout = 0.5
T_weak = 1.0
T_neutrino = 2.0
T_start_bbn = 10.0

bbn_events = [
    (T_freezeout, "BBN\nfreeze-out", "orange"),
    (T_weak, "Weak\nfreeze-out", "purple"),
    (T_neutrino, "Neutrino\ndecoupling", "brown"),
    (T_start_bbn, "BBN\nstart", "darkred"),
]

for T_event, label, color in bbn_events:
    ax_main.axvline(T_event, color=color, linestyle="--", alpha=0.7, linewidth=1.5)

    y_pos = 1e2 if T_event < 2 else 1e-1
    ax_main.text(
        T_event * 1.1,
        y_pos,
        label,
        fontsize=8,
        alpha=1,
        color=color,
        rotation=0,
        verticalalignment="center",
    )


from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ax_inset = inset_axes(
    ax_main,
    width="44%",
    height="44%",
    loc="lower left",
    bbox_to_anchor=(0.02, 0.02, 1, 1),
    bbox_transform=ax_main.transAxes,
)


zoom_T_min = 1e-4
zoom_T_max = 1e-4 * 1.1
T_center = (zoom_T_min + zoom_T_max) / 2

print(
    f"Preset zoom window: T = {zoom_T_min:.4f} to {zoom_T_max:.4f} MeV (center: {T_center:.4f} MeV)"
)
print(f"Linear width: {zoom_T_max - zoom_T_min:.4f} MeV")


T_zoom_mask = (T_grid >= zoom_T_min) & (T_grid <= zoom_T_max)
T_zoom = T_grid[T_zoom_mask]
t_GR_zoom = t_GR[T_zoom_mask]
t_NP2_zoom = t_NP2[T_zoom_mask]
t_NP3_zoom = t_NP3[T_zoom_mask]

if len(T_zoom) > 0:
    rel_diff_2 = (t_NP2_zoom - t_GR_zoom) / t_GR_zoom * 100
    rel_diff_3 = (t_NP3_zoom - t_GR_zoom) / t_GR_zoom * 100

    ax_inset.semilogx(T_zoom, rel_diff_2, "b-", lw=2)
    ax_inset.semilogx(T_zoom, rel_diff_3, "g-", lw=2)
    ax_inset.axhline(0, color="k", linestyle="--", alpha=0.5, lw=1)

    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.set_xticklabels([])
    ax_inset.set_yticklabels([])
    ax_inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax_inset.xaxis.set_visible(False)
    ax_inset.yaxis.set_visible(False)
    ax_inset.grid(True, alpha=0.3)

    if len(T_zoom) > 0:
        y_min, y_max = ax_inset.get_ylim()
        fake_axis_y = y_min - (y_max - y_min) * 0.1

        ax_inset.plot(
            [T_zoom[0], T_zoom[-1]],
            [fake_axis_y, fake_axis_y],
            "k-",
            linewidth=0.8,
            alpha=1,
        )

        tick_positions = [T_zoom[0], T_zoom[len(T_zoom) // 2], T_zoom[-1]]
        tick_labels = [f"{t:.2e}" for t in tick_positions]

        for i, (tick_x, label) in enumerate(zip(tick_positions, tick_labels)):
            tick_height = (y_max - y_min) * 0.02
            ax_inset.plot(
                [tick_x, tick_x],
                [fake_axis_y, fake_axis_y + tick_height],
                "k-",
                linewidth=0.8,
                alpha=1,
            )

            ax_inset.text(
                tick_x,
                fake_axis_y - (y_max - y_min) * 0.02,
                label,
                ha="center",
                va="top",
                fontsize=8,
                alpha=1,
            )

    ax_inset_zoom = inset_axes(
        ax_inset,
        width="50%",
        height="50%",
        loc="center right",
        bbox_to_anchor=(0.02, 0.1, 1, 1),
        bbox_transform=ax_inset.transAxes,
    )

    if len(T_zoom) > 0:
        t_GR_zoom_rel = (t_GR_zoom - t_GR_zoom) / t_GR_zoom * 100
        t_NP2_zoom_rel = (t_NP2_zoom - t_GR_zoom) / t_GR_zoom * 100

        ax_inset_zoom.semilogx(T_zoom, t_GR_zoom_rel, "k-", lw=2, alpha=0.8)
        ax_inset_zoom.semilogx(T_zoom, t_NP2_zoom_rel, "b-", lw=2, alpha=0.8)
        ax_inset_zoom.axhline(0, color="k", linestyle="--", alpha=0.3, lw=1)

        ax_inset_zoom.set_xticks([])
        ax_inset_zoom.set_yticks([])
        ax_inset_zoom.set_xticklabels([])
        ax_inset_zoom.set_yticklabels([])
        ax_inset_zoom.tick_params(
            left=False, bottom=False, labelleft=False, labelbottom=False
        )
        ax_inset_zoom.xaxis.set_visible(False)
        ax_inset_zoom.yaxis.set_visible(False)
        ax_inset_zoom.grid(True, alpha=0.2)

        y_min_nested, y_max_nested = ax_inset_zoom.get_ylim()
        fake_axis_y_nested = y_min_nested - (y_max_nested - y_min_nested) * 0.15

        ax_inset_zoom.plot(
            [T_zoom[0], T_zoom[-1]],
            [fake_axis_y_nested, fake_axis_y_nested],
            "k-",
            linewidth=0.6,
            alpha=0.6,
        )

        tick_positions_nested = [T_zoom[0], T_zoom[len(T_zoom) // 2], T_zoom[-1]]
        tick_labels_nested = [f"{t:.2e}" for t in tick_positions_nested]

        for i, (tick_x, label) in enumerate(
            zip(tick_positions_nested, tick_labels_nested)
        ):
            tick_height_nested = (y_max_nested - y_min_nested) * 0.03
            ax_inset_zoom.plot(
                [tick_x, tick_x],
                [fake_axis_y_nested, fake_axis_y_nested + tick_height_nested],
                "k-",
                linewidth=0.7,
                alpha=1,
            )

            ax_inset_zoom.text(
                tick_x,
                fake_axis_y_nested - (y_max_nested - y_min_nested) * 0.03,
                label,
                ha="center",
                va="top",
                fontsize=8,
                alpha=1,
            )

        for spine in ax_inset_zoom.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(1)
            spine.set_alpha(0.6)

        nested_zoom_y_min = (
            np.min([np.min(t_GR_zoom_rel), np.min(t_NP2_zoom_rel)]) * 0.95
        )
        nested_zoom_y_max = (
            np.max([np.max(t_GR_zoom_rel), np.max(t_NP2_zoom_rel)]) * 1.05
        )

        from matplotlib.patches import ConnectionPatch

        con_nested_1 = ConnectionPatch(
            (np.max(T_zoom), nested_zoom_y_max),
            (0, 1),
            "data",
            "axes fraction",
            axesA=ax_inset,
            axesB=ax_inset_zoom,
            color="red",
            linestyle=":",
            alpha=0.5,
            linewidth=1,
        )
        con_nested_2 = ConnectionPatch(
            (np.max(T_zoom), nested_zoom_y_min),
            (0, 0),
            "data",
            "axes fraction",
            axesA=ax_inset,
            axesB=ax_inset_zoom,
            color="red",
            linestyle=":",
            alpha=0.5,
            linewidth=1,
        )
        ax_inset.add_artist(con_nested_1)
        ax_inset.add_artist(con_nested_2)

    zoom_t_min = np.min(t_GR_zoom) * 0.95
    zoom_t_max = np.max(t_GR_zoom) * 1.05
else:
    zoom_T_min, zoom_T_max = 0.5, 2.0
    zoom_t_min = 0.1
    zoom_t_max = 1.0


from matplotlib.patches import Rectangle, ConnectionPatch

rect = Rectangle(
    (zoom_T_min, zoom_t_min),
    zoom_T_max - zoom_T_min,
    zoom_t_max - zoom_t_min,
    linewidth=1.5,
    edgecolor="red",
    facecolor="none",
    linestyle="--",
)
ax_main.add_patch(rect)


con1 = ConnectionPatch(
    (zoom_T_max, zoom_t_max),
    (0, 1),
    "data",
    "axes fraction",
    axesA=ax_main,
    axesB=ax_inset,
    color="red",
    linestyle="--",
    alpha=0.7,
)
con2 = ConnectionPatch(
    (zoom_T_max, zoom_t_min),
    (0, 0),
    "data",
    "axes fraction",
    axesA=ax_main,
    axesB=ax_inset,
    color="red",
    linestyle="--",
    alpha=0.7,
)
ax_main.add_artist(con1)
ax_main.add_artist(con2)

fig.tight_layout()

png = Path("time_temperature_prym_compare_m1.png")
pdf = Path("time_temperature_prym_compare_m1.eps")

fig.savefig(png, dpi=1000)
fig.savefig(pdf)
print(f"✓ saved {png} and {pdf}")

plt.show()
