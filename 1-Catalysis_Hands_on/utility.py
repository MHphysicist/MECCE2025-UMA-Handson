# =============================================================================
#  Utility Functions for "Hands-on UMA Model for Catalysis"
# =============================================================================
#  Author: Muhammad H. M. Ahmed
#  Affiliation: Department of Materials Science & Engineering, KFUPM
#  Contact: husseinphysicist@gmail.com | +966 533584744 | g202318650@kfupm.edu.sa
#  LinkedIn: https://www.linkedin.com/m/in/muhammad-h-m-ahmed/
#  Event: 19th MECC 2025 Conference (Practical Introduction to AI in Materials Discovery | UMA Hands-on Session)
#
#  Description:
#  This module provides helper functions for the UMA catalysis tutorial,
#  including:
#   • UMA calculator setup and shared predictor loading
#   • 3D visualization via NGLView
#   • Adsorption site mapping and placement
#   • Constraint handling for slab relaxations
#   • Adsorption energy heatmap plotting
#
#  Developed as part of the MECCE 2025 hands-on session:
#  “Machine Learning–Accelerated Catalysis with UMA and ASE”
# =============================================================================


from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write
from fairchem.core import FAIRChemCalculator, pretrained_mlip
import nglview as nv
from nglview.shape import Shape


def load_uma_calculators(tasks: Iterable[str] = ("omat", "omol", "oc20")) -> Tuple[object, Dict[str, FAIRChemCalculator]]:
    """Return a shared UMA predictor and calculators for the requested tasks."""
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")
    calculators = {task: FAIRChemCalculator(predictor, task_name=task) for task in tasks}
    return predictor, calculators


def view_structure_ngl(
    atoms: Atoms,
    title: str = "Structure",
    style: str = "ball+stick",
    theme: str = "light",
    show_cell: bool = True,
    quality: str = "medium",
    camera: str = "orthographic",
    metalness: float = 0.05,
    roughness: float = 0.0,
    radius_scale: float = 2.5,
    bond_scale: float = 0.1,
    background: str | None = None,
    save_image: bool = False,
    image_transparent: bool = False,
    static_view: bool = True,
    z_up: bool = True,
    side_axis: str = "x",
    show_axes: bool = True,
    axes_at: str = "center",
    axes_scale: float = 0.2,
    axes_radius: float = 0.15,
    axes_label: bool = True,
) -> nv.NGLWidget:
    """Professional 3D visualization using NGLView with sensible defaults."""
    atoms_to_show = atoms.copy() if z_up else atoms

    if z_up:
        if side_axis.lower() == "x":
            atoms_to_show.rotate(90, "x", rotate_cell=True)
        elif side_axis.lower() == "y":
            atoms_to_show.rotate(-90, "y", rotate_cell=True)

    view = nv.show_ase(atoms_to_show)
    bg_color = background if background is not None else ("black" if theme == "dark" else "white")
    view.stage.set_parameters(
        backgroundColor=bg_color,
        clipNear=0,
        clipFar=100,
        clipDist=10,
        fogNear=1000,
        fogFar=2000,
        sampleLevel=1 if quality == "high" else 0,
        opacity=1.0,
    )
    view.camera = camera if camera in {"orthographic", "perspective"} else "orthographic"

    if static_view:
        view._remote_call("setSpin", target="Stage", args=[False])
        view._remote_call("setRock", target="Stage", args=[False])

    rep_params = dict(
        metalness=float(metalness),
        roughness=float(roughness),
        colorScheme="element",
        radiusScale=float(radius_scale),
        opacity=1.0,
    )
    style_map = {
        "ball+stick": ("ball+stick", dict(bondScale=float(bond_scale))),
        "spacefill": ("spacefill", {}),
        "wireframe": ("line", {"linewidth": 2}),
        "points": ("point", {"pointSize": 8}),
    }
    rep_type, extra = style_map.get(style, style_map["ball+stick"])
    view.clear_representations()
    view.add_representation(rep_type, **{**rep_params, **extra})

    if show_cell and any(atoms.get_pbc()):
        view.add_unitcell()

    if show_axes:
        cell = atoms_to_show.get_cell()
        if axes_at == "center" and cell is not None and getattr(cell, "rank", 0) > 0:
            origin = 0.5 * (cell[0] + cell[1] + cell[2])
            length_scale = float(axes_scale) * max(np.linalg.norm(v) for v in cell)
        else:
            origin = np.zeros(3)
            length_scale = 5.0 * float(axes_scale) / 0.2

        shape = Shape(view)
        for axis, color, direction in zip(
            ("x", "y", "z"),
            ([1, 0, 0], [0, 1, 0], [0, 0, 1]),
            (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])),
        ):
            end = origin + length_scale * direction
            shape.add_arrow(origin.tolist(), end.tolist(), color, float(axes_radius))
            if axes_label:
                shape.add_text(end.tolist(), axis.upper(), color, 0.6)

    view.center()
    view.layout.width = "100%"
    view.layout.height = "600px"

    if save_image:
        view.render_image(factor=4, antialias=True, trim=False, transparent=image_transparent)
        image = view.get_image()
        if image is not None:
            filename = f"{title.replace(' ', '_')}_{style}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            with open(filename, "wb") as handle:
                handle.write(image)
            print(f"Saved: {filename}")

    return view


def make_sites_xyz(slab: Atoms, sites_xy: Dict[str, Iterable[float]], height: float = 1.5) -> Dict[str, np.ndarray]:
    """Convert 2D site coordinates to 3D by adding a fixed height above the slab."""
    z_top = slab.positions[:, 2].max()
    return {name: np.array([xy[0], xy[1], z_top + float(height)], float) for name, xy in sites_xy.items()}


def place_adsorbate_at_xyz(slab: Atoms, adsorbate: Atoms, target_xyz: Iterable[float], anchor: str = "com", anchor_index: int = 0) -> Atoms:
    """Place an adsorbate at a target 3D position using its centre of mass or a reference atom."""
    target = np.asarray(target_xyz, float)
    ads = adsorbate.copy()
    reference = ads.get_center_of_mass() if anchor == "com" else ads.positions[int(anchor_index)]
    ads.translate(target - reference)

    combined = slab.copy()
    combined.extend(ads)
    combined.set_pbc([True, True, True])
    combined.wrap()
    return combined


def get_bottom_layer_constraint(slab: Atoms, fix_fraction: float = 0.4) -> Tuple[FixAtoms, np.ndarray]:
    """Fix a fraction of the slab atoms (starting from the bottom) during relaxations."""
    z = slab.positions[:, 2]
    threshold = z.min() + fix_fraction * (z.max() - z.min())
    fixed_indices = np.array([index for index, z_val in enumerate(z) if z_val < threshold], dtype=int)
    return FixAtoms(indices=fixed_indices.tolist()), fixed_indices


def plot_adsorption_heatmap(adsorption_results: Dict[str, Dict[str, Dict[str, float]]], figsize: Tuple[float, float] = (8.8, 6.8)):
    """Create a heatmap summarising adsorption energies by molecule and site."""
    rows = []
    for molecule, site_dict in adsorption_results.items():
        for site, record in site_dict.items():
            rows.append({"molecule": molecule, "site": site, "energy_eV": float(record["energy"])})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No adsorption data to plot.")

    molecule_order = sorted(df["molecule"].unique().tolist())
    site_order = (
        df.groupby("site")["energy_eV"].median().sort_values(ascending=True).index.tolist()
    )

    pivot = df.pivot(index="site", columns="molecule", values="energy_eV").reindex(
        index=site_order, columns=molecule_order
    )

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]), labels=pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]), labels=pivot.index)
    ax.set_xlabel("Adsorbate")
    ax.set_ylabel("Site")
    ax.set_title("Adsorption Energy Heatmap")

    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Energy (eV)")

    values = pivot.values.astype(float)
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    scale = max(1e-12, vmax - vmin)
    normed = (values - vmin) / scale

    for (row, col), value in np.ndenumerate(values):
        text_color = "white" if normed[row, col] < 0.45 else "black"
        ax.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=10, color=text_color)

    plt.tight_layout()
    return fig, ax

def pick_site_by_name(
    results_for_one_mol: Dict[str, Dict[str, Any]],
    site_name: str,
    mol_name: str = ""
) -> Tuple[str, Dict[str, Any]]:
    """Return (site_name, record) for a requested site; raise a clear error if missing."""
    if not results_for_one_mol:
        who = f" for {mol_name}" if mol_name else ""
        raise ValueError(f"No adsorption results found{who}.")

    if site_name not in results_for_one_mol:
        available = ", ".join(sorted(results_for_one_mol.keys()))
        who = f" for {mol_name}" if mol_name else ""
        raise ValueError(
            f'Requested site "{site_name}"{who} not found. '
            f"Available sites: {available}"
        )
    return site_name, results_for_one_mol[site_name]

def plot_neb_energy_diagram(energies, ts_index: int, initial_label: str = "NH₃*", ts_label: str = "TS", final_label: str = "NH₂* + H*", figsize=(4.3, 3.1), ax=None, return_values: bool = True,):
    """
    Plot a compact NEB energy-level diagram (Initial, TS, Final) and
    optionally return Ea(fwd), Ea(rev), and ΔE.
    """
    energies = np.asarray(energies, dtype=float)
    E0  = float(energies[0])
    Ets = float(energies[ts_index])
    Ef  = float(energies[-1])

    Ea_fwd = Ets - E0
    Ea_rev = Ets - Ef
    dE     = Ef  - E0

    # positions and relative energies (relative to Initial)
    x = np.array([0.0, 1.0, 2.0], float)
    y = np.array([0.0, Ea_fwd, dE], float)
    labels = [initial_label, ts_label, final_label]

    # figure/axes
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        created_fig = True
    else:
        fig = ax.figure

    # short horizontal levels
    level_w = 0.36
    for xi, yi in zip(x, y):
        ax.hlines(yi, xi - level_w, xi + level_w, linewidth=2.2, zorder=3)

    # baseline + guides
    ax.axhline(0.0, linewidth=1.0, zorder=1)
    ax.vlines([1.0, 2.0], [0.0, 0.0], [y[1], y[2]], linewidth=0.9, linestyles="--", zorder=2)

    # ticks & labels
    ax.set_xticks(x, labels)
    ax.set_ylabel("Energy relative to Initial (eV)")

    # limits & margins
    rng = max(y.max(), 0.0) - min(y.min(), 0.0)
    pad = 0.18 * (rng if rng > 0 else 1.0)
    ax.set_ylim(min(0.0, y.min()) - pad, max(0.0, y.max()) + pad)
    ax.set_xlim(-0.45, 2.45)

    # annotations
    def _annotate_vertical(delta, x_pos, y0=0.0, label="", x_text_shift=0.06):
        ax.annotate("", xy=(x_pos, y0 + delta), xytext=(x_pos, y0),
                    arrowprops=dict(arrowstyle="<->", linewidth=1.0), zorder=4)
        ax.text(x_pos + x_text_shift, y0 + 0.5 * delta, label,
                va="center", ha="left", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                          boxstyle="round", pad=0.18))

    _annotate_vertical(Ea_fwd, x_pos=0.12, label=f"Ea (fwd) = {Ea_fwd:.3f} eV")
    _annotate_vertical(Ea_rev, x_pos=1.88, y0=dE, label=f"Ea (rev) = {Ea_rev:.3f} eV")

    ax.annotate("", xy=(1.5, dE), xytext=(1.5, 0.0),
                arrowprops=dict(arrowstyle="<->", linewidth=1.0), zorder=4)
    ax.text(1.56, 0.5 * dE, f"ΔE = {dE:+.3f} eV",
            va="center", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                      boxstyle="round", pad=0.18))

    ax.set_title("NEB Energy Diagram", pad=8)

    # tidy spines & light y-grid
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
    ax.grid(axis="y", linewidth=0.6, alpha=0.28)

    if created_fig:
        plt.show()

    values = dict(E0=E0, Ets=Ets, Ef=Ef, Ea_fwd=Ea_fwd, Ea_rev=Ea_rev, dE=dE)
    return (fig, ax, values) if return_values else (fig, ax)