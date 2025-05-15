from __future__ import annotations

from legenddashboard.geds.phy.phy_plots import (
    phy_plot_histogram,
    phy_plot_vsTime,
    phy_plots_sc_vals_dict,
    phy_plots_types_dict,
    phy_plots_vals_dict,
    phy_resampled_vals,
    phy_unit_vals,
)

__all__ = [
    "phy_plot_histogram",
    "phy_plot_vsTime",
    "phy_plots_sc_vals_dict",
    "phy_plots_types_dict",
    "phy_plots_vals_dict",
    "phy_resampled_vals",
    "phy_unit_vals",
]

phy_plot_style_dict = {
    "Time": phy_plot_vsTime,
    "Histogram": phy_plot_histogram,
}
