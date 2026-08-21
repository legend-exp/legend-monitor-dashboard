from __future__ import annotations

from legenddashboard.geds.phy.phy_plots import (
    PlotMeta,
    phy_plot_binned_vsTime,
    phy_plot_dist_histogram,
    phy_plot_vsTime,
    phy_plots_sc_vals_dict,
    phy_plots_types_dict,
    phy_plots_vals_dict,
    phy_pulser_corr_dict,
    phy_unit_vals,
)

__all__ = [
    "PlotMeta",
    "phy_plot_binned_vsTime",
    "phy_plot_dist_histogram",
    "phy_plot_style_dict",
    "phy_plot_vsTime",
    "phy_plots_sc_vals_dict",
    "phy_plots_types_dict",
    "phy_plots_vals_dict",
    "phy_pulser_corr_dict",
    "phy_unit_vals",
]

# style *names* only; dispatch happens in phy_monitoring, which picks the
# v1 or contract-v2 implementation per run
phy_plot_style_dict = {
    "Time": phy_plot_vsTime,
    "Histogram": phy_plot_dist_histogram,
}
