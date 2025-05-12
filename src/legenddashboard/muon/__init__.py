from __future__ import annotations

from legenddashboard.muon_plots import (
    muon_plot_calshift,
    muon_plot_intlight,
    muon_plot_ratesFloor,
    muon_plot_ratesPillBox,
    muon_plot_ratesWall,
    muon_plot_spectra,
    muon_plot_spp,
    muon_plot_totalRates_daily,
    muon_plot_totalRates_hourly,
)

muon_plots_cal_dict = {
    "Cal. Spectra": muon_plot_spectra,
    "Cal. SPP Sigma": muon_plot_spp,
    "Cal. SPP Shift": muon_plot_calshift,
}


muon_plots_mon_dict = {
    "Integral Light": muon_plot_intlight,
    "Total Rates/H": muon_plot_totalRates_hourly,
    "Total Rates/D": muon_plot_totalRates_daily,
    "Pillbox Rates": muon_plot_ratesPillBox,
    "Floor Rates": muon_plot_ratesFloor,
    "Wall Rates": muon_plot_ratesWall,
}
