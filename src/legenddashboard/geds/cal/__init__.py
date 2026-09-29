from __future__ import annotations

from legenddashboard.geds.cal.detailed_plots import (
    all_detailed_plots,
    detailed_plots,
    plot_spectrum,
)
from legenddashboard.geds.cal.summary_plots import (
    get_aoe_results,
    plot_alpha,
    plot_aoe_status,
    plot_baseline_stability,
    plot_bls,
    plot_counts,
    plot_energy_residuals,
    plot_energy_resolutions_2614,
    plot_energy_resolutions_Qbb,
    plot_energy_spectra,
    plot_fep_stability_channels2d,
    plot_fft_spectra,
    plot_no_fitted_aoe_slices,
    plot_no_fitted_energy_peaks,
    plot_pulser_stability_channels2d,
    plot_pz_consts,
    plot_status,
)
from legenddashboard.geds.cal.tracking_plots import (
    plot_aoe_cut,
    plot_aoe_mean,
    plot_aoe_sig,
    plot_ctc_const,
    plot_energy,
    plot_energy_res_2614,
    plot_energy_res_Qbb,
    plot_energy_residuals_period,
    plot_tau,
    plot_tracking,
)

__all__ = [
    "all_detailed_plots",
    "detailed_plots",
    "plot_energy_residuals_period",
    "plot_spectrum",
    "plot_tracking",
]

tracking_plots = {
    "Energy Calib. Const.": plot_energy,
    "FWHM Qbb": plot_energy_res_Qbb,
    "FWHM FEP": plot_energy_res_2614,
    "Energy Residuals": plot_energy_residuals_period,
    "A/E Mean": plot_aoe_mean,
    "A/E Cut": plot_aoe_cut,
    "A/E Sigma": plot_aoe_sig,
    "PZ": plot_tau,
    "Alpha": plot_ctc_const,
}

summary_plots = {
    "Detector Status": plot_status,
    "Valid. E": plot_no_fitted_energy_peaks,
    "Energy Spectrum": plot_energy_spectra,
    "FEP Counts": plot_counts,
    "FWHM Qbb": plot_energy_resolutions_Qbb,
    "FWHM FEP": plot_energy_resolutions_2614,
    "Energy Residuals": plot_energy_residuals,
    "A/E Status": plot_aoe_status,
    "Valid. A/E": plot_no_fitted_aoe_slices,
    "A/E SF": get_aoe_results,
    "PZ": plot_pz_consts,
    "CT Alpha": plot_alpha,
    "Baseline Spectrum": plot_bls,
    "FFT Spectrum": plot_fft_spectra,
    "Baseline Stability": plot_baseline_stability,
    "FEP Stability": plot_fep_stability_channels2d,
    "Pulser Stability": plot_pulser_stability_channels2d,
}
