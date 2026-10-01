from __future__ import annotations

import argparse
import io
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import panel as pn
import param
from matplotlib.figure import Figure

import legenddashboard.geds.string_visulization as visu
from legenddashboard.geds import cal
from legenddashboard.geds.cal import native_dsp_plots as dspp
from legenddashboard.geds.cal import native_plots
from legenddashboard.geds.cal import native_psd_plots as psd
from legenddashboard.geds.cal.plot_data import (
    CommonData,
    data_keys,
    plt_data_path,
    read_group,
)
from legenddashboard.geds.cal.shelf_cache import (
    _stat_key,
    render_png,
    shelf_data_many,
    shelf_entry,
    shelf_keys,
)
from legenddashboard.geds.ged_monitoring import GedMonitoring
from legenddashboard.util import (
    get_par_cache,
    load_run_pars,
    logo_path,
    read_config,
    sorter,
)

log = logging.getLogger(__name__)

# calibration plots
plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = (16, 6)
plt.rcParams["figure.dpi"] = 100


class CalMonitoring(GedMonitoring):
    cached_data = param.Dict(default=None)
    tmp_path = param.String("/tmp/")
    plot_type_tracking = param.ObjectSelector(
        default=list(cal.tracking_plots)[1],
        objects=list(cal.tracking_plots),
    )

    parameter = param.ObjectSelector(
        default=next(iter(cal.all_detailed_plots)), objects=list(cal.all_detailed_plots)
    )

    plot_type_details = param.ObjectSelector(
        default=cal.detailed_plots[0], objects=cal.detailed_plots
    )
    plot_type_details_objects = param.List(default=cal.detailed_plots)
    channel_objects = param.List(default=[])
    psd_set = param.Parameter(default=None)  # A/E or LQ parameter set, e.g. "dplms"
    psd_set_objects = param.List(default=[])

    plot_type_summary = param.ObjectSelector(
        default=list(cal.summary_plots)[3],
        objects=list(cal.summary_plots),
    )
    # Options must be keys of cal.summary_plots (functions that support
    # ``download=True``).
    plot_types_download = param.Selector(
        objects=["FWHM Qbb", "FWHM FEP", "A/E SF", "PZ", "CT Alpha"],
        default="FWHM Qbb",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Shared, bounded cache of parsed parameter files; reused across all
        # user sessions so the high-latency filesystem is read at most once per
        # run and memory stays bounded (see legenddashboard.util.get_par_cache).
        self.cached_data = get_par_cache()
        # The view_*/download_* methods are re-rendered by Panel through their
        # @param.depends decorators (they are placed directly in the panes);
        # additional param.watch registrations would compute every plot twice
        # per interaction. Only genuine state updaters are watchers; they are
        # registered before any view pane (same precedence -> registration
        # order -> shelves are loaded before the views that show them render)
        # and before the initial update_plot_dict call.
        self.param.watch(self.update_plot_dict, ["run_dict", "run"])
        self.update_plot_dict()

    @param.depends("run_dict", "run", "sort_by", "plot_types_download")
    def download_summary_files(self, event=None):  # noqa: ARG002
        start_time = time.time()
        try:
            download_file, download_filename = cal.summary_plots[
                self.plot_types_download
            ](
                self.prod_config,
                self.run,
                self.run_dict[self.run],
                self.base_path,
                self.period,
                key=self.sort_by,
                download=True,
                sort_dets_obj=self.sort_obj,
                cache_data=self.cached_data,
            )
            # Serialise on click, in memory. The filename is derived only from
            # experiment/period/run/plot type, so every session viewing the
            # same run resolved to the *same* path on disk: concurrent renders
            # truncated and rewrote the file while another session's download
            # was streaming it. Keeping the CSV per-session also means it can
            # never be served stale.
            ret = pn.widgets.FileDownload(
                callback=lambda df=download_file: io.StringIO(df.to_csv(index=False)),
                filename=download_filename,
                button_type="success",
                embed=False,
                name="Click to download 'csv'",
                width=350,
            )
        except Exception:
            log.exception("Failed to build summary download for %s", self.run)
            ret = pn.widgets.FileDownload(
                None,
                filename="temp",
                button_type="success",
                embed=False,
                name="Click to download 'csv'",
                width=350,
            )
        log.debug("Time to download summary files: %.3fs", time.time() - start_time)
        return ret

    @param.depends("run_dict", "run", "sort_by", "string", "plot_type_summary")
    def view_summary(self, event=None):  # noqa: ARG002
        start_time = time.time()
        figure = None
        try:
            if not self.cached_data:
                self.cached_data = get_par_cache()
            if self.plot_type_summary in [
                "FWHM Qbb",
                "FWHM FEP",
                "Energy Residuals",
                "A/E Status",
                "PZ",
                "CT Alpha",
                "Valid. E",
                "Valid. A/E",
                "A/E SF",
            ]:
                figure = cal.summary_plots[self.plot_type_summary](
                    self.prod_config,
                    self.run,
                    self.run_dict[self.run],
                    self.base_path,
                    self.period,
                    key=self.sort_by,
                    sort_dets_obj=self.sort_obj,
                    cache_data=self.cached_data,
                )

            elif self.plot_type_summary in ["Detector Status", "FEP Counts"]:
                # elif self.plot_type_summary in ["Detector Status"]:
                strings_dict, meta_visu_chan_dict, meta_visu_channel_map = sorter(
                    self.base_path,
                    self.run_dict[self.run]["timestamp"],
                    key="String",
                    sort_dets_obj=self.sort_obj,
                )
                meta_visu_source, meta_visu_xlabels = visu.get_plot_source_and_xlabels(
                    meta_visu_chan_dict, meta_visu_channel_map, strings_dict
                )
                # self.meta_visu_chan_dict, self.meta_visu_channel_map = chan_dict, channel_map
                figure = cal.summary_plots[self.plot_type_summary](
                    self.prod_config,
                    self.run,
                    self.run_dict[self.run],
                    self.base_path,
                    meta_visu_source,
                    meta_visu_xlabels,
                    self.period,
                    key=self.sort_by,
                    sort_dets_obj=self.sort_obj,
                    cache_data=self.cached_data,
                )
            elif self.plot_type_summary in [
                "Baseline Spectrum",
                "Energy Spectrum",
                "Baseline Stability",
                "FEP Stability",
                "Pulser Stability",
            ]:
                figure = cal.summary_plots[self.plot_type_summary](
                    self.prod_config,
                    self.common_dict,
                    self.channel_map,
                    self.strings_dict[self.string],
                    self.string,
                    self.run,
                    self.period,
                    self.run_dict[self.run],
                    key=self.sort_by,
                    sort_dets_obj=self.sort_obj,
                    cache_data=self.cached_data,
                )
            elif self.plot_type_summary == "FFT Spectrum":
                figure = cal.summary_plots[self.plot_type_summary](
                    self.prod_config,
                    self._string_ffts(),
                    self.channel_map,
                    self.strings_dict[self.string],
                    self.string,
                    self.run,
                    self.period,
                    self.run_dict[self.run],
                    key=self.sort_by,
                    sort_dets_obj=self.sort_obj,
                    cache_data=self.cached_data,
                )
            else:
                figure = Figure()
        except Exception:
            log.exception(
                "Failed to build summary plot '%s' for %s",
                self.plot_type_summary,
                self.run,
            )
        log.debug("Time to get summary plot: %.3fs", time.time() - start_time)
        return figure

    @param.depends("run_dict", "date_range", "sort_by", "string", "plot_type_tracking")
    def view_tracking(self, event=None):  # noqa: ARG002
        figure = None
        try:
            if self.plot_type_tracking != "Energy Residuals":
                figure = cal.plot_tracking(
                    self._get_run_dict(),
                    self.base_path,
                    cal.tracking_plots[self.plot_type_tracking],
                    self.string,
                    self.period,
                    self.plot_type_tracking,
                    key=self.sort_by,
                    cache_data=self.cached_data,
                    sort_dets_obj=self.sort_obj,
                )
            else:
                figure = cal.plot_energy_residuals_period(
                    self._get_run_dict(),
                    self.base_path,
                    self.period,
                    key=self.sort_by,
                    cache_data=self.cached_data,
                    sort_dets_obj=self.sort_obj,
                )
        except Exception:
            log.exception(
                "Failed to build tracking plot '%s' for %s %s",
                self.plot_type_tracking,
                self.period,
                self.string,
            )
        return figure

    def update_plot_dict(self, *events):  # noqa: ARG002
        start_time = time.time()
        run_info = self.run_dict[self.run]
        file_stem = (
            f"{run_info['experiment']}-{self.period}-{self.run}-cal-"
            f"{run_info['timestamp']}"
        )
        plt_base = Path(self.prod_config["paths"]["plt"])
        self.plot_dict = (
            plt_base / f"hit/cal/{self.period}/{self.run}" / f"{file_stem}-plt_hit"
        )
        # Build the dsp path explicitly rather than str.replace("hit", "dsp"),
        # which would corrupt any other "hit" substring in the deployment path.
        self.dsp_plot_dict = (
            plt_base / f"dsp/cal/{self.period}/{self.run}" / f"{file_stem}-plt_dsp"
        )
        self.dsp_plt_data = plt_data_path(self.dsp_plot_dict)

        self.plt_data = plt_data_path(self.plot_dict)
        if self.plt_data.exists():
            channels = data_keys(self.plt_data)
        else:
            channels = shelf_keys(self.plot_dict)
        if "common" in channels:
            channels.remove("common")
        if not channels:
            msg = f"No channels found in plot file {self.plot_dict}"
            raise RuntimeError(msg)
        self.strings_dict, self.chan_dict, self.channel_map = sorter(
            self.base_path,
            run_info["timestamp"],
            "String",
            sort_dets_obj=self.sort_obj,
        )

        self.channel_objects = channels
        if self.channel not in channels:  # keep the user's channel when valid
            self.channel = channels[0]

        self.update_strings()
        log.debug("Time to update plot dict: %.3fs", time.time() - start_time)

    # Shelve contents are loaded lazily through the process-wide cache, so a
    # run switch costs a key listing and only the views actually shown pay
    # for unpickling (once per file version, shared by all sessions).
    @property
    def common_dict(self):
        if self.plt_data.exists():
            return CommonData(self.plt_data)
        return shelf_entry(self.plot_dict, "common")

    @property
    def plot_dict_ch(self):
        return shelf_entry(self.plot_dict, self.channel[:9])

    @property
    def dsp_dict(self):
        return shelf_entry(self.dsp_plot_dict, self.channel[:9])

    def _det_data(self, *keys):
        """Plot data of the current channel from the LH5, or None without it."""
        return read_group(self.plt_data, self.channel[:9], *keys)

    def _ecal_data(self, parameter):
        """``ecal[parameter]`` plot data, from the LH5 when present."""
        data = self._det_data("ecal", parameter)
        return data if data is not None else self.plot_dict_ch["ecal"][parameter]

    def _dsp_data(self, *keys):
        """DSP plot data of the current channel from the lh5, or None without it."""
        return read_group(self.dsp_plt_data, self.channel[:9], *keys)

    def _string_ffts(self):
        """Noise PSDs of the current string, from the dsp plot lh5 or the shelf."""
        dets = self.strings_dict[self.string]
        keys = ("noise_optimisation", "nopt", "fft")
        if not self.dsp_plt_data.exists():
            return shelf_data_many(self.dsp_plot_dict, dets, keys)
        ffts = {det: read_group(self.dsp_plt_data, det, *keys) for det in dets}
        return {det: fft for det, fft in ffts.items() if fft is not None}

    def _view_dsp(self):
        """Native PZ / optimisation / noise / DPLMS plot, or None to use the PNG."""
        plot = self.plot_type_details
        title = f"{self.channel[:9]} | {self.parameter} | {plot}"
        if self.parameter == "PZ":
            if plot.endswith("slope"):
                data = self._dsp_data("pz", f"{plot}_data")
                return None if data is None else dspp.plot_slopes(data, title)
            wfs = self._dsp_data("pz", "waveforms_data")
            if wfs is None:
                return None
            if plot == "waveforms_zoomed":  # same waveforms, zoom window only
                wfs = {**wfs, **(self._dsp_data("pz", "waveforms_zoomed_data") or {})}
            return dspp.plot_waveforms(wfs, title, ylabel="Normalised ADU")
        if self.parameter == "Optimisation":
            filt, kind = plot.split("_")
            data = self._dsp_data(f"{filt}_optimisation", "data")
            if data is None:
                return None
            return dspp.plot_optimiser(
                data, title, "kernel" if kind == "kernel" else "acq"
            )
        if self.parameter == "Noise":
            if plot == "fft":
                fft = self._dsp_data("noise_optimisation", "nopt", "fft")
                return None if fft is None else dspp.plot_fft(fft, title)
            filt, kind = plot.split("_")
            data = self._dsp_data("noise_optimisation", "nopt", filt, f"{kind}_data")
            if data is None:
                return None
            if kind == "optimization":
                return dspp.plot_nopt_optimization(data, title, filt)
            return dspp.plot_nopt_distributions(data, title)
        if self.parameter == "DPLMS":
            if plot == "filter":
                coeffs = self._dsp_data("dplms", "coefficients")
                return (
                    None
                    if coeffs is None
                    else dspp.plot_dplms_filter({"coefficients": coeffs}, title)
                )
            data = self._dsp_data("dplms", f"{plot}_data")
            if data is None:
                return None
            if plot == "wf_sel":
                return dspp.plot_dplms_selection(data, title)
            return dspp.plot_waveforms(data, title)
        return None

    def _dsp_figure(self):
        """Pickled figure of a DSP plot from the dsp shelf."""
        dsp, plot = self.dsp_dict, self.plot_type_details
        if self.parameter == "PZ":
            return dsp["pz"][plot]
        if self.parameter == "Optimisation":
            filt, kind = plot.split("_")
            return dsp[f"{filt}_optimisation"][f"{kind}_space"]
        if self.parameter == "Noise":
            nopt = dsp["noise_optimisation"]["nopt"]
            if plot == "fft":
                return nopt["fft"]["fig"]
            filt, kind = plot.split("_")
            return nopt[filt][kind]
        return dsp["dplms"][plot]

    def _det_pars(self):
        pars = load_run_pars(
            self.prod_config,
            "hit",
            self.period,
            self.run,
            self.run_dict[self.run],
            self.cached_data,
        )
        return pars[self.channel[:9]]

    def _section_figure(self, section):
        """Pickled figure of an aoe/lq plot, flat or nested ``<name>`` layout."""
        sec = self.plot_dict_ch[section]
        if self.plot_type_details in sec:
            return sec[self.plot_type_details]
        chosen = sec.get(self.psd_set)
        if isinstance(chosen, dict) and self.plot_type_details in chosen:
            return chosen[self.plot_type_details]
        for sub in sec.values():
            if isinstance(sub, dict) and self.plot_type_details in sub:
                return sub[self.plot_type_details]
        raise KeyError(self.plot_type_details)

    def _psd_results(self, section):
        """``(set name, results)`` of the selected A/E / LQ set; name None if flat."""
        res = self._det_pars()["results"].get(section, {})
        if "cal_energy_param" in res:  # pre-``params:`` layout, a single set
            return None, res
        name = self.psd_set if self.psd_set in res else next(iter(sorted(res)), None)
        return name, res.get(name, {})

    @param.depends("run", "channel", "parameter", watch=True)
    def update_psd_sets(self):
        if self.parameter not in {"A/E", "LQ"}:
            return
        try:
            res = self._det_pars()["results"].get(
                "aoe" if self.parameter == "A/E" else "lq", {}
            )
        except Exception:
            res = {}
        sets = [] if "cal_energy_param" in res else sorted(res)
        self.psd_set_objects = sets
        if self.psd_set not in sets:
            self.psd_set = sets[0] if sets else None

    def _view_psd(self):
        """Native A/E or LQ plot, or None to use the shelf PNG."""
        section = "aoe" if self.parameter == "A/E" else "lq"
        name, res = self._psd_results(section)
        plot = self.plot_type_details
        title = f"{self.channel[:9]} | {self.parameter} {name or ''} | {plot}"
        if plot == "mean_time":
            return psd.plot_time_series(
                res.get("1000-1300keV"), title, "mean", "A/E mean"
            )
        if plot == "stability":
            return psd.plot_time_series(
                res.get("DEP_means"), title, "mean", "LQ DEP mean"
            )
        keys = (section,) if name is None else (section, name)
        data = self._det_data(*keys, f"{plot}_data")
        if data is None:
            return None
        if section == "aoe":
            builders = {
                "spectrum": lambda: psd.plot_spectra(data, title, psd.AOE_SPECTRA),
                "sf_v_energy": lambda: psd.plot_sf_vs_energy(data, title),
                "classifier": lambda: psd.plot_classifier(
                    data, title, "A/E classifier"
                ),
                "plot_dt_dep": lambda: psd.plot_dt_dep(data, title),
                "compt_bands_uncorrected": lambda: psd.plot_compt_bands(
                    data, title, "A/E"
                ),
                "compt_bands_corrected": lambda: psd.plot_compt_bands(
                    data, title, "A/E"
                ),
                "mean_fit": lambda: psd.plot_energy_corr(
                    data, res.get("correction_fit_results"), title, "mean"
                ),
                "sigma_fit": lambda: psd.plot_energy_corr(
                    data, res.get("correction_fit_results"), title, "sigma"
                ),
                "cut_fit": lambda: psd.plot_cut_fit(data, title),
                "survival_fractions": lambda: psd.plot_survival_curves(data, title),
            }
        else:
            builders = {
                "spectrum": lambda: psd.plot_spectra(data, title, psd.LQ_SPECTRA),
                "sf_v_energy": lambda: psd.plot_sf_vs_energy(data, title),
                "classifier": lambda: psd.plot_classifier(data, title, "LQ classifier"),
                "survival_fractions": lambda: psd.plot_survival_curves(
                    data, title, cut_key="cut_val"
                ),
                "cut_fit": lambda: psd.plot_lq_cut_fit(
                    data, res.get("cut_fit_pars"), title
                ),
                "drift_time": lambda: psd.plot_lq_drift_time(
                    data, res.get("rt_correction"), title
                ),
            }
        build = builders.get(plot)
        return build() if build is not None else None

    def _view_energy(self):
        """Native plot for an energy parameter, or None to use the shelf PNG."""
        param_name, plot = self.parameter, self.plot_type_details
        title = f"{self.channel[:9]} | {param_name} | {plot}"
        if plot in {"cal_fit", "fwhm_fit", "peak_fits"}:
            det = self._det_pars()
            fits = det["results"]["ecal"][param_name]
            cal_op = det["pars"]["operations"][param_name]
            if plot == "cal_fit":
                return native_plots.plot_cal_fit(fits["pk_fits"], cal_op, title)
            if plot == "fwhm_fit":
                return native_plots.plot_fwhm_fit(
                    fits["pk_fits"],
                    fits.get("eres_linear"),
                    fits.get("eres_quadratic"),
                    title,
                )
            hists = self._det_data("ecal", param_name, "peak_hists")
            if hists is None:
                return None
            return native_plots.plot_peak_fits(hists, fits["pk_fits"], cal_op, title)
        if plot in {"2614_timemap", "pulser_timemap"}:
            hist = self._det_data("ecal", param_name, f"{plot}_data")
            if hist is None:
                return None
            return native_plots.plot_timemap(hist, title, "Energy (keV)")
        ecal = self._ecal_data(param_name)
        if plot in {"spectrum", "logged_spectrum"}:
            return cal.plot_spectrum(
                ecal["spectrum"], self.channel, log=plot != "spectrum"
            )
        if plot == "survival_frac":
            return native_plots.plot_survival_frac(ecal["survival_frac"], title)
        if plot == "cut_spectrum":
            return native_plots.plot_cut_spectra(ecal["spectrum"], title)
        if plot == "peak_track":
            return native_plots.plot_peak_track(ecal, title)
        return None

    def _png_pane(self, get_figure):
        """Rasterise a cached (shared) figure once; serve PNG bytes after."""
        # fingerprint the shelf (path+mtime+size) so a regenerated shelf drops
        # the stale PNG, and key on the channel id the shelve lookup uses
        key = (
            *_stat_key(self.plot_dict),
            self.channel[:9],
            self.parameter,
            self.plot_type_details,
            self.psd_set,
        )
        return pn.pane.PNG(
            io.BytesIO(render_png(key, get_figure)), sizing_mode="scale_width"
        )

    def _qc_plots(self):
        """QC cut names of the current channel (lh5 plot data, else the shelf)."""
        qc = self._det_data("qc")
        if qc is not None:
            return sorted(k[: -len("_data")] for k in qc if k.endswith("_data"))
        try:
            return sorted(self.plot_dict_ch["qc"])
        except Exception:
            return []

    @param.depends("channel", watch=True)
    def update_qc_plots(self):
        if self.parameter == "QC":  # cut names can change with the detector
            self.update_plot_type_details()

    @param.depends("parameter", watch=True)
    def update_plot_type_details(self):
        start_time = time.time()
        if self.parameter == "QC":
            plots = self._qc_plots() or ["none"]
        else:
            plots = cal.all_detailed_plots[self.parameter]
        self.param.plot_type_details.objects = plots  # else the selector rejects them
        self.plot_type_details_objects = plots
        self.plot_type_details = plots[0]
        log.debug("Time to update plot type details: %.3fs", time.time() - start_time)

    @param.depends(
        "run_dict", "run", "channel", "parameter", "plot_type_details", "psd_set"
    )
    def view_details(self, event=None):  # noqa: ARG002
        fig_pane = pn.pane.Matplotlib(Figure(), sizing_mode="scale_width")
        try:
            if self.parameter in {"A/E", "LQ"}:
                section = "aoe" if self.parameter == "A/E" else "lq"
                native = self._view_psd()
                fig_pane = (
                    native
                    if native is not None
                    else (self._png_pane(lambda: self._section_figure(section)))
                )
            elif self.parameter == "Baseline":
                hist = self._det_data("ecal", f"{self.plot_type_details}_data")
                if hist is not None:
                    fig_pane = native_plots.plot_timemap(
                        hist, f"{self.channel[:9]} | baseline", "Baseline (ADC)"
                    )
                else:
                    fig_pane = self._png_pane(
                        lambda: self.plot_dict_ch["ecal"][self.plot_type_details]
                    )
            elif self.parameter == "QC":
                plot = self.plot_type_details
                data = self._det_data("qc", f"{plot}_data")
                if data is not None:
                    fig_pane = native_plots.plot_qc_cut(
                        data, f"{self.channel[:9]} | QC | {plot}"
                    )
                else:
                    fig_pane = self._png_pane(lambda: self.plot_dict_ch["qc"][plot])
            elif self.parameter in {"PZ", "Optimisation", "Noise", "DPLMS"}:
                native = self._view_dsp()
                fig_pane = (
                    native if native is not None else self._png_pane(self._dsp_figure)
                )
            else:
                native = self._view_energy()
                fig_pane = (
                    native
                    if native is not None
                    else self._png_pane(
                        lambda: self.plot_dict_ch["ecal"][self.parameter][
                            self.plot_type_details
                        ]
                    )
                )
        except Exception:
            log.exception(
                "Failed to build detailed plot '%s'/'%s' for channel %s",
                self.parameter,
                self.plot_type_details,
                self.channel,
            )
        return fig_pane

    def build_detailed_pane(self, widget_widths: int = 140):
        details_ch_param = pn.widgets.Select(
            value=self.param.channel,
            options=self.param.channel_objects,
            width=widget_widths,
        )

        details_type_param = pn.widgets.Select(
            # 'plot_type_details': {'widget_type': pn.widgets.RadioButtonGroup, 'button_type': 'success',
            #         'orientation':"vertical", 'width': widget_widths}},
            value=self.param.plot_type_details,
            options=self.param.plot_type_details_objects,
            width=widget_widths,
        )

        details_param_currentValue = pn.pane.Markdown(f"## {self.parameter}")
        details_param = pn.widgets.MenuButton(
            name="Detailed Plots",
            button_type="primary",
            width=widget_widths,
            items=self.param.parameter.objects,
        )

        def update_details_plots(event):
            self.parameter = event.new
            details_param_currentValue.object = f"## {event.new}"

        details_param.on_click(update_details_plots)

        return pn.Column(
            pn.Row(
                pn.pane.SVG(
                    logo_path / "Calibration.svg",
                    height=25,
                ),
                details_param,
            ),
            pn.Row("## Current Plot:", details_param_currentValue),
            pn.Row(
                "Channel:",
                details_ch_param,
                "Plot type:",
                details_type_param,
                "Set:",
                pn.widgets.Select(
                    value=self.param.psd_set,
                    options=self.param.psd_set_objects,
                    width=widget_widths,
                ),
            ),
            pn.param.ParamMethod(self.get_run_and_channel, lazy=True),
            pn.param.ParamMethod(
                self.view_details,
                lazy=True,
                loading_indicator=True,
                sizing_mode="stretch_width",
            ),
            name="Cal. Details",
            sizing_mode="scale_both",
        )

    def build_summary_pane(self, widget_widths: int = 140):
        summary_param_currentValue = pn.pane.Markdown(f"## {self.plot_type_summary}")
        summary_param = pn.widgets.MenuButton(
            name="Summary Plots",
            button_type="primary",
            width=widget_widths,
            items=self.param.plot_type_summary.objects,
        )

        def update_summary_plots(event):
            self.plot_type_summary = event.new
            summary_param_currentValue.object = f"## {event.new}"

        summary_param.on_click(update_summary_plots)

        summary_param_download = pn.Param(
            self.param,
            widgets={
                "plot_types_download": {
                    "widget_type": pn.widgets.Select,
                    "width": widget_widths,
                }
            },
            parameters=["plot_types_download"],
            show_labels=False,
            show_name=False,
        )
        return pn.Column(
            pn.Row(
                pn.pane.SVG(
                    logo_path / "Calibration.svg",
                    height=25,
                ),
                summary_param,
            ),
            pn.Row("## Current Plot:", summary_param_currentValue),
            "Download Raw Data",
            pn.Row(
                summary_param_download,
                pn.param.ParamMethod(self.download_summary_files, lazy=True),
            ),
            pn.param.ParamMethod(
                self.view_summary,
                lazy=True,
                loading_indicator=True,
                sizing_mode="stretch_width",
            ),
            name="Cal. Summary",
            sizing_mode="scale_both",
        )

    def build_tracking_pane(self, widget_widths: int = 140):
        # tracking_range_param = pn.Param(
        #     self.param,
        #     widgets={
        #         "date_range": {
        #             "widget_type": pn.widgets.DatetimeRangePicker,
        #             "width": widget_widths,
        #             "enable_time": False,
        #             "enable_seconds": False,
        #         }
        #     },
        #     parameters=["date_range"],
        #     show_labels=False,
        #     show_name=False,
        # )

        tracking_param_currentValue = pn.pane.Markdown(f"## {self.plot_type_tracking}")
        tracking_param = pn.widgets.MenuButton(
            name="Tracking Plots",
            button_type="primary",
            width=widget_widths,
            items=self.param.plot_type_tracking.objects,
        )

        def update_tracking_plots(event):
            self.plot_type_tracking = event.new
            tracking_param_currentValue.object = f"## {event.new}"

        tracking_param.on_click(update_tracking_plots)

        return pn.Column(
            pn.Row(
                pn.pane.SVG(
                    logo_path / "Calibration.svg",
                    height=25,
                ),
                tracking_param,
            ),
            pn.Row("## Current Plot:", tracking_param_currentValue),
            # pn.Row("Selected time range:", tracking_range_param),
            pn.param.ParamMethod(
                self.view_tracking,
                lazy=True,
                loading_indicator=True,
                sizing_mode="stretch_width",
            ),
            name="Cal. Tracking",
            sizing_mode="scale_both",
        )

    def build_cal_panes(self, widget_widths: int = 140):
        # update_plot_dict already ran in __init__ (and re-runs via its
        # period/run watcher); no need to redo the shelve reads here.
        return {
            "Cal. Summary": self.build_summary_pane(widget_widths),
            "Cal. Details": self.build_detailed_pane(widget_widths),
            "Cal. Tracking": self.build_tracking_pane(widget_widths),
        }

    @classmethod
    def init_cal_panes(
        cls,
        base_path,
        widget_widths: int = 140,
    ):
        """
        Initialize the calibration panes.

        Args:
            widget_widths (int): Width of the widgets.

        Returns:
            dict: Dictionary containing the calibration panes.
        """
        cal_monitor = cls(base_path=base_path)
        return cal_monitor.build_cal_panes(widget_widths)

    @classmethod
    def display_cal_panes(
        cls,
        base_path,
        notebook=False,
        widget_widths: int = 140,
    ):
        """
        View the calibration panes.

        Args:
            widget_widths (int): Width of the widgets.

        Returns:
            pn.Row: Row containing the calibration panes.
        """
        cal_monitor = cls(base_path=base_path, notebook=notebook)
        sidebar = cal_monitor.build_sidebar()
        return pn.Row(
            sidebar, pn.Tabs(*cal_monitor.build_cal_panes(widget_widths).values())
        )


def run_dashboard_cal() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("config_file", type=str)
    argparser.add_argument("-p", "--port", type=int, default=9000)
    argparser.add_argument(
        "-w", "--widget_widths", type=int, default=140, required=False
    )
    args = argparser.parse_args()

    config = read_config(args.config_file)
    cal_panes = CalMonitoring.display_cal_panes(
        config.base, widget_widths=args.widget_widths
    )
    print("Starting Cal. Monitoring on port ", args.port)  # noqa: T201
    pn.serve(cal_panes, port=args.port, show=False)
