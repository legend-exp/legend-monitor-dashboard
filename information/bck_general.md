# L200 Monitoring

This dashboard shows the status of all L200 detector systems in a semi-live fashion.
The plots are automatically updated every **6 hours** directly from LNGS via `rsync`. So far, the calibration has to be updated manually via the
Admin panel.
In case of questions, bug reports or feature requests, please contact [Florian Henkes](mailto:florian.henkes@tum.de) or report in the corresponding Slack channel.

## General information

The dashboard is based on the [Panel](https://panel.holoviz.org/) library and is hosted on a local server at TUM with connection to LNGS. Since the computing power is limited, the number of parallel users is limited to 10 and every session will be discarded after 10 minutes of inactivity. Since regular updates are performed, the dashboard might be unavailable for a few minutes from time to time. Furthermore, the dashboard is not fully optimized for mobile devices, but should work on most. Since there might be updated versions and new releases, the general dashboard access should always be done via the [L200 Monitoring](https://legend.edm.nat.tum.de) link by clicking on the buttons.

## General Layout

The dashboard is divided into three main sections:

- **Header**: The header contains the title of the dashboard and the current time of the last update. Furthermore, you find the general navigation buttons to select a period and a run as well as to control the sorting and the string selection of the HPGe detectors. The buttons change all plots at once why it takes some time to update all plots since all data has to be reloaded. These buttons are implemented as FloatingWidgets and can be moved around the dashboard to arange according to your needs. In addition, you find a spinning wheel in the right corner which indicates that the dashboard is currently updating the plots.
- **Main**: The main panel contains all plots sorted by Germanium detector calibration plots (_Cal._), Physics Monitoring plots for HPGe detector (_Phy._), SiPM monitoring plots (_SiPM_), Muon veto monitoring plots (_Muon_), LLama monitoring plots (_Llama_) as well as this General panel (_General_) and a Metadata tab (_Metadata_) which also contains detector visualizations.
- **Sidebar**: The sidebar contains contains selectors for the different tabs in the main panel sorted according to the main panel's tab names. You can expand and collapse the sidebar selectors by clicking on the small arrow next to the tab name.

## Interactivity

### Panels

The dashboard is build upon the [GoldenLayout](https://golden-layout.com) framework and therefore designed to be interactive. You can move around the different tabs by dragging them around and build your own custom layout for working with the plots to show different plots at the sime time. In addition, you can also maximize the sidebar or the main panel by clicking on the maximize buttons in right corners. If you want to reload the full page to get back to starting layout, just click on the logo in the top left corner.

### Plots

Almost all plots are build upon the [Bokeh](https://docs.bokeh.org/en/latest/index.html) library and therefore interactive. Each Bokeh plot give you the possibility to interact with the data. By default, you can move the data around by clicking and dragging the plot with your mouse. To zoom in and out, you can use the mouse wheel after clicking on the zoom button (third button from the top) or by drawing a rectangle in the plot (second button from the top). Also, the magnifying glass button (fourth and fifth button from the top) allows you to zoom in and out the x-axis. The reset button (sixth button from the top) allows you to reset the plot to the default view. With the seventh button, you can download your current plot selection as _PNG_. In addition, most of the plots have a hover tool which allows you to get more information about the data points by hovering over them with your mouse. This can be de-/activated by clicking on last button.
Last but not least, each legend is interactive so that you can de-select the corresponding data by clicking on the legend entry. This is especially useful if you only want to see a specific detector to show its feature in a specific variable.

## For shifters

If you report in the commissiioning call, please export the corresponding plots you want to show by downloading the figure as _PNG_ from the Bokeh window and add them to your slides. The download contains your current zoom and selection so that it is easy to export also specific features inside the plots.

In the following, you find a preliminary list of the most important plots to look at that will be updated regularly. Please feel free to give feedback on that list and add/remove plots if necessary.

### Germanium Detectors

- Baseline vs. Time (_Phy_ / _Cal_)
- Noise/Baseline RMS vs. Time (_Phy_)
- Pulser Stability vs. Time (_Phy_ / _Cal_)
- FEP Stability vs. Time (_Cal_)

### SiPM Detectors

- Light Intensity (PE/s) vs. Time (_SiPM_)

## Muon Veto PMTs

- Rate vs. Time (_Muon_)
