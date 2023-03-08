import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def plot_spectrum(plot_dict, channel, log=False):
    fig = go.Figure()
    bins=plot_dict["bins"]
    counts = plot_dict["counts"]

    fig.add_trace(go.Scatter(x=bins, 
                             y=counts, 
                             name=channel,
                             line_shape='hvh', 
                             line =dict(width=1)))

    fig.update_traces(mode='lines')

    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='grey',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=True,
            showline=True,
            linecolor='grey',
            linewidth=2,
            showticklabels=True,
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            )
        ),
        autosize=False,
        margin=dict(
            autoexpand=False,
            l=100,
            r=20,
            t=110,
        ),
        showlegend=False,
        plot_bgcolor='white'
    )
    annotations =[]
    # Title
    annotations.append(dict(xref='paper', yref='paper', x=0.2, y=1.05,
                                  xanchor='left', yanchor='bottom',
                                  text=channel,
                                  font=dict(family='Arial',
                                            size=20,
                                            color='rgb(82, 82, 82)'),
                                  showarrow=False))
    # X label
    annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,
                                  xanchor='center', yanchor='top',
                                  text='Energy (keV)',
                                  font=dict(family='Arial',
                                            size=12,
                                            color='rgb(82, 82, 82)'),
                                  showarrow=False))

    #Y label
    annotations.append(dict(xref='paper', yref='paper', x=-0.1, y=0.5,
                                  xanchor='left', yanchor='middle',
                                  text='Counts',
                                    textangle =270,
                                  font=dict(family='Arial',
                                            size=12,
                                            color='rgb(82, 82, 82)'),
                                  showarrow=False))
    
    fig.update_layout(
        yaxis = dict(
            showexponent = 'all',
            exponentformat = "none"
        )
    )
    if log is True:
        fig.update_yaxes(type="log",)
        fig.update_layout(
            yaxis = dict(
                showexponent = 'all',
                exponentformat = 'power'
            )
        )
    fig.update_layout(annotations=annotations)
    return fig

def plot_survival_frac(plot_dict):
    fig = plt.figure()
    plt.step(plot_dict["bins"], plot_dict["sf"], where="mid")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Survival Fraction (%)")
    plt.close()
    return fig

def plot_cut_spectra(plot_dict):
    fig = plt.figure()
    plt.step(plot_dict["bins"], plot_dict["counts"], where="mid", label="After Cuts")
    plt.step(plot_dict["bins"], plot_dict["cut_counts"], where="mid", label="Cut Spectrum")
    if np.isnan(plot_dict["pulser_counts"]).all():
            pass
    else:
        plt.step(plot_dict["bins"], plot_dict["pulser_counts"], where="mid", label="Pulser")
    
    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.close()
    return fig

def track_peaks(plot_dict):
    time_bins = plot_dict["2614_stability"]["time"]
    th_counts = plot_dict["2614_stability"]["energy"]
    th_spread = plot_dict["2614_stability"]["spread"]
    th_mean = np.mean(th_counts[~np.isnan(th_counts)][:10])
    th_shift = 100*(th_counts-th_mean)/th_mean
    th_shift_err = 100*th_spread/th_mean
    
    lep_counts = plot_dict["583_stability"]["energy"]
    lep_spread = plot_dict["583_stability"]["spread"]
    lep_mean = np.mean(lep_counts[~np.isnan(lep_counts)][:10])
    lep_shift = 100*(lep_counts-lep_mean)/lep_mean
    lep_shift_err = 100*lep_spread/lep_mean


    fig = plt.figure(figsize = (8,6))
    plt.step(time_bins, th_shift, where="mid", label="2.6 MeV peak", color="blue")
    
    plt.fill_between(time_bins, th_shift-th_shift_err,th_shift+th_shift_err, 
                        step="mid", alpha=0.1, color="blue")
    
    plt.step(time_bins, lep_shift, where="mid", label="583 keV peak", color="orange")
    plt.fill_between(time_bins, lep_shift-lep_shift_err,lep_shift+lep_shift_err, 
                        step="mid", alpha=0.1, color="orange")
    
    pulser_counts = plot_dict["pulser_stability"]["energy"]
    pulser_spread = plot_dict["pulser_stability"]["spread"]
    if np.isnan(pulser_counts).all():
        pass
    else:
        pulser_mean = np.mean(pulser_counts[~np.isnan(pulser_counts)][:10])
        pulser_shift = 100*(pulser_counts-pulser_mean)/pulser_mean
        pulser_shift_err = 100*pulser_spread/pulser_mean
        plt.step(time_bins, pulser_shift, where="mid", color="red", label=f"{pulser_mean:.0f} keV Pulser")
        plt.fill_between(time_bins, pulser_shift-pulser_shift_err,pulser_shift+pulser_shift_err, 
                            step="mid", alpha=0.1, color="red")

    ticks, labels = plt.xticks()
    plt.xlabel(
        f"Time starting : {datetime.utcfromtimestamp(ticks[0]).strftime('%d/%m/%y %H:%M')}"
    )
    plt.ylabel("% Shift")
    plt.xticks(
    ticks,
    [datetime.utcfromtimestamp(tick).strftime("%H:%M") for tick in ticks],
    )
    plt.xlim([time_bins[0]-10, time_bins[-1]+10])

    
    plt.grid(which='both')
    plt.legend()
    plt.ylim([-1,1])
    plt.close()
    return fig