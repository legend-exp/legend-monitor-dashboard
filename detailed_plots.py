import plotly.graph_objects as go


def plot_spectrum(bins, counts, channel, log=False):
    fig = go.Figure()

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