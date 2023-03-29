import numpy as np
import math

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, LabelSet, LinearColorMapper, BasicTicker, ColorBar, FixedTicker, FuncTickFormatter, Legend, LegendItem, PrintfTickFormatter
from bokeh.palettes import *

# # display bokeh plots in a notebook
# from bokeh.io import output_notebook
# from bokeh.resources import INLINE
# output_notebook(INLINE)

# functions needed to plot the detectors in their correct shape

def is_coax(d):
    return d['type'] == 'coax'

def is_taper(f):
    return f != {'angle_in_deg': 0, 'height_in_mm': 0} and f != {'radius_in_mm': 0, 'height_in_mm': 0}

def is_bulletized(f):
    return 'bulletization' in f and f['bulletization'] != {'top_radius_in_mm': 0, 'bottom_radius_in_mm': 0, 'borehole_radius_in_mm': 0, 'contact_radius_in_mm': 0}

def has_groove(f):
    return 'groove' in f and f['groove'] != {'outer_radius_in_mm': 0, 'depth_in_mm': 0, 'width_in_mm': 0}

def has_borehole(f):
    return 'borehole' in f and f['borehole'] != {'gap_in_mm': 0, 'radius_in_mm': 0}

def plot_geometry(d, R, H):
    
    coax = is_coax(d)
    
    g = d['geometry']
    
    DH = g['height_in_mm']
    DR = g['radius_in_mm']

    xbot = []
    ybot = []

    botout = g['taper']['bottom']['outer']
    if is_taper(botout):
        TH = botout['height_in_mm']
        TR = botout['radius_in_mm'] if 'radius_in_mm' in botout else TH * math.sin(botout['angle_in_deg'] * math.pi/180)
        xbot.extend([DR, DR-TR])
        ybot.extend([H-DH+TH,H-DH])
    else:
        xbot.append(DR)
        ybot.append(H-DH)

    if has_groove(g):
        GR = g['groove']['outer_radius_in_mm']
        GH = g['groove']['depth_in_mm']
        GW = g['groove']['width_in_mm']
        xbot.extend([GR,GR,GR-GW,GR-GW])
        ybot.extend([H-DH,H-DH+GH,H-DH+GH,H-DH])
        
    if coax:
        BG = g['borehole']['gap_in_mm']
        BR = g['borehole']['radius_in_mm']
        xbot.extend([BR, BR])
        ybot.extend([H-DH, H-DH+BG])
        

    xtop = []
    ytop = []

    topout = g['taper']['top']['outer']
    if is_taper(topout):
        TH = topout['height_in_mm']
        TR = TH * math.sin(topout['angle_in_deg'] * math.pi/180)
        xtop.extend([DR, DR-TR])
        ytop.extend([H-TH, H])
    else:
        xtop.append(DR)
        ytop.append(H)

    if has_borehole(g) and not coax:
        BG = g['borehole']['gap_in_mm']
        BR = g['borehole']['radius_in_mm']

        topin  = g['taper']['top']['inner']
        if is_taper(topin):
            TH = topin['height_in_mm']
            TR = TH * math.sin(topin['angle_in_deg'] * math.pi/180)
            xtop.extend([BR+TR,BR,BR])
            ytop.extend([H, H-TH, H-DH+BG])     
        else:
            xtop.extend([BR, BR])
            ytop.extend([H, H-DH+BG])


    x = np.hstack(([-x+R for x in xbot], [x+R for x in xbot[::-1]],[x + R for x in xtop],[-x+R for x in xtop[::-1]]))  
    y = np.hstack((ybot, ybot[::-1],ytop,ytop[::-1]))  
    return x, y


def get_plot_source_and_xlabels(chan_dict, channel_map, strings_dict, ΔR = 160, ΔH = 40):

    xs = []
    ys = []
    ch = []
    hw = []
    sw = []
    dn = []
    st = []
    pos = []
    ax = []
    ay = []

    maxH = 0
    R = 0
    H = 0

    xlabels = dict()

    for (name, string) in strings_dict.items():

        xlabels[R] = name

        for channel_no in string:
            ch.append(channel_no)
            status = chan_dict[channel_map[channel_no]['name']]
            hw.append(status['usability'])
            sw.append(status['processable'])
            channel = channel_map[channel_no]
            dn.append(channel['name'])
            st.append(channel['location']['string'])
            pos.append(channel['location']['position'])
            x,y = plot_geometry(channel, R, H)
            xs.append(x)
            ys.append(y)
            H -= channel['geometry']['height_in_mm'] + ΔH
            ax.append(R)
            ay.append(H+10)
        R += ΔR
        maxH = min(H, maxH)
        H = 0

    return ColumnDataSource(
        dict(
            xs=xs,   # x-coordinates (needed for plotting)
            ys=ys,   # y-coordinates (needed for plotting)
            ch=ch,   # channel numbers
            hw=hw,   # hardware status
            sw=sw,   # software status
            dn=dn,   # detector name
            st=st,   # string number
            pos=pos, # position in string
            ax=ax,
            ay=ay,
            mass=list(map(lambda i: f"{float(channel_map[i]['production']['mass_in_g'])}g", ch)),
        )
    ), xlabels


def create_detector_plot(source, display_dict, xlabels, ctitle = "", palette = inferno(256), ticker = BasicTicker(), formatter = None):
    
    tooltips = [
    ("Detector Name", "@dn"),
    ("Channel", "@ch"),
    ("String", "@st"),
    ("Usability", "@hw"),
    ("Processable", "@sw"),
    ("Mass", "@mass"),
    (f"{ctitle}", "$y")
    ]
    
    # To Do: find optimal width and height values for plotting (do not hardcode)
    p = figure(title="LEGEND detector monitoring", width=1200, height=920,
        tools="pan,box_zoom,hover,reset,save", tooltips=tooltips, match_aspect=True)

    # handle colors according to display_dict
    values = list(filter(lambda v: v is not None, display_dict.values()))
    minvalue = min(values); maxvalue = max(values)
    color_mapper = LinearColorMapper(palette=palette, low=minvalue, high=maxvalue)
    colors = list(map(lambda v : 'white' if v is None else palette[int((v - minvalue)/(maxvalue-minvalue) * (len(palette) - 1))], display_dict.values()))
    source.data["color"] = colors
    
    # plot detector geometries with respective colors
    p.patches("xs", "ys", source = source, line_color = "black", color = "color")
    color_bar = ColorBar(color_mapper=color_mapper, ticker=ticker, title = ctitle)
    if not formatter is None: color_bar.formatter = formatter
    p.add_layout(color_bar, 'right')
    
    # annotate detector names underneath the detectors
    labels = LabelSet(x='ax', y='ay', text='dn',source=source, text_font_size = "10px", text_align="center")
    p.add_layout(labels)

    # remove unnecessary lines in the plot
    p.outline_line_color = None
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 0 
    p.xaxis.ticker = list(xlabels)
    p.xaxis.major_label_overrides = xlabels
    p.yaxis.visible = False
    p.xaxis.major_label_text_font_style = "bold"
    p.xaxis.axis_label_text_font_size = "25px"
    # return the plot
    return p


def plot_visu_usability(source, chan_dict, channel_map, xlabels):
    color_dict = {'on': 1, 'off': 0}
    display_dict = {i : color_dict[chan_dict[channel_map[i]['name']]['usability']] for i in source.data['ch']}
    palette = ('red', 'green')
    ctitle = 'Usability'
    ticker = FixedTicker(ticks=[0.25,0.75], tags = ['off', 'on'])
    formatter = FuncTickFormatter(code="""
        var mapping = {0.25: "off", 0.75: "on"};
        return mapping[tick];
    """)
    return create_detector_plot(source, display_dict, xlabels, ctitle = ctitle, palette = palette, ticker = ticker, formatter = formatter)

def plot_visu_processable(source, chan_dict, channel_map, xlabels):
    color_dict = {True: 1, False: 0}
    display_dict = {i : color_dict[chan_dict[channel_map[i]['name']]['processable']] for i in source.data['ch']}
    palette = ('red', 'green')
    ctitle = 'Processable'
    ticker = FixedTicker(ticks=[0.25,0.75], tags = ['True', 'False'])
    formatter = FuncTickFormatter(code="""
        var mapping = {0.25: "True", 0.75: "False"};
        return mapping[tick];
    """)
    return create_detector_plot(source, display_dict, xlabels, ctitle = ctitle, palette = palette, ticker = ticker, formatter = formatter)


def plot_visu_mass(source, chan_dict, channel_map, xlabels):
    display_dict = {i : channel_map[i]['production']['mass_in_g'] for i in source.data['ch']}
    ctitle = 'Mass in g'
    return create_detector_plot(source, display_dict, xlabels, ctitle = ctitle)


def plot_visu_depletion(source, chan_dict, channel_map, xlabels):
    display_dict = {i : channel_map[i]['production']['mass_in_g'] for i in source.data['ch']}
    ctitle = "Depletion voltage in V (manufacturer)"
    palette = viridis(256)
    return create_detector_plot(source, display_dict, xlabels, ctitle = ctitle, palette=palette)


def plot_visu_operation(source, chan_dict, channel_map, xlabels):
    display_dict = {i : channel_map[i]['characterization']['manufacturer']['recommended_voltage_in_V'] for i in source.data['ch']}
    ctitle = 'Operational voltage in V (manufacturer)'
    palette = viridis(256)
    return create_detector_plot(source, display_dict, xlabels, ctitle = ctitle, palette=palette)


def plot_visu_enrichment(source, chan_dict, channel_map, xlabels):
    display_dict = {i : np.nan_to_num(channel_map[i]['production']['enrichment'])*100 for i in source.data['ch']}
    ctitle = 'Enrichment in %'
    return create_detector_plot(source, display_dict, xlabels, ctitle = ctitle)


