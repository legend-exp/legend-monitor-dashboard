import numpy as np
import matplotlib.pyplot as plt
import math
import json

from os import listdir

def is_taper(f):
    return f != {'angle_in_deg': 0, 'height_in_mm': 0} and  f != {'radius_in_mm': 0, 'height_in_mm': 0}

def is_bulletized(f):
    return 'bulletization' in f and f['bulletization'] != {'top_radius_in_mm': 0, 'bottom_radius_in_mm': 0, 'borehole_radius_in_mm': 0, 'contact_radius_in_mm': 0}

def has_groove(f):
    return 'groove' in f and f['groove'] != {'outer_radius_in_mm': 0, 'depth_in_mm': 0, 'width_in_mm': 0}

def has_borehole(f):
    return 'borehole' in f and f['borehole'] != {'gap_in_mm': 0, 'radius_in_mm': 0}

# compose a list of all detectors with JSON metadata files
# should be what is in https://github.com/legend-exp/legend-detectors
datadir = "/remote/ceph/user/h/hagemann/LEGEND/legend-detectors/germanium/diodes/"
detectors = list(map(lambda x: x[:-5], filter(lambda x : x.endswith(".json") and not x.startswith("C"), listdir(datadir))))

# this should be provided by the user (e.g. HV on/off and a detector_map)
from random import shuffle, choices
HV_on = dict(zip(detectors, choices([True, False], k = len(detectors))))
detector_map = [choices(detectors, k = 6) for _ in range(12)]



###############################################
# HERE COMES THE PLOTTING CODE FOR MATPLOTLIB #
###############################################

fig = plt.figure(figsize = (20,10))
ax = fig.gca()

maxH = 0
R = 0
H = 0

for s in detector_map:
    maxDR = 0
    shuffle(s)
    
    for d in s:
        f = open(datadir + d + '.json')
        j = json.load(f)
        f.close()
        g = j['geometry']
        
        detcolor = 'green' if HV_on[d] else 'red'

        DH = g['height_in_mm']
        DR = g['radius_in_mm']
        CR = g['contact']['radius_in_mm']
        CH = g['contact']['depth_in_mm']

        pts = np.array([[R-DR,H], [R+DR,H], [R+DR,H-DH], [R-DR,H-DH]])
        bulk = plt.Polygon(pts, facecolor = detcolor, linewidth = 0)
        contact = plt.Polygon(np.array([[R-CR,H-DH], [R+CR,H-DH], [R+CR,H-DH+CH], [R-CR,H-DH+CH]]), facecolor = 'black', edgecolor = 'black')

        ax.add_patch(bulk)
        ax.add_patch(contact)

        if has_groove(g):
            GR = g['groove']['outer_radius_in_mm']
            GH = g['groove']['depth_in_mm']
            GW = g['groove']['width_in_mm']
            groove = plt.Polygon(np.array([[R-GR,H-DH],[R-GR,H-DH+GH],[R-GR+GW,H-DH+GH],[R-GR+GW,H-DH]]), facecolor = 'white')
            ax.add_patch(groove)
            groove = plt.Polygon(np.array([[R+GR,H-DH],[R+GR,H-DH+GH],[R+GR-GW,H-DH+GH],[R+GR-GW,H-DH]]), facecolor = 'white')
            ax.add_patch(groove)

        if has_borehole(g):
            BG = g['borehole']['gap_in_mm']
            BR = g['borehole']['radius_in_mm']
            borehole = plt.Polygon(np.array([[R-BR,H],[R-BR,H-DH+BG],[R+BR,H-DH+BG],[R+BR,H]]), color = 'white')
            ax.add_patch(borehole)

            topin  = g['taper']['top']['inner']
            if is_taper(topin):
                TH = topin['height_in_mm']
                TR = TH * math.sin(topin['angle_in_deg'] * math.pi/180)
                taper = plt.Polygon(np.array([[R-BR-TR,H],[R-BR,H-TH],[R+BR,H-TH],[R+BR+TR,H]]), color = 'white')
                ax.add_patch(taper)


        topout = g['taper']['top']['outer']
        if is_taper(topout):
            TH = topout['height_in_mm']
            TR = TH * math.sin(topout['angle_in_deg'] * math.pi/180)
            taper = plt.Polygon(np.array([[R+DR,H-TH], [R+DR-TR,H], [R+DR,H]]), color = 'white')
            ax.add_patch(taper)
            taper = plt.Polygon(np.array([[R-DR,H-TH], [R-DR+TR,H], [R-DR,H]]), color = 'white')
            ax.add_patch(taper)


        botout = g['taper']['bottom']['outer']
        if is_taper(botout):
            TH = botout['height_in_mm']
            TR = botout['radius_in_mm'] if 'radius_in_mm' in botout else TH * math.sin(botout['angle_in_deg'] * math.pi/180)
            taper = plt.Polygon(np.array([[R+DR,H-DH+TH], [R+DR-TR,H-DH], [R+DR,H-DH]]), color = 'white')
            ax.add_patch(taper)
            taper = plt.Polygon(np.array([[R-DR,H-DH+TH], [R-DR+TR,H-DH], [R-DR,H-DH]]), color = 'white')
            ax.add_patch(taper)

        ax.annotate(j['name'], (R, H-DH/2), ha = 'center', va = 'center')

        maxDR = max(DR, maxDR)
        H -= DH + 15
    R += 100
    maxH = min(H, maxH)
    H = 0
    
ax.axis('off')
ax.set_aspect("equal")
ax.set_xlim(-maxDR-10,R + maxDR-90)
ax.set_ylim(maxH-10,10)