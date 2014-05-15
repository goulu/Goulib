#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
hex RGB colors and related functions
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012-, Philippe Guglielmetti"
__license__ = "LGPL"

import math2

# http://stackoverflow.com/questions/214359/converting-hex-color-to-rgb-and-vice-versa

def rgb_to_hex(rgb):
    """:param rgb: tuple (r,g,b) of 3 ints 0-255
    :result: string "#rrggbb" in hex suitable for HTML color"""
    return '#%02x%02x%02x' % tuple(rgb)

def hex_to_rgb(value,scale=1):
    """:param value: string "#rrggbb" in hex suitable for HTML color
    :param scale: float optional 1./255 to scale output to [0,1] floats
    :result: tuple (r,g,b) of 3 ints 0-255"""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(scale*int(value[i:i+lv/3], 16) for i in range(0, lv, lv/3))

# http://stackoverflow.com/questions/14088375/how-can-i-convert-rgb-to-cmyk-and-vice-versa-in-python

def rgb_to_cmyk(r,g,b):
    """:param r,g,b: floats of red,green,blue in [0..1] range
    :return: tuple of 4 floats (cyan, magenta, yellow, black) in [0..1] range
    """
    c = 1 - r
    m = 1 - g
    y = 1 - b

    k = min(c, m, y)
    c = (c - k) / (1 - k)
    m = (m - k) / (1 - k)
    y = (y - k) / (1 - k)
    return (c,m,y,k)

#http://stackoverflow.com/questions/876853/generating-color-ranges-in-python
    
def color_range(n,start,end):
    """:param n: int number of colors to generate
    :param start: string hex color or color name
    :param end: string hex color or color name
    :result: list of n hexcolors interpolated between start and end, included
    """
    import colorsys
    from itertools2 import ilinear
    if start in color: start=color[start]
    start=hex_to_rgb(start,1./255)
    start=colorsys.rgb_to_hsv(*start)
    if end in color: end=color[end]
    end=hex_to_rgb(end,1./255)
    end=colorsys.rgb_to_hsv(*end)
    res=[]
    for hsv in ilinear(start,end,n):
        rgb=colorsys.hsv_to_rgb(*hsv)
        hex=rgb_to_hex(tuple(int(255*x) for x in rgb))
        res.append(hex)
    return res

color = {
    # dict of most used colors indexed by name
    'black': '#000000',
    'blue': '#0000ff',
    'fuchsia': '#ff00ff',
    'green': '#008000',
    'grey': '#808080',
    'lime': '#00ff00',
    'maroon': '#800000',
    'navy': '#000080',
    'olive': '#808000',
    'purple': '#800080',
    'red': '#ff0000',
    'silver': '#c0c0c0',
    'teal': '#008080',
    'white': '#ffffff',
    'yellow': '#ffff00',
    'aliceblue': '#f0f8ff',
    'antiquewhite': '#faebd7',
    'aqua': '#00ffff',
    'aquamarine': '#7fffd4',
    'azure': '#f0ffff',
    'beige': '#f5f5dc',
    'bisque': '#ffe4c4',
    'black': '#000000',
    'blanchedalmond': '#ffebcd',
    'blue': '#0000ff',
    'blueviolet': '#8a2be2',
    'brown': '#a52a2a',
    'burlywood': '#deb887',
    'cadetblue': '#5f9ea0',
    'chartreuse': '#7fff00',
    'chocolate': '#d2691e',
    'coral': '#ff7f50',
    'cornflowerblue': '#6495ed',
    'cornsilk': '#fff8dc',
    'crimson': '#dc143c',
    'cyan': '#00ffff',
    'darkblue': '#00008b',
    'darkcyan': '#008b8b',
    'darkgoldenrod': '#b8860b',
    'darkgray': '#a9a9a9',
    'darkgrey': '#a9a9a9',
    'darkgreen': '#006400',
    'darkkhaki': '#bdb76b',
    'darkmagenta': '#8b008b',
    'darkolivegreen': '#556b2f',
    'darkorange': '#ff8c00',
    'darkorchid': '#9932cc',
    'darkred': '#8b0000',
    'darksalmon': '#e9967a',
    'darkseagreen': '#8fbc8f',
    'darkslateblue': '#483d8b',
    'darkslategray': '#2f4f4f',
    'darkslategrey': '#2f4f4f',
    'darkturquoise': '#00ced1',
    'darkviolet': '#9400d3',
    'deeppink': '#ff1493',
    'deepskyblue': '#00bfff',
    'dimgray': '#696969',
    'dimgrey': '#696969',
    'dodgerblue': '#1e90ff',
    'firebrick': '#b22222',
    'floralwhite': '#fffaf0',
    'forestgreen': '#228b22',
    'fuchsia': '#ff00ff',
    'gainsboro': '#dcdcdc',
    'ghostwhite': '#f8f8ff',
    'gold': '#ffd700',
    'goldenrod': '#daa520',
    'gray': '#808080',
    'grey': '#808080',
    'green': '#008000',
    'greenyellow': '#adff2f',
    'honeydew': '#f0fff0',
    'hotpink': '#ff69b4',
    'indianred': '#cd5c5c',
    'indigo': '#4b0082',
    'ivory': '#fffff0',
    'khaki': '#f0e68c',
    'lavender': '#e6e6fa',
    'lavenderblush': '#fff0f5',
    'lawngreen': '#7cfc00',
    'lemonchiffon': '#fffacd',
    'lightblue': '#add8e6',
    'lightcoral': '#f08080',
    'lightcyan': '#e0ffff',
    'lightgoldenrodyellow': '#fafad2',
    'lightgray': '#d3d3d3',
    'lightgrey': '#d3d3d3',
    'lightgreen': '#90ee90',
    'lightpink': '#ffb6c1',
    'lightsalmon': '#ffa07a',
    'lightseagreen': '#20b2aa',
    'lightskyblue': '#87cefa',
    'lightslategray': '#778899',
    'lightslategrey': '#778899',
    'lightsteelblue': '#b0c4de',
    'lightyellow': '#ffffe0',
    'lime': '#00ff00',
    'limegreen': '#32cd32',
    'linen': '#faf0e6',
    'magenta': '#ff00ff',
    'maroon': '#800000',
    'mediumaquamarine': '#66cdaa',
    'mediumblue': '#0000cd',
    'mediumorchid': '#ba55d3',
    'mediumpurple': '#9370d8',
    'mediumseagreen': '#3cb371',
    'mediumslateblue': '#7b68ee',
    'mediumspringgreen': '#00fa9a',
    'mediumturquoise': '#48d1cc',
    'mediumvioletred': '#c71585',
    'midnightblue': '#191970',
    'mintcream': '#f5fffa',
    'mistyrose': '#ffe4e1',
    'moccasin': '#ffe4b5',
    'navajowhite': '#ffdead',
    'navy': '#000080',
    'oldlace': '#fdf5e6',
    'olive': '#808000',
    'olivedrab': '#6b8e23',
    'orange': '#ffa500',
    'orangered': '#ff4500',
    'orchid': '#da70d6',
    'palegoldenrod': '#eee8aa',
    'palegreen': '#98fb98',
    'paleturquoise': '#afeeee',
    'palevioletred': '#d87093',
    'papayawhip': '#ffefd5',
    'peachpuff': '#ffdab9',
    'peru': '#cd853f',
    'pink': '#ffc0cb',
    'plum': '#dda0dd',
    'powderblue': '#b0e0e6',
    'purple': '#800080',
    'red': '#ff0000',
    'rosybrown': '#bc8f8f',
    'royalblue': '#4169e1',
    'saddlebrown': '#8b4513',
    'salmon': '#fa8072',
    'sandybrown': '#f4a460',
    'seagreen': '#2e8b57',
    'seashell': '#fff5ee',
    'sienna': '#a0522d',
    'silver': '#c0c0c0',
    'skyblue': '#87ceeb',
    'slateblue': '#6a5acd',
    'slategray': '#708090',
    'slategrey': '#708090',
    'snow': '#fffafa',
    'springgreen': '#00ff7f',
    'steelblue': '#4682b4',
    'tan': '#d2b48c',
    'teal': '#008080',
    'thistle': '#d8bfd8',
    'tomato': '#ff6347',
    'turquoise': '#40e0d0',
    'violet': '#ee82ee',
    'wheat': '#f5deb3',
    'white': '#ffffff',
    'whitesmoke': '#f5f5f5',
    'yellow': '#ffff00',
    'yellowgreen': '#9acd32'}

color_lookup=dict([v,k] for k,v in color.iteritems()) #http://code.activestate.com/recipes/252143-invert-a-dictionary-one-liner/

def nearest_color(x):
    """:return: name of the nearest color"""
    if isinstance(x,basestring):
        rgb=hex_to_rgb(x,1./255)
    else:
        rgb=math2.sat(x,0,1)
    res=min(color.values(),key=lambda _:math2.dist(rgb, hex_to_rgb(_,1./255)))
    return color_lookup[res]

from math2 import rint

class Color(object):
    '''class to allow simple math operations on colors'''
    def __init__(self,c):
        ''':param c: either color name, hex string, or (r,g,b) tuple in [0..255] int or [0,.1.] float range'''
        if isinstance(c,basestring):
            try: #is c a color name ?
                c=color[c]
            except: #assume it's a hex string
                pass
            self.rgb=hex_to_rgb(c,1./255)
        else: # assume (r,g,b) tuple
            if max(c)>1:
                c=tuple(_/255. for _ in c)
            self.rgb=c
        
        try: #to guess the color name
            self.name=color_lookup[self.hex]
            return
        except: #find the closest one
            pass
        self.name='~'+nearest_color(self.rgb)
            
    @property
    def hex(self):
        rgb=math2.sat(self.rgb)
        return rgb_to_hex((rint(_*255) for _ in rgb))
    
    def __repr__(self):
        if self.name!='unknown':
            return self.name
        else:
            return self.hex
        
    def _repr_html_(self):
        return '<div style="color:%s">%s</div>'%(self.hex,self.name)
        
    def __eq__(self,other):
        try:
            return self.hex==other.hex
        except:
            return self.name==other
    
    def __add__(self,other):
        return Color(math2.vecadd(self.rgb,other.rgb))
    
    def __sub__(self,other):        
        return Color(math2.vecsub(self.rgb,other.rgb))
