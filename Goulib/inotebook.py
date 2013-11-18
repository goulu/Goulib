#http://nbviewer.ipython.org/5162445


from io import BytesIO

import IPython.core
import PIL


def display_pil_image(im):
   """Displayhook function for PIL Images, rendered as PNG."""

   b = BytesIO()
   im.save(b, format='png')
   data = b.getvalue()

   ip_img = IPython.core.display.Image(data=data, format='png', embed=True)
   return ip_img._repr_png_()


# register display func with PNG formatter:
png_formatter = get_ipython().display_formatter.formatters['image/png']
dpi = png_formatter.for_type(PIL.Image.Image, display_pil_image)

def plot(pwfs,ylim=None,labels=None,offset=None,max=None):
    """plots Piecewise function(s)"""
    from Goulib.piecewise import Piecewise
    if isinstance(pwfs,Piecewise):
        pwfs=[pwfs]
    for i,pwf in enumerate(pwfs):
        (x,y)=pwf.points(max=max)
        if offset:
            x=[x+i*offset for x in x]
            y=[y+i*offset for y in y]
        try:
            label=labels[i]
        except:
            label=''
        pylab.plot(x, y, label=label)
        
    if ylim: pylab.ylim(ylim)
    try:
        if labels: pylab.legend()
    except: pass
    pylab.show() 