"""
plotable rich object display on IPython/Jupyter notebooks 
"""

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2015, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"

# import matplotlib and set backend once for all
from . import itertools2
import os
import io
import sys
import logging
import base64
import matplotlib

if os.getenv('TRAVIS'):  # are we running https://travis-ci.org/ automated tests ?
    matplotlib.use('Agg')  # Force matplotlib  not to use any Xwindows backend
elif sys.gettrace():  # http://stackoverflow.com/questions/333995/how-to-detect-that-python-code-is-being-executed-through-the-debugger
    matplotlib.use('Agg')  # because 'QtAgg' crashes python while debugging
else:
    pass
    # matplotlib.use('pdf') #for high quality pdf, but doesn't work for png, svg ...

logging.info('matplotlib backend is %s' % matplotlib.get_backend())


class Plot(object):
    """base class for plotable rich object display on IPython notebooks
    inspired from http://nbviewer.ipython.org/github/ipython/ipython/blob/3607712653c66d63e0d7f13f073bde8c0f209ba8/docs/examples/notebooks/display_protocol.ipynb
    """

    def _plot(self, ax, **kwargs):
        """abstract method, must be overriden

        :param ax: `matplotlib.axis` 
        :return ax: `matplotlib.axis` after plot
        """
        raise NotImplementedError(
            'objects derived from plot.PLot must define a _plot method')
        return ax

    def render(self, fmt='svg', **kwargs):
        return render([self], fmt, **kwargs)  # call global function

    def save(self, filename, **kwargs):
        return save([self], filename, **kwargs)  # call global function

    # for IPython notebooks

    def _repr_html_(self):
        """default rich format is svg plot"""
        try:
            return self._repr_svg_()
        except NotImplementedError:
            pass
        # this returns  the same as _repr_png_, but is Table compatible
        buffer = self.render('png')
        s = base64.b64encode(buffer).decode('utf-8')
        return '<img src="data:image/png;base64,%s">' % s

    def html(self, **kwargs):
        from IPython.display import HTML
        return HTML(self._repr_html_(**kwargs))

    def svg(self, **kwargs):
        from IPython.display import SVG
        return SVG(self._repr_svg_(**kwargs))

    def _repr_svg_(self, **kwargs):
        return self.render(fmt='svg', **kwargs).decode('utf-8')

    def png(self, **kwargs):
        from IPython.display import Image
        return Image(self._repr_png_(**kwargs), embed=True)

    def _repr_png_(self, **kwargs):
        return self.render(fmt='png', **kwargs)

    def plot(self, **kwargs):
        """ renders on IPython Notebook
        (alias to make usage more straightforward)
        """
        return self.svg(**kwargs)


def render(plotables, fmt='svg', **kwargs):
    """renders several Plot objects"""
    import matplotlib.pyplot as plt

    # extract optional arguments used for rasterization
    printargs, kwargs = itertools2.dictsplit(
        kwargs,
        ['dpi', 'transparent', 'facecolor', 'background', 'figsize']
    )

    ylim = kwargs.pop('ylim', None)
    xlim = kwargs.pop('xlim', None)
    title = kwargs.pop('title', None)

    fig, ax = plt.subplots()

    labels = kwargs.pop('labels', [None] * len(plotables))
    # slightly shift the points to make superimposed curves more visible
    offset = kwargs.pop('offset', 0)

    for i, obj in enumerate(plotables):
        if labels[i] is None:
            labels[i] = str(obj)
        if not title:
            try:
                title = obj._repr_latex_()
                # check that title can be used in matplotlib
                from matplotlib.mathtext import MathTextParser
                parser = MathTextParser('path').parse(title)
            except Exception as e:
                title = labels[i]
        ax = obj._plot(ax, label=labels[i], offset=i * offset, **kwargs)

    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)

    ax.set_title(title)

    if len(labels) > 1:
        ax.legend()

    output = io.BytesIO()
    fig.savefig(output, format=fmt, **printargs)
    data = output.getvalue()
    plt.close(fig)
    return data


def png(plotables, **kwargs):
    from IPython.display import Image
    return Image(render(plotables, 'png', **kwargs), embed=True)


def svg(plotables, **kwargs):
    from IPython.display import SVG
    return SVG(render(plotables, 'svg', **kwargs))


plot = svg


def save(plotables, filename, **kwargs):
    ext = filename.split('.')[-1].lower()
    kwargs.setdefault('dpi', 600)  # force good quality
    return open(filename, 'wb').write(render(plotables, ext, **kwargs))
