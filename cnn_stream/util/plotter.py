"""
Author: Foivos Tsimpourlas

In-house plotter module that plots data.
Based on plotly module
"""
import pathlib
import numpy as np
import itertools
from typing import Dict, List, Tuple

from plotly import graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px

from cnn_stream.util import log as l

example_formats = """
  margin = {'l': 0, 'r': 0, 't': 0, 'b': 0}   # Eliminates excess background around the plot (Can hide the title)
  plot_bgcolor = 'rgba(0,0,0,0)' or "#000fff" # Sets the background color of the plot
"""

def _get_generic_figure(**kwargs) -> go.Layout:
  """
  Constructor of a basic plotly layout.
  All keyword arguments are compatible with plotly documentation

  Exceptions:
    axisfont instead of titlefont. Reserved titlefont for title's 'font' size property.
  """
  # Title and axis names
  title  = kwargs.get('title', "")
  x_name = kwargs.get('x_name', "")
  y_name = kwargs.get('y_name', "")

  # Font sizes
  titlefont   = kwargs.get('titlefont', 38)
  axisfont    = kwargs.get('axisfont', 38)
  tickfont    = kwargs.get('tickfont', 32)

  # Plot line and axis options
  showline    = kwargs.get('showline',  True)
  linecolor   = kwargs.get('linecolor', 'black')
  gridcolor   = kwargs.get('gridcolor', "#eee")
  mirror      = kwargs.get('mirror',    False)
  showgrid    = kwargs.get('showgrid',  True)
  linewidth   = kwargs.get('linewidth', 2)
  gridwidth   = kwargs.get('gridwidth', 1)
  margin      = kwargs.get('margin', {'l': 40, 'r': 45, 't': 75, 'b': 0})
  x_tickangle = kwargs.get('x_tickangle', None)

  # Legend
  legend_x   = kwargs.get('legend_x', 1.02)
  legend_y   = kwargs.get('legend_y', 1.0)
  traceorder = kwargs.get('traceorder', "normal")
  legendfont = kwargs.get('legendfont', 24)

  # Background
  plot_bgcolor = kwargs.get('plot_bgcolor', "#fff")

  # Violin options
  violingap  = kwargs.get('violingap', 0)
  violinmode = kwargs.get('violinmode', 'overlay')

  title = dict(text = title, font = dict(size = titlefont))
  yaxis = dict(
             title     = y_name,   showgrid = showgrid,
             showline  = showline, linecolor = linecolor,
             mirror    = mirror,   linewidth = linewidth,
             gridwidth = gridwidth,
             tickfont  = dict(size = tickfont),
             titlefont = dict(size = axisfont)
          )
  xaxis = dict(
            title     = x_name,   showgrid = showgrid,
            showline  = showline, linecolor = linecolor,
            mirror    = mirror,   linewidth = linewidth,
            tickfont  = dict(size = tickfont),
            titlefont = dict(size = axisfont),
          )
  layout = go.Layout(
    plot_bgcolor = plot_bgcolor,
    margin       = margin,
    legend       = dict(x = legend_x, y = legend_y, traceorder = traceorder, font = dict(size = legendfont)),
    title        = title,
    xaxis        = xaxis,
    yaxis        = yaxis,
    violingap    = violingap,
    violinmode   = violinmode,
  )
  fig = go.Figure(layout = layout)
  if x_tickangle:
    fig.update_xaxes(tickangle = 45)
  fig.update_yaxes(automargin = True)
  return fig

def _write_figure(fig       : go.Figure,
                  plot_name : str,
                  path      : pathlib.Path = None,
                  **kwargs
                  ) -> None:
  """
  Write plotly image & and html file if path exists.
  Otherwise only show html file.
  """
  if path:
    path.mkdir(parents = True, exist_ok = True)
    outf = lambda ext: str(path / "{}.{}".format(plot_name, ext))
    try:
      fig.write_html (outf("html"))
    except ValueError:
      l.logger().warn("HTML plot failed")
    try:
      fig.write_image(outf("png"), width = kwargs.get('width'), height = kwargs.get('height'))
    except ValueError:
      l.logger().warn("PNG plot failed")
  else:
    fig.show()
  return

def GrouppedBars(groups    : Dict[str, Tuple[List, List]],
                 plot_name : str,
                 text      : List[str] = None,
                 path      : pathlib.Path = None,
                 **kwargs,
                 ) -> None:
  """
  Plots groups of bars.

  Groups must comply to the following format:
  groups = {
    'group_name': ([], [])
  }
  """
  # colors
  fig = _get_generic_figure(**kwargs)

  palette = itertools.cycle(px.colors.qualitative.T10)
  for group, (x, y) in groups.items():
    fig.add_trace(
      go.Bar(
        name = str(group),
        x = x,
        y = [(0.2+i if i == 0 else i) for i in y],
        marker_color = next(palette),
        textposition = kwargs.get('textposition', 'inside'),
        # text = text,
        text = ["" if i < 100 else "*" for i in y],
        textfont = dict(color = "white", size = 140),
      )
    )
  _write_figure(fig, plot_name, path, **kwargs)
  return
