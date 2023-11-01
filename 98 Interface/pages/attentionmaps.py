#This is the attention maps page
#There will be three figures with sliders here too slide across the Average attention maps
#There will also be a figure that will allow users to select any single test samplw to display, along with the attention map of it.

import dash
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd


dash.register_page(__name__,
                   path = "/attention_maps",
                   title="Attention Maps",
                   name="Attention Maps")

layout = html.Div([
    html.H1("Here be Attention Maps"),
    html.Div(children=["This is the attention maps page. We display mean attention maps here, sir!",
        html.Div("There will be no z ")]
    )
])