# PCA PAge
import dash
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

dash.register_page(__name__,
                   path = "/pca",
                   title="PCA Plot",
                   name="PCA Plot")

layout = html.Div([
    html.H1("This is the PCA of the app."),
    html.Div(
        html.Div(children="There will be an interactive PCA plot here. You'll be able to click on a stage to highlight it against others.",
                 style={
                    'textAlign':'center'
                    })
        )
])

