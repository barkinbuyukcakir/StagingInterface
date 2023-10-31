#This is the home page of the application
#We will just give general information on what's available
#and provide links to the other pages

import dash
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd


dash.register_page(__name__)

layout = html.Div([
    html.H1("This is the home page of the app."),
    html.Div(
        html.Div(children="Home page here",
                 style={
                    'textAlign':'center'
                    })
        )
])