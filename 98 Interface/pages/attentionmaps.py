#This is the attention maps page
#There will be three figures with sliders here too slide across the Average attention maps per stage
#There will also be a figure that will allow users to select any single test sample to display, along with the attention map of it.

import dash
from dash import Dash, html, dcc
import plotly.express as px
from skimage import io


dash.register_page(__name__,
                   path = "/attention_maps",
                   title="Attention Maps",
                   name="Attention Maps")

def get_image(model,stage):
    im = io.imread("")

def average_images(current_stage):
    cur_stage = int(current_stage)
    next_stage  = int(current_stage+1)
    w1 = next_stage - current_stage 
    w2 = 1-w1
    # cur_stage * w1 + next_stage * w2

layout = html.Div([
    html.H1("Here be Attention Maps"),
    html.Div(children=["This is the attention maps page. We display mean attention maps here, sir!",
        html.Div("There will be as")]
    )
])