#This is the attention maps page
#There will be three figures with sliders here too slide across the Average attention maps per stage
#There will also be a figure that will allow users to select any single test sample to display, along with the attention map of it.

import dash
from dash import Dash, html, dcc, callback, Input, Output,State
from dash.exceptions import PreventUpdate
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
import numpy as np
import pandas as pd


dash.register_page(__name__,
                   path = "/attention_maps",
                   title="Attention Maps",
                   name="Attention Maps")

df =  pd.read_excel('03 Reports/Summary.xlsx')
df = df.dropna()

models = df.iloc[:,0].dropna().unique()

def get_image(model,stage):
    "Recursively look for image at stage"
    folder = df[df.iloc[:,0] == model].AverageAttentionsPath.values[0]
    try:
        att_im = np.array(Image.open(folder+f"/stage_{stage}_atMean.png").resize((300,300)).convert("RGB"))
        mean_im = np.array(Image.open(folder+f"/stage_{stage}_imMean.png").resize((300,300)).convert("RGB"))
        w_mean = np.array(Image.open(folder+f"/stage_{stage}_wMean.png").resize((300,300)).convert("RGB"))
        not_found= False
    except FileNotFoundError:
        not_found=True
        stage +=1
        att_im,mean_im,w_mean,stage= get_image(model,stage)
    return att_im,mean_im,w_mean,stage

def average_images(current_stage):
    cur_stage = int(current_stage)
    next_stage  = int(current_stage+1)
    w1 = next_stage - current_stage 
    w2 = 1-w1
    # cur_stage * w1 + next_stage * w2


def layout(model_id=None,**qstrings):
    return html.Div([
        html.H1("Here be Attention Maps"),
        html.Div(children=["This is the attention maps page. We display mean attention maps here, sir!",
            html.Div("There will be as")]
        ),
        html.Div([
            dcc.Dropdown([m for m in models],id = "curModel",value=model_id),
            dcc.Graph(id="att_map")]),
        html.Div(
            dcc.Slider(min=0,max=9,marks={i: f"Stage {i}" for i in range(10)}, value=0,id = 'stageSlider'),
            html.Button(id="freezeButton")
        ),
        html.Div(
            dcc.Graph(id="freeze")
        )
    ])




@callback(
    Output("att_map",'figure'),
    Output("stageSlider",'min'),
    Input("stageSlider",'value'),
    Input("curModel","value"),
    Input("stageSlider",'min')
)
def update_figure(stage,model,slider_min):

    if stage-int(stage) == 0:
        att_im,mean_im,w_mean,found_stage = get_image(model,stage)
        figures = [
            px.imshow(mean_im),
            px.imshow(att_im),
            px.imshow(w_mean)
        ]
        fig = make_subplots(rows=1,cols=len(figures),subplot_titles = ["Mean Image of Stage", "Mean Attention Map", "Accuracy-Weighted Mean Map"])
        for i,figure in enumerate(figures):
            for trace in range(len(figure["data"])):
                fig.append_trace(figure["data"][trace],row=1,col=i+1)
        if not found_stage==stage:
            return fig,found_stage
        else:
            return fig,slider_min

    else:
        pass

