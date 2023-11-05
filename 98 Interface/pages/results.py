#This is the attention maps page
#There will be three figures with sliders here too slide across the Average attention maps
#There will also be a figure that will allow users to select any single test samplw to display, along with the attention map of it.

import dash
from dash import Dash, html, dcc, clientside_callback, Input, Output, callback,State
import plotly.express as px
import pandas as pd
import os
from dash.exceptions import PreventUpdate
import dash_ag_grid as dag



df = pd.read_excel("./03 Reports/Summary.xlsx",index_col=[i for i in range(11)])
df = df.reset_index(level=[i for i in range(11)])
#Simplify Table
df = df.loc[[df[df.ModelId == n].index[0] for n in df.ModelId.unique()],df.columns[:9]]
df[["MeanAccStd","MeanStd"]]=df.MeanAccStd.str.split(" ",expand=True)
df = df.rename(columns = {"MeanAccStd":"MeanAcc"})
dash.register_page(__name__,
                   path = "/results",
                   title="Results",
                   name="Results")
import numpy as np


coldef0 = [
        {"headerName": "ModelId",
         "field":"ModelId",
         "cellRenderer":"StockLink"},
         
    ]
coldefrest = []
for i in df.columns[1:]:
    if i not in ["CLAHE","RandomAffine"]:
        coldefrest.append({"field": i,'sortable': True,'filter':'agNumberColumnFilter'})
    else:
        coldefrest.append({"field": i,'sortable': True,'filter':'agTextColumnFilter'})

grid = dag.AgGrid(
    columnDefs=coldef0+coldefrest,
    rowData=df.to_dict('records'),
    columnSize='sizeToFit',
    id='resTable'
)

layout = html.Div([
    dcc.Store(id="linkStore",data=None),
    dcc.Store(id="dummy"),
    html.H1("Table of Results"),
    html.Div(children=[
        html.Div(
        """This is a simplified view of the experiment results.\n
        You can access the full results table by clicking the button below.
        """),
        html.A(
            html.Button("Go to Full Results"),
            id='toFullResults',
            href="/results/fullresults"
        ),
        html.Div([
            dcc.Markdown(
                '''
            To go to the attention map analysis of a model, click the model name in the table.

            You can sort the values in the table by clicking the column names. Hold `Shift` for multi-sorting. You can also filter results using the menu next to each column name.
            '''
            )
            ,
        ]),
        html.Div(children=[
            #TODO: Add filtering option
            grid
            # dash.dash_table.DataTable(
            #     columns=[
            #         {'name':i,'id':i,"deletable":False} for i in df.columns if i!="id"
            #     ],
            #     data = df.to_dict(orient="records"),
            #                           style_cell={
            #                               'textAlign':'left'
            #                           },
            #                           style_data={
            #                               'whiteSpace':'normal'
            #                           },
            #                           sort_action='native',
            #                           sort_mode="multi",
            #                           id="resTable")
                                      
        ])

        ]   
        )
]
)

# @callback(
#         Output('modelStore','data'),
#         Output("linkStore",'data'),
#         Input("resTable",'active_cell'),
#         Input("resTable",'derived_virtual_row_ids'),
#         Input("resTable",'selected_row_ids')
# )
# def update_link(active_cell,row_ids,selected_row_ids):
#     if row_ids is None:
#         dff = df
#         row_ids = df["id"]
#     else:
#         dff = df.loc[row_ids]
#     if active_cell is None:
#         raise PreventUpdate
#     if not active_cell["column_id"]=='ModelId':
#         return PreventUpdate
#     return df.loc[active_cell["row"],active_cell["column_id"]],"/attention_maps"



# clientside_callback(
#     """
#     function(link){
#         const cur_host = window.location.host
#         window.location.pathname = '/' + link
#         return 0;
#     }
#     """,
#     Output("dummy","data"),
#     Input("linkStore",'data')
# )

