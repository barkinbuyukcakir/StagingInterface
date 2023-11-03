#This is the attention maps page
#There will be three figures with sliders here too slide across the Average attention maps
#There will also be a figure that will allow users to select any single test samplw to display, along with the attention map of it.

import dash
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import os



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

def multiindex_table(df):
    # storing rowSpan values for every cell of index;
    # if rowSpan==0, html item is not going to be included
    pos = np.diff(df.index.codes, axis=1, prepend=-1)
    for row in pos:
        counts = np.diff(np.flatnonzero(np.r_[row, 1]))
        row[row.astype(bool)] = counts

    # filling up header of table;
    column_names = df.columns.values
    headTrs = html.Tr([html.Th(n) for n in df.index.names] +
                      [html.Th(n) for n in column_names])
    # filling up rows of table;
    bodyTrs = []
    for rowSpanVals, idx, col in zip(pos.T, df.index.tolist(), df.to_numpy()):
        rowTds = []
        for name, rowSpan in zip(idx, rowSpanVals):
            if rowSpan != 0:
                rowTds.append(html.Td(name, rowSpan=rowSpan))
        for name in col:
            rowTds.append(html.Td(name))
        bodyTrs.append(html.Tr(rowTds))

    table = html.Table([
        html.Thead(headTrs),
        html.Tbody(bodyTrs)
    ])
    return table

layout = html.Div([
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
            """To go to the attention map of analysis of the model, click the model name in the table:""",
        ]),
        html.Div(children=[
            #TODO: Add filtering option
            dash.dash_table.DataTable(data = df.to_dict(orient="records"),
                                      style_cell={
                                          'textAlign':'left'
                                      },
                                      style_data={
                                          'whiteSpace':'normal'
                                      },
                                      sort_action='native')
                                      
        ])

        ]   
        )
]
)

"""
<iframe width="700" height="400" frameborder="0" scrolling="no" src="https://kuleuven-my.sharepoint.com/personal/barkin_buyukcakir_kuleuven_be/_layouts/15/Doc.aspx?sourcedoc={4b1b03ec-8eca-4399-bc4f-ccec8030c312}&action=embedview&Item='Sheet1'!A%3AO&wdHideGridlines=True&wdDownloadButton=True&wdInConfigurator=True&wdInConfigurator=True"></iframe>
<iframe width="402" height="346" frameborder="0" scrolling="no" src="https://kuleuven-my.sharepoint.com/personal/barkin_buyukcakir_kuleuven_be/_layouts/15/Doc.aspx?sourcedoc={4b1b03ec-8eca-4399-bc4f-ccec8030c312}&action=embedview&wdHideHeaders=True&wdInConfigurator=True&wdInConfigurator=True"></iframe>
"""