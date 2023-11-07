#This is the attention maps page
#There will be three figures with sliders here too slide across the Average attention maps
#There will also be a figure that will allow users to select any single test samplw to display, along with the attention map of it.

import dash
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import os



df = pd.read_excel("./03 Reports/Summary.xlsx",index_col=[i for i in range(11)])
dff = df.reset_index(level=[i for i in range(11)])
dff.drop(columns=["AverageAttentionsPath"],inplace=True)
dash.register_page(__name__,
                   path = "/results/fullresults",
                   title="Full Results",
                   name="Full Results")
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
        dcc.Markdown(
            '''
        You can access the full results on this page. 
        It's recommended you download the Excel file for better viewing by using the download button on the bottom right.
'''
        ),
        html.Iframe(src="https://kuleuven-my.sharepoint.com/personal/barkin_buyukcakir_kuleuven_be/_layouts/15/Doc.aspx?sourcedoc={4b1b03ec-8eca-4399-bc4f-ccec8030c312}&action=embedview&Item=Table1&wdHideGridlines=True&wdDownloadButton=True&wdInConfigurator=True&wdInConfigurator=True",
        style={
            "height":'700px',
            "width":"100%",
        }
        )
        ]   
        )
]
)

"""
<iframe width="402" height="346" frameborder="0" scrolling="no" src="https://kuleuven-my.sharepoint.com/personal/barkin_buyukcakir_kuleuven_be/_layouts/15/Doc.aspx?sourcedoc={4b1b03ec-8eca-4399-bc4f-ccec8030c312}&action=embedview&Item=Table1&wdHideGridlines=True&wdDownloadButton=True&wdInConfigurator=True&wdInConfigurator=True"></iframe>
"""