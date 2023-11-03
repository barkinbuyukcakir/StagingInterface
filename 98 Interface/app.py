import dash
from dash import Dash, html, dcc



app = Dash(__name__, use_pages=True)

# app.layout = html.Div([
#     # html.H1('Dental Staging with VIT'),
#     html.Div(children=[
#         dcc.Link(f"{page['name']}", href=page["relative_path"]) for page in dash.page_registry.values()
#         ],className="sidenav"),
#     html.Div(className="main",children=[
#         html.H1("Dental Staging"),
#         html.Div("Main Content here?")
#     ]),
#     dash.page_container
# ],className="sidenav")
app.layout = html.Div([
    html.Div(children = [
                html.Div("Dental Staging with VIT",className="app-header--title")
            ],className="app-header"),
    html.Div(children=[
        html.A(f"{page['name']}", href=page["relative_path"],className="sidenav--a") for page in dash.page_registry.values() if page["name"] != "Full Results"
    ],
    className="sidenav"),
    html.Div(children=[
            dash.page_container],
            className="main")
])

if __name__ == '__main__':
    app.run(debug=True)