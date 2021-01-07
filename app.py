# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed Nov 18 18:07:05 2020

# @author: holdenbruce
# """


# # conda install -c plotly plotly
# # conda install -c conda-forge dash


# # -*- coding: utf-8 -*-

# # Run this app with `python app.py` and
# # visit http://127.0.0.1:8050/ in your web browser.

# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# import plotly.express as px
# import pandas as pd

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# colors = {
#     'background': '#111111',
#     'text': '#7FDBFF'
# }

# # assume you have a "long-form" data frame
# # see https://plotly.com/python/px-arguments/ for more options
# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

# fig.update_layout(
#     plot_bgcolor=colors['background'],
#     paper_bgcolor=colors['background'],
#     font_color=colors['text']
# )

# app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
#     html.H1(
#         children='Hello Dash',
#         style={
#             'textAlign': 'center',
#             'color': colors['text']
#         }
#     ),

#     html.Div(children='Dash: A web application framework for Python.', style={
#         'textAlign': 'center',
#         'color': colors['text']
#     }),

#     dcc.Graph(
#         id='example-graph-2',
#         figure=fig
#     )
# ])

# if __name__ == '__main__':
#     app.run_server(debug=True)
             
#              Note:
# The layout is composed of a tree of "components" like html.Div and dcc.Graph.
# The dash_html_components library has a component for every HTML tag. The html.H1(children='Hello Dash') component generates a <h1>Hello Dash</h1> HTML element in your application.
# Not all components are pure HTML. The dash_core_components describe higher-level components that are interactive and are generated with JavaScript, HTML, and CSS through the React.js library.
# Each component is described entirely through keyword attributes. Dash is declarative: you will primarily describe your application through these attributes.
# The children property is special. By convention, it's always the first attribute which means that you can omit it: html.H1(children='Hello Dash') is the same as html.H1('Hello Dash'). Also, it can contain a string, a number, a single component, or a list of components.
# The fonts in your application will look a little bit different than what is displayed here. This application is using a custom CSS stylesheet to modify the default styles of the elements. You can learn more in the css tutorial, but for now you can initialize your app with









# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# import pandas as pd

# df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/c78bf172206ce24f77d6363a2d754b59/raw/c353e8ef842413cae56ae3920b8fd78468aa4cb2/usa-agricultural-exports-2011.csv')


# def generate_table(dataframe, max_rows=10):
#     return html.Table([
#         html.Thead(
#             html.Tr([html.Th(col) for col in dataframe.columns])
#         ),
#         html.Tbody([
#             html.Tr([
#                 html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
#             ]) for i in range(min(len(dataframe), max_rows))
#         ])
#     ])


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# # colors = {
# #     'background': '#111111',
# #     'text': '#7FDBFF'
# # }

# app.layout = html.Div(children=[
#     html.H4(children='US Agriculture Exports (2011)'),
#     generate_table(df)
# ])

# if __name__ == '__main__':
#     app.run_server(debug=True)


#             Second note:
# In this example, we modified the inline styles of the html.Div and html.H1 components with the style property.
# html.H1('Hello Dash', style={'textAlign': 'center', 'color': '#7FDBFF'}) is rendered in the Dash application as <h1 style="text-align: center; color: #7FDBFF">Hello Dash</h1>.
# There are a few important differences between the dash_html_components and the HTML attributes:
# The style property in HTML is a semicolon-separated string. In Dash, you can just supply a dictionary.
# The keys in the style dictionary are camelCased. So, instead of text-align, it's textAlign.
# The HTML class attribute is className in Dash.
# The children of the HTML tag is specified through the children keyword argument. By convention, this is always the first argument and so it is often omitted.
# Besides that, all of the available HTML attributes and tags are available to you within your Python context.






import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')
# df.head()
# df.columns
# fig = px.scatter(df, x="gdp per capita", y="life expectancy",
                 # size="population", color="continent", hover_name="country",
                 # log_x=True, size_max=60)



df = pd.read_csv('HaitiPixels.csv')
fig = px.scatter(df, x=['Red', 'Green', 'Blue'], y="Class",
                 color="Class", hover_name="Class",
                 log_x=True, size_max=60)

app.layout = html.Div([
    dcc.Graph(
        id='life-exp-vs-gdp',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)