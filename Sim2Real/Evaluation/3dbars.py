import plotly.data as pdata
from barchart3d import barchart3d

df = pdata.gapminder()
df = df[df['year'] == 2007].sort_values(by='pop', ascending=False).head(10)
fig = barchart3d(
    df['country'].to_list(), (df['pop']/1e06).round(1).to_list(),
    'Top 10 most populous countries in 2007 [Gapminder]', 'Population, mln',
    colorscale='Bluered', opacity=0.6, flatshading=True)
fig.show()