import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Imports and co.
    """)
    return


@app.cell
def _():
    import requests, json

    OLLAMA_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama3.1:8b"

    def ask_ollama(prompt, context=None, model=DEFAULT_MODEL, stream=True, options=None):
        """Send a prompt to your local Ollama model and get a response."""
        full = f"Context:\n{context}\n\nUser:\n{prompt}" if context else prompt
        payload = {"model": model, "prompt": full, "stream": stream}
        if options:
            payload["options"] = options

        r = requests.post(f"{OLLAMA_URL}/api/generate",
                          json=payload, stream=stream, timeout=600)
        r.raise_for_status()

        if stream:
            out = ""
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    out += data.get("response", "")
            return out.strip()
        else:
            return r.json().get("response", "").strip()

    return


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import marimo as mo
    return mo, pd


@app.cell
def _(pd):
    pd.read_csv('/Users/rasheedalbainy/Documents/Bootcamp/projects/moosic/Data/raw/3_spotify_5000_songs.csv')
    return


@app.cell
def _(pd):
    songs_df = pd.DataFrame(pd.read_csv('/Users/rasheedalbainy/Documents/Bootcamp/projects/moosic/Data/raw/3_spotify_5000_songs.csv'))
    return (songs_df,)


@app.cell
def _(songs_df):
    songs_df
    return


@app.cell
def _(songs_df):
    df=songs_df.copy()
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    # Data exploration & Cleaning
    """)
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    return


@app.cell
def _(df):
    df.set_index('index', inplace=True)
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df.isnull().values.any()
    return


@app.cell
def _(df):
    df.duplicated().sum()
    return


@app.cell
def _(df):
    df.loc[df.duplicated(subset=['name', 'artist'], keep=False)]


    return


@app.cell
def _(df):
    df.drop_duplicates(subset=['name', 'artist'], keep='first', inplace=True)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    print(df.columns)
    return


@app.cell
def _(df):
    df.columns = df.columns.str.strip()
    return


@app.cell
def _(df):
    df.loc[ 
        (df['danceability'] ==  0) &
        (df['speechiness'] ==  0) &
        (df['tempo'] ==  0)
    ]

    return


@app.cell
def _(df):
    df.loc[ 
        (df['tempo'] ==  0)
    ]
    return


@app.cell
def _(df):
    df.loc[ 
        (df['tempo'] ==  0)
    ]

    return


@app.cell
def _(df):
    df.drop(629,axis=0, inplace=True)
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    import random
    for i in df.index:
        if df.loc[i, 'speechiness'] > 0.43:
            df.loc[i, 'speechiness'] = round(random.uniform(0.35, 0.42),2)
    
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df.drop(['type', 'id', 'html'], axis=1, inplace=True)
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df.reset_index(drop=True, inplace=True)

    return


@app.cell
def _(df):
    df
    return


if __name__ == "__main__":
    app.run()
