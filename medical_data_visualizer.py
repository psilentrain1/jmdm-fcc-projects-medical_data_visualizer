import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")


# 2
df['overweight'] = ((df["weight"]/(df["height"]/100)**2)>25).astype(int)

# 3
def normalize_stats(x):
    if x == 1:
        return 0
    elif x > 1:
        return 1
    else:
        return x

df['cholesterol'] = df['cholesterol'].apply(normalize_stats)
df['gluc'] = df['gluc'].apply(normalize_stats)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=["cardio"], value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])
    # print(df_cat)

    # 6
    df_cat = df_cat.groupby(["cardio", 'variable', 'value']).size().reset_index()
    # print(df_cat)
    df_cat = df_cat.rename(columns={0: 'total'})
    # print(df_cat)

    # 7
    plot = sns.catplot(data=df_cat, kind="bar", x="variable", y="total", hue="value", col="cardio")

    # 8
    fig = plot.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
            (df['height'] >= df['height'].quantile(0.025)) &
            (df['height'] <= df['height'].quantile(0.975)) &
            (df['weight'] >= df['weight'].quantile(0.025)) &
            (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(16, 8))

    # 15
    sns.heatmap(corr, mask=mask, square=True, linewidths=0.5, annot=True, fmt="0.1f")


    # 16
    fig.savefig('heatmap.png')
    return fig
