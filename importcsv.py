from math import ceil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("data/Untitled form (Responses) - Form Responses 1.csv")
print(data)
feats = [i for i in range(32) if i not in [10, 18, 19, 23, 30]]
columns = []
for f in feats:
    columns.append(f"Cat-{f}")
    columns.append(f"Cow-{f}")
    columns.append(f"Spider-{f}")
data.columns = columns
print(data)


def barplots(classes, c1, c2, c3, c4, c5):
    products = ['chew', 'cigarette', 'hookah', 'cigar', 'e_cigarette']
    per_totals = [3.2, 3.5, 0.9, 1.1, 6.7]
    per_users = [36.1, 26, 20, 16, 31]
    classes_new = [c.split('-')[0] for c in classes]
    # sort both lists together
    # per_1, classes_1 = zip(c1, classes)
    # per_2, classes_2 = zip(c2, classes)
    # per_3, classes_3 = zip(c3, classes)
    # per_4, classes_4 = zip(c4, classes)
    # per_5, classes_5 = zip(c5, classes)

    n_groups = len(c1)

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 4)
    index = np.arange(n_groups)
    bar_width = 0.15

    c1_b = plt.bar(index - bar_width * 2, c1, bar_width, label=0)
    c2_b = plt.bar(index - bar_width, c2, bar_width, label=1)
    c3_b = plt.bar(index, c3, bar_width, label=2)
    c3_b = plt.bar(index + bar_width, c4, bar_width, label=3)
    c3_b = plt.bar(index + bar_width * 2, c5, bar_width, label=4)

    # list of ticks: combine the ticks from both groups
    # followed by the list of corresponding labels
    # note that matplotlib takes care of sorting both lists while maintaining the correspondence

    plt.xticks(index,
               # tick positions
               classes_new,  # label corresponding to each tick
               rotation=30)

    # plt.xlabel('Tobacco Product')
    # plt.ylabel('Percent of Users')
    # plt.title('Figure 2. Tobacco Use for 16-18 Year Olds')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"feat-{classes[0].split('-')[-1]}-answers.png")
    plt.close(fig)

cats = []
spiders = []
cows = []
for f in feats:
    d = data[[f"Cat-{f}", f"Cow-{f}", f"Spider-{f}"]]
    # print(d),
    newd = []

    for row in d.iterrows():
        # print(row)
        rn, series = row
        nan_in_df_all = series.isnull().values.all()
        nan_in_df_any = series.isnull().values.any()
        # print(nan_in_df_all)
        # print(nan_in_df_any)
        if nan_in_df_all:
            continue
        if nan_in_df_any:
            series = series.fillna(0)
        newd.append(series)
        series_sum = series.sum()
        if series_sum > 2 and series_sum != 4:
            pd_series = pd.Series(series.values / series_sum * 4, index=series.keys())
            series.update(pd_series.round(0))
            # print(series)

    dfnew = pd.DataFrame(newd)
    dfcounts = {i: [0 for _ in range(len(dfnew.columns))] for i in range(5)}
    for j in dfcounts:
        for i in range(len(dfcounts[j])):
            dfcounts[j][i] = len(dfnew[dfnew[dfnew.columns[i]] == j])
    barplots(list(dfnew.keys()), dfcounts[0], dfcounts[1], dfcounts[2], dfcounts[3], dfcounts[4])
    dfcounts = pd.DataFrame(dfcounts)

    # print(dfcounts)

    # print(dfnew)

    m = dfnew.mean()
    cats.append((m[f"Cat-{f}"],f))
    cows.append((m[f"Cow-{f}"],f))
    spiders.append((m[f"Spider-{f}"],f))
cats = sorted(cats)
cats = [(0.0, i ) for i in [10, 18, 19, 23, 30]] + cats
cows = sorted(cows)
cows = [(0.0, i ) for i in [10, 18, 19, 23, 30]] + cows
spiders = sorted(spiders)
spiders = [(0.0, i ) for i in [10, 18, 19, 23, 30]] + spiders
lcats = len(cats)-5
cats_lp = cats[:lcats//3+5]
cats_mp = cats[lcats//3+5:lcats//3*2+5]
cats_hp = cats[lcats//3*2+5:]
print("cats_lp = ",[c[1] for c in cats_lp])
print("cats_mp = ",[c[1] for c in cats_mp])
print("cats_hp = ",[c[1] for c in cats_hp])
lcows = len(cows)-5
cows_lp = cows[:lcows//3+5]
cows_mp = cows[lcows//3+5:lcows//3*2+5]
cows_hp = cows[lcows//3*2+5:]
print("cows_lp = ",[c[1] for c in cows_lp])
print("cows_mp = ",[c[1] for c in cows_mp])
print("cows_hp = ",[c[1] for c in cows_hp])

lspiders = len(spiders)-5
spiders_lp = spiders[:lspiders//3+5]
spiders_mp = spiders[lspiders//3+5:lspiders//3*2+5]
spiders_hp = spiders[lspiders//3*2+5:]
print("spiders_lp = ",[c[1] for c in spiders_lp])
print("spiders_mp = ",[c[1] for c in spiders_mp])
print("spiders_hp = ",[c[1] for c in spiders_hp])
bin_spacing = 1/10
def roundtobin(num):
    return ceil(num/bin_spacing)*bin_spacing
plt.hist([j for j,_ in cats],np.arange(min(cats)[0], max(cats)[0]+bin_spacing, step=bin_spacing))
fig = plt.gcf()
fig.set_size_inches(12, 4)
y_cat=plt.gca().get_ylim()[-1]+0.1
catline1 = roundtobin(max(cats_lp)[0])
#plt.axvline(x=catline1, linewidth=1, color='k')
#plt.text(x= catline1/2-1/6, y=y_cat, s="LP", )
catline2 = roundtobin(max(cats_mp)[0])
#plt.axvline(x=catline2, linewidth=1, color='k')
#plt.text(x= (catline1+catline2)/2-1/6, y=y_cat, s="MP")

#plt.text(x= (4+catline2)/2-1/6, y=y_cat, s="HP")
plt.savefig("hist-cats.png")
plt.close()

plt.hist([j for j,_ in cows],np.arange(min(cows)[0], max(cows)[0]+bin_spacing, step=bin_spacing))
fig = plt.gcf()
fig.set_size_inches(12, 4)
y_cow=plt.gca().get_ylim()[-1]+0.1
cowline1 = roundtobin(max(cows_lp)[0])
#plt.axvline(x=cowline1, linewidth=1, color='k')
#plt.text(x= cowline1/2-1/6, y=y_cow, s="LP", )
cowline2 = roundtobin(max(cows_mp)[0])
#plt.axvline(x=cowline2, linewidth=1, color='k')
#plt.text(x= (cowline1+cowline2)/2-1/6, y=y_cow, s="MP")

#plt.text(x= (4+cowline2)/2-1/6, y=y_cow, s="HP")
plt.savefig("hist-cows.png")
plt.close()

plt.hist([j for j,_ in spiders],np.arange(min(spiders)[0], max(spiders)[0]+bin_spacing, step=bin_spacing))
y_spider=plt.gca().get_ylim()[-1]+0.1
fig = plt.gcf()
fig.set_size_inches(12, 4)

spiderline1 = roundtobin(max(spiders_lp)[0])
#plt.axvline(x=spiderline1, linewidth=1, color='k')
#plt.text(x= spiderline1/2-1/6, y=y_spider, s="LP", )
spiderline2 = roundtobin(max(spiders_mp)[0])
#plt.axvline(x=spiderline2, linewidth=1, color='k')
#plt.text(x= (spiderline1+spiderline2)/2-1/6, y=y_spider, s="MP")

#plt.text(x= (4+spiderline2)/2-1/6, y=y_spider, s="HP")
plt.savefig("hist-spiders.png")
plt.close()

pass