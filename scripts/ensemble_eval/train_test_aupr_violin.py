#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
raw = pd.read_csv("train_test_aupr_violin.csv")

# %%
# select the data
df = raw.loc[raw['partition'].isin(["test","full"])]
df = df.loc[df['percent'].isin([0.5, 1.0, 2.0, 5.0, 10.0])]
# df = df.loc[df['percent'].isin([10.0, 20.0, 30.0, 40.0, 50.0])]

#%%
# plot with grous:  group "part_size", subgroup "partition"
sns.set(style="darkgrid")

#https://stackoverflow.com/questions/70767421/creating-a-violin-plot-with-seaborn-without-x-and-y-values-but-with-hue
ax = sns.violinplot(x="percent", y="aupr", hue="partition", data=df, split=True, palette="Pastel1")
ax.set(xlabel='Percent Ground Truth as Training Set', ylabel='AUPR')
plt.axhline(y=0.3503, color='r', linestyle='-', label="ARACNe-AP")
plt.axhline(y=0.3938, color='b', linestyle='-', label="R^4")
plt.axhline(y=0.4198, color='g', linestyle='-', label="Mu from Full")
ax.legend()
plt.legend(loc='lower right')
# plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

# %%
fig = ax.get_figure()
fig.savefig('/home/tpan/data/GRN_paper/revision/p1/train_test_combo.pdf')

# %%
raw2 = pd.read_csv("train_test_aupr_violin_tf.csv")

# %%
# select the data
df2 = raw2.loc[raw2['partition'].isin(["test","full"])]
df2 = df2.loc[df2['percent'].isin([0.5, 1.0, 2.0, 5.0, 10.0])]
# df = df.loc[df['percent'].isin([10.0, 20.0, 30.0, 40.0, 50.0])]

#%%
# plot with grous:  group "part_size", subgroup "partition"

#https://stackoverflow.com/questions/70767421/creating-a-violin-plot-with-seaborn-without-x-and-y-values-but-with-hue
ax2 = sns.violinplot(x="percent", y="aupr", hue="partition", data=df2, split=True, palette="Pastel1")
ax2.set(xlabel='Percent Ground Truth as Training Set', ylabel='AUPR')
plt.axhline(y=0.3667, color='r', linestyle='-', label="ARACNe-AP")
plt.axhline(y=0.4988, color='b', linestyle='-', label="R^4")
plt.axhline(y=0.5069, color='g', linestyle='-', label="Mu from Full")
ax2.legend()
plt.legend(loc='lower right')
# plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

# %%
fig2 = ax2.get_figure()
fig2.savefig('/home/tpan/data/GRN_paper/revision/p1/tf_train_test_combo.pdf')



# %%
sns.set(rc={'figure.figsize':(6,3.75)})
sns.set_style("ticks")


fig3, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [10, 9]})
sns.violinplot(ax=ax1, x="percent", y="aupr", hue="partition", data=df, split=True, palette="Pastel1")
ax1.set(ylabel='AUPRC')
ax1.axhline(y=0.3503, color='r', linestyle='-', label="ARACNe-AP")
ax1.axhline(y=0.3938, color='b', linestyle='-', label="R^4")
ax1.axhline(y=0.4198, color='g', linestyle='-', label="M^4, full")
#ax1.get_legend().remove()
ax1.legend(loc="upper left")
ax1.set(xlabel=None)
ax1.set_title('GCN')

sns.violinplot(ax=ax2, x="percent", y="aupr", hue="partition", data=df2, split=True, palette="Pastel1")
ax2.axhline(y=0.3667, color='r', linestyle='-', label="ARACNe-AP")
ax2.axhline(y=0.4988, color='b', linestyle='-', label="R^4")
ax2.axhline(y=0.5069, color='g', linestyle='-', label="M^4, full")
#ax2.legend(loc='center right')
ax2.get_legend().remove()
ax2.set(xlabel=None, ylabel=None)
ax2.set_title('GRN')

plt.gcf().text(0.5,0.01,"Percent Ground Truth as Training Set", ha="center")
fig.tight_layout()

fig3.savefig('/home/tpan/data/GRN_paper/revision/p1/both_train_test_combo.pdf')

# %%
