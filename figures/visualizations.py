import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from scipy import optimize
from scipy import stats
from scipy.stats import gamma
import os

dirname = os.path.dirname(__file__)

#############################################################################
# ORIGINAL SNAKE VERSION ANALYSIS
#############################################################################

# functions for data cleaning and preprocessing
def get_dataframe(raw_df):
  df = raw_df.apply(lambda x: x.iloc[0].split()[-6:], axis=1, result_type='expand')
  df = df.rename({i: col_names[i] for i in range(6)}, axis=1)
  df = df.astype(int)
  return df

def where_food(df):
  food_pos_ghost = df[df['ghost_ate']==1][['ghost_x', 'ghost_y']].copy()
  food_pos_ghost.rename({'ghost_x': 'x', 'ghost_y': 'y'}, axis=1, inplace=True)
  food_pos_player = df[df['player_ate']==1][['player_x', 'player_y']].copy()
  food_pos_player.rename({'player_x': 'x', 'player_y': 'y'}, axis=1, inplace=True)
  return pd.concat([food_pos_ghost, food_pos_player])

def where_food_when1(df):
  food_pos_ghost = df[df['ghost_ate']==1][['ghost_x', 'ghost_y']].copy()
  food_pos_ghost.rename({'ghost_x': 'x', 'ghost_y': 'y'}, axis=1, inplace=True)
  return food_pos_ghost

col_names = ['player_x', 'player_y', 'player_ate', 'ghost_x', "ghost_y", "ghost_ate"]
food_list = []
dfs = []

for i in range(1,11):
  exec("raw_df{} = pd.read_csv(os.path.join(dirname, '../data/data{}.txt'), header=0, names=['placeholder'])".format(i,i))
  exec("df{} = get_dataframe(raw_df{})".format(i,i))
  exec("dfs.append(df{})".format(i))
  exec("food{} = where_food(df{})".format(i,i))
  exec("food_list.append(food{})".format(i))

jumps_food = []

for i in range(1,21):
  exec("jumps_{} = pd.read_csv(os.path.join(dirname, '../data/{}.txt'), header=0, names=['placeholder'])".format(i,i))
  exec("jumps_{} = get_dataframe(jumps_{})".format(i,i))
  exec("jumps_{} = where_food(jumps_{})".format(i,i))
  exec("jumps_food.append(jumps_{})".format(i))

jumps_food_dfs = food_list + jumps_food
whole_food = pd.concat(jumps_food_dfs)

left, width = 0.1, 0.65
bottom, height= 0.1, 0.65
spacing = 0.005
rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]
plt.figure(figsize=(8,8))
ax_scatter = plt.axes(rect_scatter)
ax_scatter.tick_params(direction="in")
ax_histx = plt.axes(rect_histx)
ax_histx.tick_params(direction='in', labelbottom=False)
ax_histy = plt.axes(rect_histy)
ax_histy.tick_params(direction='in', labelleft=False)
whole_food.plot.scatter(x="x", y="y", ax=ax_scatter)
whole_food.hist(column="x", ax=ax_histx)
whole_food.hist(column="y", ax=ax_histy, orientation="horizontal")
plt.savefig(os.path.join(dirname,"../figures/scatter_hist.png"))

plt.clf()
freq = pd.Series([8, 11, 13, 11, 7, 1, 1], index=[0, 1, 2, 3, 4, 5, 6])
freq=freq/sum(freq)
freq.plot(kind='bar', rot=0)
plt.xlabel('No. of targets on a row')
plt.ylabel('Frequency')
plt.savefig(os.path.join(dirname, '../figures/one_row_freq.png'))


######################################################################################

# SNAKE MOTION

jumps_whole_dfs = []
jumps = []

for i in range(1,21):
  exec("jumps_{}_m = pd.read_csv(os.path.join(dirname, '../data/{}.txt'), header=0, names=['placeholder'])".format(i,i))
  exec("jumps_{}_m = get_dataframe(jumps_{}_m)".format(i,i))
  exec("jumps.append(jumps_{}_m)".format(i))

jumps_whole_dfs = dfs + jumps

def get_direction(last_row, row, who='ghost_'):
  x_step = row[who+'x'] - last_row[who+'x']
  y_step = row[who+'y'] - last_row[who+'y']
  if x_step > 0:
    direc = 'r'
  elif x_step < 0:
    direc = 'l'
  elif y_step > 0:
    direc = 'u'
  elif y_step < 0:
    direc = 'd'
  else:
    direc = None
  return direc


def direction(df):
  copy_df = df.copy()
  direction = [get_direction(df.loc[prev_index], df.loc[index]) for prev_index, index in zip(df.index[:-1], df.index[1:]) ]
  copy_df['direction'] = [direction[0]] + direction
  fix_direc = copy_df[copy_df['ghost_ate']==0].copy().reset_index()
  fix_direc['turn'] = ['0'] + [1 if prev_dir is not dir else 0 
                             for prev_dir, dir in zip(fix_direc.iloc[:-1]['direction'], fix_direc.iloc[1:]['direction'])]
  turns = fix_direc[fix_direc['turn']==1].index
  time_between_turns = turns[1:] - turns[:-1]
  return time_between_turns

plt.clf()
whole_data = pd.concat(jumps_whole_dfs)
whole_data = whole_data.astype(int)
directions_list = [direction(df) for df in jumps_whole_dfs]
whole_time_turns = np.concatenate(directions_list)
whole_time_turns = whole_time_turns[whole_time_turns != 1]
_, bins, _= plt.hist(whole_time_turns, density=True)
mu, sigma = sp.stats.expon.fit(whole_time_turns)
best_fit = sp.stats.expon.pdf(bins, mu, sigma)
plt.plot(bins, best_fit)
plt.ylabel("Histogram")
plt.xlabel("Time between jumps")
plt.tick_params(direction="in")
plt.savefig(os.path.join(dirname, "../figures/jump_distribution.png"))



#############################################################################
# MODIFIED SNAKE VERSION ANALYSIS
#############################################################################

# TIME BETWEEN TURNS
for lamda in range(3,11):

    #Read data
    times_between_turns = pd.read_csv(os.path.join(dirname, f'../data/time_between_turns_500_{lamda}.txt'))

    # Fit data
    n_bins = 45
    bins = np.arange(0, n_bins, 1)
    sp.stats.expon.fit(times_between_turns)
    loc, scale =  sp.stats.expon.fit(times_between_turns)
    best_fit = sp.stats.expon.pdf(bins, loc, scale)
    
    # Plot 
    plt.clf()
    times_between_turns.hist(bins=bins, density=True, label='data')
    plt.plot(bins, best_fit, label='fit')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title(f'$\lambda = 1/{lamda}$')
    plt.ylim([0, 0.4])
    plt.tick_params(direction="in")
    plt.savefig(os.path.join(dirname, f'../figures/time_between_jumps{lamda}.png'))

##########################################################################

# COVER TIME

for lamda in range(3,11):
    exec(f"f{lamda} = open(os.path.join(dirname, '../data/cover_time_500_{lamda}.txt'))")

cover_3 = ([float(x) for x in f3])
cover_4 = ([float(x) for x in f4])
cover_5 = ([float(x) for x in f5])
cover_6 = ([float(x) for x in f6])
cover_7 = ([float(x) for x in f7])
cover_8 = ([float(x) for x in f8])
cover_9 = ([float(x) for x in f9])
cover_10 = ([float(x) for x in f10])

cover = [cover_3,cover_4,cover_5,cover_6,cover_7,cover_8,cover_9,cover_10]

fig, ax = plt.subplots()
ax.boxplot(cover)
plt.tick_params(direction='in')
positions = (1,2,3,4,5,6,7,8)
labels = ("1/3", "1/4", "1/5", "1/6", "1/7", "1/8", "1/9", "1/10")
plt.xticks(positions, labels)
plt.xlabel("$\lambda$")
plt.ylabel("Cover time")
plt.savefig(os.path.join(dirname, f'../figures/cover_time.png'))

##################################################################

# SURVIVAL TIME

bins = np.arange(0, 200, 1)
for i in range(3, 11):
  exec(f"g{i} = open(os.path.join(dirname, '../data/survival_time_10000_{i}.txt'))")
  exec(f"survival_{i} = ([float(x) for x in g{i}])")


# Histograms
histogram3 = np.histogram(survival_3, bins=bins, density=True)[0]
histogram4 = np.histogram(survival_4, bins=bins, density=True)[0]
histogram5 = np.histogram(survival_5, bins=bins, density=True)[0]
histogram6 = np.histogram(survival_6, bins=bins, density=True)[0]
histogram7 = np.histogram(survival_7, bins=bins, density=True)[0]
histogram8 = np.histogram(survival_8, bins=bins, density=True)[0]
histogram9 = np.histogram(survival_9, bins=bins, density=True)[0]
histogram10 = np.histogram(survival_10, bins=bins, density=True)[0]

# Fit 
bins = 0.5*(bins[1:] + bins[:-1])
shape4, loc4, scale4 = sp.stats.gamma.fit(survival_4)
shape10, loc10, scale10 = sp.stats.gamma.fit(survival_10) 
best_fit4 = sp.stats.gamma.pdf(bins, shape4, loc4, scale4)
best_fit10 = sp.stats.gamma.pdf(bins, shape10, loc10, scale10)

# Plot
fig, axs = plt.subplots(ncols=2, sharey=True)
axs[0].plot(bins, histogram4, label='rate $\lambda=1/4$')
axs[1].plot(bins, histogram10, label='rate $\lambda=1/10$')
axs[0].plot(bins, best_fit4)
axs[1].plot(bins, best_fit10)
axs[0].legend(loc="best")
axs[1].legend(loc="best")
axs[0].set_xlabel('t')
axs[1].set_xlabel('t')
axs[0].set_ylabel('f(t)')
axs[0].tick_params(direction='in')
axs[1].tick_params(direction='in')

plt.savefig(os.path.join(dirname, f'../figures/survival_time_lambdas_gamma.png'))

# Save means
means = np.zeros(8)
for i in range(8):
    exec(f"shape, loc, scale = sp.stats.gamma.fit(survival_{i+3})")
    means[i] =sp.stats.gamma.mean(shape, loc, scale)

with open(os.path.join(dirname, '../results/survival_time_expon_mean.txt'), 'w') as file:
    for element in means:
        file.write(str(element) + "\n")


# Geometric dist approximation for survival time
geometric_dist = np.array([0.04*(1-0.04)**t for t in range(200)])
shape, loc, scale = sp.stats.gamma.fit(geometric_dist)
x = np.arange(0, 200, 1)
best_fit10 = sp.stats.gamma.pdf(x, shape10, loc10, scale10)
best_fit4 = sp.stats.gamma.pdf(x, shape4, loc4, scale4)

# Plot
fig, ax = plt.subplots()
ax.plot(x, geometric_dist, label='geometric dist $p=0.04$')
ax.plot(x, best_fit10, label='Survival time for  $\lambda = 1/10$')
ax.plot(x, best_fit4, label='Survival time for $\lambda = 1/4$')
ax.legend(loc="best")
ax.set_xlabel('t')
ax.set_ylabel('f(t)')
ax.tick_params(direction='in')

# Save plot
plt.savefig(os.path.join(dirname, f'../figures/geometric_comparison.png'))

########################################################################################################

# WINNING TIME

# Read data
for lamda in range(3, 11):
    exec(f"h = open(os.path.join(dirname, '../data/winning_time_1000_{lamda}.txt'), 'r')")
    exec(f"winning_{lamda} = ([float(x) for x in h])")


# Data transformation
bins = np.arange(0, 1500, 15)
histo3 = np.histogram(winning_3, bins=bins, density=True)[0]
histo4 = np.histogram(winning_4, bins=bins, density=True)[0]
histo5 = np.histogram(winning_5, bins=bins, density=True)[0]
histo6 = np.histogram(winning_6, bins=bins, density=True)[0]
histo7 = np.histogram(winning_7, bins=bins, density=True)[0]
histo8 = np.histogram(winning_8, bins=bins, density=True)[0]
histo9 = np.histogram(winning_9, bins=bins, density=True)[0]
histo10 = np.histogram(winning_10, bins=bins, density=True)[0]

bins = 0.5*(bins[1:] + bins[:-1])

# Plot
file = open(os.path.join(dirname, '../results/wiining_time_fit_params.txt'), 'w')
for i in [3,4,5,6,7,8,9,10]: # se pueden poner varios, pero creo que es mejor solo uno
    plt.clf()
    exec(f"plt.plot(bins, histo{i}, label='$\lambda = 1/{i}$')")
    exec(f"mu, var = sp.stats.norm.fit(winning_{i})")
    file.write(' '.join((str(i), str(mu), str(var),'\n')))
    exec(f"plt.plot(bins, sp.stats.norm.pdf(bins, loc=mu, scale=var), label='Normal fit')")
    plt.legend(loc="best")
    plt.tick_params(direction="in")
    plt.xlabel("Winning time")
    plt.ylabel("Distribution")
    plt.savefig(os.path.join(dirname, f"winning_time{i}.png"))

file.close()

# Save means

means = np.zeros(8)
for i in range(8):
    exec(f"means[i] = np.mean(winning_{i+3})")

with open(os.path.join(dirname, '../results/winning_time_mean.txt'), 'w') as file:
    for element in means:
        file.write(str(element) + "\n")
