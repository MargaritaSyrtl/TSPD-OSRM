import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# results = [54, 51, 51, 50, 28, 52, 50, 27, 28, 27]
# results = [37, 34, 40, 41, 38, 39, 40, 37, 44, 43]
# results = [27, 28, 27, 28, 58, 93, 60, 90, 60, 91]
# results = [43, 47, 34, 43, 45, 35, 40, 40, 44, 39]
# results = [70, 68, 70, 27, 26, 27, 27, 28, 65, 69]
# results = [36, 36, 40, 35, 38, 39, 39, 40, 40, 36]
# results = [35, 37, 39, 36, 33, 39, 36, 36, 38, 39]
# results = [115, 118, 75, 77, 77, 76, 79, 78, 77, 74]
# results = [36, 40, 36, 36, 38, 38, 37, 38 , 38, 37]
# results = [85, 83, 83, 83, 77, 79, 80, 79, 84, 83]
# results = [40, 38, 36, 38, 40, 38, 38, 43, 43, 43]
# results = [42, 43, 38, 43, 38, 43, 38, 39, 37, 37]
# results = [70, 68, 70, 27, 26, 27, 27, 28, 65, 69]
# results = [28, 77, 74, 27, 75, 73, 75, 73]
#results = [117, 120, 86, 82, 87, 84, 52, 81, 80, 117]
#mean = np.mean(results)
#std = np.std(results)
#print(f"Mean = {mean}, Std = {std}")


# GA
makespan_ga = [41, 40, 44, 42, 42, 37, 43, 38, 39, 42]  # minutes
exec_time_ga = [55, 152, 102, 100, 48, 149, 51, 106, 105, 114]  # seconds
# mean and standard deviation
mean_ga = np.mean(makespan_ga)
std_ga = np.std(makespan_ga, ddof=1)

# RL
makespan_rl = []
exec_time_rl = []
# mean and standard deviation
mean_rl = np.mean(makespan_rl)
std_rl = np.std(makespan_rl, ddof=1)

# 95% confidence interval
ci_ga = stats.t.interval(0.95, len(makespan_ga)-1, loc=mean_ga, scale=std_ga/np.sqrt(len(makespan_ga)))
ci_rl = stats.t.interval(0.95, len(makespan_rl)-1, loc=mean_rl, scale=std_rl/np.sqrt(len(makespan_rl)))


# Boxplot
plt.boxplot([makespan_ga, makespan_rl], labels=["GA", "RL"])
plt.ylabel("Makespan (minutes)")
plt.title("Comparison of Makespan Distributions")
plt.show()


# CDF
def plot_cdf(data, label):
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(data)+1) / float(len(data))
    plt.plot(sorted_data, yvals, label=label)


plot_cdf(makespan_ga, "GA")
plot_cdf(makespan_rl, "RL")
plt.xlabel("Makespan (minutes)")
plt.ylabel("Cumulative Probability")
plt.title("CDF of Makespan")
plt.legend()
plt.grid(True)
plt.show()

# t-test (comparison of averages)
t_stat, p_val = stats.ttest_ind(makespan_ga, makespan_rl)

# Mann-Whitney U (comparison of distributions)
u_stat, p_val_u = stats.mannwhitneyu(makespan_ga, makespan_rl)

