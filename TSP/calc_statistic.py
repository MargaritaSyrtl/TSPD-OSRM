""" statistical tests that allow comparison of two samples """
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from loguru import logger


"""makespan analyse"""
# GA
makespan_ga = [53, 52, 55, 55, 58, 60, 58, 60, 50, 57]  # minutes
exec_time_ga = [105, 131, 90, 110, 162, 72, 120, 120, 130]  # seconds
# mean and standard deviation
mean_ga = np.mean(makespan_ga)
logger.info(f"mean_ga: {mean_ga}")
std_ga = np.std(makespan_ga, ddof=1)
logger.info(f"std_ga: {std_ga}")


# RL
makespan_rl = [48, 45, 54, 55, 47, 47, 50, 59, 58, 60]
exec_time_rl = [108, 112, 108, 108, 109, 117, 116, 116, 112]
# mean and standard deviation
mean_rl = np.mean(makespan_rl)
logger.info(f"mean_rl: {mean_rl}")
std_rl = np.std(makespan_rl, ddof=1)
logger.info(f"std_rl: {std_rl}")


# 95% confidence interval
ci_ga = stats.t.interval(0.95, len(makespan_ga)-1, loc=mean_ga, scale=std_ga/np.sqrt(len(makespan_ga)))
ci_rl = stats.t.interval(0.95, len(makespan_rl)-1, loc=mean_rl, scale=std_rl/np.sqrt(len(makespan_rl)))
logger.info(f"ci_ga: {ci_ga}")
logger.info(f"ci_rl: {ci_rl}")


# Boxplot
logger.debug("Comparison of Makespan Distributions")
plt.boxplot([makespan_ga, makespan_rl], labels=["GA", "RL"])
plt.ylabel("Makespan (minutes)")
# plt.title("Comparison of Makespan Distributions")
plt.show()


# CDF
def plot_cdf(data, label):
    """cumulative distribution function.
    Shows what proportion of sample values are less than or equal to a certain threshold."""
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(data)+1) / float(len(data))
    plt.plot(sorted_data, yvals, label=label)


logger.debug("CDF of Makespan")
plot_cdf(makespan_ga, "GA")
plot_cdf(makespan_rl, "RL")
plt.xlabel("Makespan (minutes)")
plt.ylabel("Cumulative Probability")
# plt.title("CDF of Makespan")
plt.legend()
plt.grid(True)
plt.show()

# t-test (comparison of averages)
t_stat, p_val = stats.ttest_ind(makespan_ga, makespan_rl)
logger.info(f"t_stat: {t_stat}")
logger.info(f"p_val: {p_val}")  # if < 0.05, the difference is statistically significant

# Mann-Whitney U (comparison of distributions)
u_stat, p_val_u = stats.mannwhitneyu(makespan_ga, makespan_rl)
logger.info(f"u_stat: {u_stat}")
logger.info(f"p_val_u: {p_val_u}")  # if < 0.05, the difference is statistically significant


"""execution time analyse"""
# GA
mean_exec_ga = np.mean(exec_time_ga)
logger.info(f"mean_exec_ga: {mean_exec_ga}")
std_exec_ga = np.std(exec_time_ga, ddof=1)
logger.info(f"std_exec_ga: {std_exec_ga}")
ci_exec_ga = stats.t.interval(0.95, len(exec_time_ga)-1, loc=mean_exec_ga, scale=std_exec_ga/np.sqrt(len(exec_time_ga)))
logger.info(f"ci_exec_ga: {ci_exec_ga}")

# RL
mean_exec_rl = np.mean(exec_time_rl)
logger.info(f"mean_exec_rl: {mean_exec_rl}")
std_exec_rl = np.std(exec_time_rl, ddof=1)
logger.info(f"std_exec_rl: {std_exec_rl}")
ci_exec_rl = stats.t.interval(0.95, len(exec_time_rl)-1, loc=mean_exec_rl, scale=std_exec_rl/np.sqrt(len(exec_time_rl)))
logger.info(f"ci_exec_rl: {ci_exec_rl}")


logger.debug("Comparison of Execution Times")
plt.boxplot([exec_time_ga, exec_time_rl], labels=["GA", "RL"])
plt.ylabel("Execution Time (seconds)")
# plt.title("Comparison of Execution Times")
plt.grid(True)
plt.show()


logger.debug("CDF of exec time")
plot_cdf(exec_time_ga, "GA")
plot_cdf(exec_time_rl, "RL")
plt.xlabel("Exec time (seconds)")
plt.ylabel("Cumulative Probability")
# plt.title("CDF of exec time")
plt.legend()
plt.grid(True)
plt.show()

# t-test
t_exec, p_exec = stats.ttest_ind(exec_time_ga, exec_time_rl)
logger.info(f"t_exec: {t_exec}")
logger.info(f"p_exec: {p_exec}")
# Mann-Whitney U
u_exec, p_exec_u = stats.mannwhitneyu(exec_time_ga, exec_time_rl)
logger.info(f"u_exec: {u_exec}")
logger.info(f"p_exec_u: {p_exec_u}")





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

