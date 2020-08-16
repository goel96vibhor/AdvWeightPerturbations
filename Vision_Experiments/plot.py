import csv,time, os
import pandas as pd
import matplotlib.pyplot as plt

dir = "GradientStatsPercentile_Abs_Norm/"
create_grad_norm = False # create L1, L2 norm plots
over_all_percentile_plot = False   # create percentile plots 
per_param_percentile_plot = True
if create_grad_norm:
    ## PLOT 1: plot grad norm L1, L2 epoch wise
    grad_norm_stats = dir + "GradNormStats.log" 
    grad_norm = pd.read_csv(grad_norm_stats, header=None)
    grad_norm.columns = ['epoch', 'iteration', 'param_name', 'L2', 'L1']
    grad_norm['L2'] = grad_norm['L2']**0.5
    # add row index
    grad_norm['step'] = grad_norm['epoch']*390 + grad_norm['iteration']
    params = sorted(grad_norm['param_name'].unique())

    for param in params:
        print(param)
        window = 20
        p_data = grad_norm[grad_norm["param_name"]==param]
        p_data = p_data.sort_values(by=['step'])
        p_data['MovingAvg_Window_' + str(window)] = p_data['L1'].rolling(window=window).mean()
        plot = p_data.plot.line(x='step', y='MovingAvg_Window_' + str(window), title="ParameterName: " + param + ", Norm: L1")
        # plt.show()
        plt.savefig(dir+"Plots/"+ param + "L1_Norm_" + "Window_" + str(window) + ".png")
        plt.clf()
        plt.close()
        # time.sleep(10)
        p_data['MovingAvg_Window_' + str(window)] = p_data['L2'].rolling(window=window).mean()
        plot = p_data.plot.line(x='step', y='MovingAvg_Window_' + str(window), title="ParameterName: " + param + ", Norm: L2")
        plt.savefig(dir+"Plots/"+ param + "L2_Norm_" + "Window_" + str(window) + ".png")
        plt.clf()
        plt.close()

if over_all_percentile_plot:
    ## PLOT 2: get percentile plots 
    over_all_stats =  dir + "OverallStatsAbs.log"
    over_all_stats = pd.read_csv(over_all_stats, header=None)
    over_all_stats['step'] = over_all_stats[0]*390 + over_all_stats[1]
    print(over_all_stats)
    over_all_stats.plot.line(x='step', y=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    plt.savefig(dir+"Plots/Percentile_OverAllAbs.png")
    plt.clf()
    plt.close()


if per_param_percentile_plot:
    stats =  dir + "PerParamStatsAbs.log"
    stats = pd.read_csv(stats, header=None)
    stats['step'] = stats[0]*390 + stats[1]
    print(stats)
    params = sorted(stats[2].unique())
    params = sorted(stats[2].unique())
    for param in params:
        print(param)
        # window = 20
        data = stats[stats[2]==param]
        data = data.sort_values(by=['step'])

        data.plot.line(x='step', y=[3,4,5,6,7,8,9,10,11])
        plt.legend(["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"])

        plt.savefig(dir+"Plots/Percentile_" + param + "_.png")
        plt.clf()
        plt.close()


