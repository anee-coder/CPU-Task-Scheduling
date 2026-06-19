import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df_orig  = pd.read_csv(os.path.join(BASE_DIR, "cloud_task_scheduling_dataset.csv"))
df_pred  = pd.read_csv(os.path.join(BASE_DIR, "predicted_execution_times_realistic(cloud_task).csv"))

def sjf_non_preemptive(burst_order, actual_burst):
    order  = sorted(range(len(burst_order)), key=lambda i: burst_order[i])
    clock  = 0.0
    wt_list, tat_list = [], []
    for i in order:
        wt_list.append(clock)
        clock += actual_burst[i]
        tat_list.append(clock)
    return round(np.mean(wt_list), 4), round(np.mean(tat_list), 4)

o_wt, o_tat = sjf_non_preemptive(
    df_orig["Execution_Time (ms)"].values,
    df_orig["Execution_Time (ms)"].values
)

p_wt, p_tat = sjf_non_preemptive(
    df_pred["Predicted_Time_ms"].values,
    df_pred["Actual_Time_ms"].values
)

print("SJF Non-Preemptive - Comparison (all values in ms)\n")
print(f"{'Metric':<25} {'Original':>12} {'Predicted':>12} {'Difference':>12}")
print("-" * 63)
print(f"{'Avg Waiting Time':<25} {o_wt:>12.4f} {p_wt:>12.4f} {p_wt - o_wt:>+12.4f}")
print(f"{'Avg Turnaround Time':<25} {o_tat:>12.4f} {p_tat:>12.4f} {p_tat - o_tat:>+12.4f}")
print("-" * 63)
print(f"\nAvg Waiting Time   deviation : {abs(p_wt - o_wt) / o_wt * 100:.4f}%")
print(f"Avg Turnaround Time deviation : {abs(p_tat - o_tat) / o_tat * 100:.4f}%")
