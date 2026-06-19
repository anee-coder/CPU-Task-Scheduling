# ML-Augmented CPU Task Scheduling

Cloud task scheduling optimization using Multiple Linear Regression to predict execution times with ~90% accuracy.

## 📦 Repository Contents

| File | Description |
|------|-------------|
| `rm_code_final(cloud_task).py` | ML model for cloud task execution time prediction using Multiple Linear Regression (19 features) |
| `sjf_code.py` | Shortest Job First (SJF) scheduling algorithm with performance comparison (original vs predicted times) |
| `rm_code.py` | Legacy/placeholder file |
| `cloud_task_scheduling_dataset.csv` | Input dataset - Cloud task resource requirements and actual execution times |
| `predicted_execution_times(cloud_task).csv` | Output file - ML predictions with actual vs predicted execution times |

## 🚀 Getting Started

### Requirements
```bash
pip install pandas numpy scikit-learn
```

### Run ML Model (Training & Prediction)
```bash
python rm_code_final(cloud_task).py
```

**Output**:
- Model training details, coefficients, and cross-validation scores printed to console
- `predicted_execution_times_cloud.csv` - Predictions for all cloud tasks

### Run Scheduler Comparison
```bash
python sjf_code.py
```

**Output**:
- Comparison table of SJF scheduling metrics using original vs predicted execution times
- Metrics: Average Waiting Time, Average Turnaround Time, Deviation %

## 📊 Input Dataset: cloud_task_scheduling_dataset.csv

```
Columns: Task_ID, CPU_Usage (%), RAM_Usage (MB), Disk_IO (MB/s), 
         Network_IO (MB/s), Priority, Execution_Time (s)
```

## 🤖 ML Model Details

### Feature Engineering (19 Total Features)

**Original Features (5)**:
- CPU_Usage, RAM_Usage, Disk_IO, Network_IO, Priority

**Engineered Features (14)**:
- Ratios: RAM_per_CPU, IO_per_CPU
- Sums: IO_total (Disk_IO + Network_IO)
- Products: CPU_x_RAM, Priority_x_CPU, Priority_x_RAM, Disk_x_Net
- Powers: CPU_sq, RAM_sq, sqrt_RAM
- Logarithms: log_CPU, log_RAM, log_Disk
- Inverses: inv_CPU

### Algorithm & Performance
- **Type**: Multiple Linear Regression (scikit-learn)
- **Train/Test Split**: 80/20
- **Validation**: 5-Fold Cross-Validation
- **Accuracy**: ~90% R² score
- **Realism Factor**: Gaussian noise (27% std) simulates real-world variability:
  - Network jitter and packet delays
  - OS scheduling overhead
  - Memory pressure and I/O contention

## 📈 Output: predicted_execution_times(cloud_task).csv

```
Columns: Task_ID, CPU_Usage (%), RAM_Usage (MB), Disk_IO (MB/s),
         Network_IO (MB/s), Priority, Actual_Time_ms, Predicted_Time_ms
```

## 🔍 Shortest Job First (SJF) Scheduler

Compares scheduling efficiency using original vs ML-predicted execution times:
- Calculates average waiting time and turnaround time
- Shows percentage deviation between predictions and actual values
- Demonstrates impact of better execution time estimates on scheduling

## 📝 Implementation Notes

- Random seed: 42 (reproducibility)
- Execution times clipped to minimum 0.1s
- Cross-platform compatible (CRLF handling)
- Priority incorporated in all predictions and scheduling

## 🎯 Use Case

Optimize cloud task scheduling by:
1. Training ML model on historical cloud task data
2. Predicting execution times for incoming tasks
3. Using predictions with SJF scheduler for improved efficiency
4. Achieving more accurate scheduling than baseline algorithms

---

**Author**: Kalyan-Madhu  
**Repository**: [ML-Augmented CPU Task Scheduling](https://github.com/Kalyan-Madhu/ML-Augmented-CPU-Task-Scheduling-)  
**Last Updated**: June 2026