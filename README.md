# ML-Augmented CPU Task Scheduling

Cloud task scheduling optimization using a Multiple Linear Regression model to predict execution times and improve scheduling decisions.

Repository contents
- `# ML-Augmented CPU Task Scheduling.md` — project notes and overview
- `rm_code_final(cloud_task).py` — ML model training & prediction
- `sjf_code.py` — SJF scheduler and comparison using actual vs predicted times

Requirements
```
pip install pandas numpy scikit-learn
```

Quick start
- Train and predict execution times:
```
python rm_code_final(cloud_task).py
```
- Run scheduler comparison:
```
python sjf_code.py
```

Data files
- Place dataset or Excel files in a `data/` folder. If files are large (>50MB) consider enabling Git LFS and tracking `*.xlsx`.

Notes
- Random seed: 42 for reproducibility
- Execution times clipped to a minimum value

License
- MIT (add a LICENSE file if desired)
