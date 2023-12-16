# NAGASIL

Code and data for "Harnessing Network Effect for Fake News Mitigation: Selecting Debunkers via Self-imitation Learning"

---

This repo contains the requirement file used for constructing the python environment, the suggested way is to create a new Python 3.6 environment (tested on Python 3.6, all Python 3 should work, but might be incompatible with some required packages) using Conda.

For example:
  ```
  $ conda create -n tf1_13-gpu python=3.6
  $ conda activate tf1_13-gpu
  $ python -m pip install -r requirements.txt
  ```

---

We provide two main scrips to organize the experiments:
```
  $ bash sbatch_run_seir_all_budget_wseed_realdata.sh   # For main experiment results on Facebook data
  $ bash sbatch_run_seir_all_budget_wseed_1250.sh   # For main experiment results on synthetic Twitter data
  $ bash sbatch_run_seir_one_budget_wseed.sh        # For ablation study
```

Hints:
- If you are running the code on a Slurm cluster, just set up your python environment and run provided script to submit experiment jobs to the cluster. From our experience, all the experiments would cost about 20k core hours.
- If you are using other job scheduling systems like PBS, please refer to documents about how to transform Slurm jobs into PBS jobs.
- If you did not use any job scheduling systems, you can replace the "sbatch sbatch[1-3].sh" with "python" to run the experiments.

Note that our implementation did not guarantee deterministic on GPUs, we suggest running the experiments on CPU computing clusters to reproduce the experiment results. 

Provided code is distributed under MIT LICENSE.

