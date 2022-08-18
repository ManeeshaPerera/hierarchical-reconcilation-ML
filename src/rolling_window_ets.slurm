#!/bin/bash
# The project ID which this job should run under:
#SBATCH --account="punim0512"

# The name of the job:
#SBATCH --job-name="rolling-ml-ets"

# Partition for the job:
#SBATCH --partition gpgpu
#SBATCH --qos gpgpumse

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=06-00:00:0

#Job array
#SBATCH --array=0,1,2,3

# Send yourself an email when the job:
#SBATCH --mail-user=pereramg@student.unimelb.edu.au
#SBATCH --mail-type=FAIL,END

source /usr/local/module/spartan_new.sh
module load fosscuda/2019b
module load gcc/8.3.0 cuda/10.1.243 openmpi/3.1.4
module load python/3.7.4
module load tensorflow/2.3.1-python-3.7.4

# The job command(s):
# case 1 lambda 1 -- last param doesn't add value
python run_ml_method_rolling_origin.py ${SLURM_ARRAY_TASK_ID} ets '' '' 0 ''
# case 1 lambda [0.01, 0.09]
python run_ml_method_rolling_origin.py ${SLURM_ARRAY_TASK_ID} ets True '' 0 ''
# case 1 lambda [0.1, 0.9]
python run_ml_method_rolling_origin.py ${SLURM_ARRAY_TASK_ID} ets True '' 1 ''
# case 1 lambda [1, 4]
python run_ml_method_rolling_origin.py ${SLURM_ARRAY_TASK_ID} ets True '' 2 ''
# case 1 lambda [0.01, 5]
python run_ml_method_rolling_origin.py ${SLURM_ARRAY_TASK_ID} ets True '' 3 ''


# case 2 lambda 1
python run_ml_method_rolling_origin.py ${SLURM_ARRAY_TASK_ID} ets '' True 0 ''
# case 2 lambda [0.01, 0.09]
python run_ml_method_rolling_origin.py ${SLURM_ARRAY_TASK_ID} ets True True 0 ''
# case 2 lambda [0.1, 0.9]
python run_ml_method_rolling_origin.py ${SLURM_ARRAY_TASK_ID} ets True True 0 ''
# case 2 lambda [1, 4]
python run_ml_method_rolling_origin.py ${SLURM_ARRAY_TASK_ID} ets True True 0 ''
# case 2 lambda [0.01, 5]
python run_ml_method_rolling_origin.py ${SLURM_ARRAY_TASK_ID} ets True True 0 ''