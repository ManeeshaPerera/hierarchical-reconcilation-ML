#!/bin/bash
# The project ID which this job should run under:
#SBATCH --account="punim0512"

# The name of the job:
#SBATCH --job-name="wiki-arima"

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
#SBATCH --array=0,1,2,3,4,5,6,7,8,9

# Send yourself an email when the job:
#SBATCH --mail-user=pereramg@student.unimelb.edu.au
#SBATCH --mail-type=FAIL,END

source /usr/local/module/spartan_new.sh
module load fosscuda/2019b
module load gcc/8.3.0 cuda/10.1.243 openmpi/3.1.4
module load python/3.7.4
module load tensorflow/2.3.1-python-3.7.4

# The job command(s):
# all are remove skip tests
# case 1 lambda 1 -- last param doesn't add value
python run_ml_reconciliation.py 2 arima '' '' 0 True ${SLURM_ARRAY_TASK_ID}
# case 1 lambda [0.01, 0.09]
python run_ml_reconciliation.py 2 arima True '' 0 True ${SLURM_ARRAY_TASK_ID}
# case 1 lambda [0.1, 0.9]
python run_ml_reconciliation.py 2 arima True '' 1 True ${SLURM_ARRAY_TASK_ID}
# case 1 lambda [1, 4]
python run_ml_reconciliation.py 2 arima True '' 2 True ${SLURM_ARRAY_TASK_ID}
# case 1 lambda [0.01, 5]
python run_ml_reconciliation.py 2 arima True '' 3 True ${SLURM_ARRAY_TASK_ID}


# case 2 lambda 1 -- last param doesn't add value
python run_ml_reconciliation.py 2 arima '' True 0 True ${SLURM_ARRAY_TASK_ID}
# case 2 lambda [0.01, 0.09]
python run_ml_reconciliation.py 2 arima True True 0 True ${SLURM_ARRAY_TASK_ID}
# case 2 lambda [0.1, 0.9]
python run_ml_reconciliation.py 2 arima True True 1 True ${SLURM_ARRAY_TASK_ID}
# case 2 lambda [1, 4]
python run_ml_reconciliation.py 2 arima True True 2 True ${SLURM_ARRAY_TASK_ID}
# case 2 lambda [0.01, 5]
python run_ml_reconciliation.py 2 arima True True 3 True ${SLURM_ARRAY_TASK_ID}