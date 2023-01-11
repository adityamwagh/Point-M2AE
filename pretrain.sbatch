#!/bin/bash
#SBATCH --nodes=1                        # requests 1 compute server
#SBATCH --ntasks-per-node=1              # runs 1 task on each server
#SBATCH --cpus-per-task=8                # uses 2 compute cores per task
#SBATCH --time=1:00:00                  # for one hour
#SBATCH --mem=64GB
#SBATCH --job-name=pretrain_kitti
#SBATCH --output=result.out
#SBATCH --gres=gpu:rtx8000:1   # To request specific GPU (v100 or rtx8000)


singularity exec --nv \
	    --overlay /scratch/amw9425/overlays/upcr.ext3:ro \
	    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python torch-test.py"