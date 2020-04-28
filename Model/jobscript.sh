#!/bin/bash
#BATCH -A JOB_AUTHOR
#SBATCH -e OutputFromRunningOnKOKO/run_gan_ErrorFile.%j.txt
#SBATCH -o OutputFromRunningOnKOKO/run_gan_outFile.%j.txt
#SBATCH -p longq7-mri
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4

module load cuda-10.1.243-gcc-8.3.0-ti55azn 
module load openblas-0.3.7-gcc-8.3.0-oqk2bly
module load fftw-3.3.8-gcc-8.3.0-wngh6wh
module load cudnn-7.6.5.32-10.1-linux-x64-gcc-8.3.0-vldxhwt

python3 AdaptiveGraphStructureEmbedding/Model/run_gan.py

