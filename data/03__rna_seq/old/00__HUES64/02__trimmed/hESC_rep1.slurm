#!/bin/bash
#SBATCH --job-name=trim_hESC_rep1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000
#SBATCH --time=3880
#SBATCH --partition=serial_requeue
#SBATCH --output=../00__HUES64/02__trimmed/hESC_rep1.stdout
#SBATCH --error=../00__HUES64/02__trimmed/hESC_rep1.stderr
#SBATCH --mail-type=END
#SBATCH --mail-user=kaia.mattioli@gmail.com

module load cutadapt

mkdir -p ../00__HUES64/02__trimmed
/n/home04/mmele/bin/trim_galore_zip/trim_galore --paired -q 20 -o ../00__HUES64/02__trimmed ../00__HUES64/01__fastq/hESC_rep1_R1.fastq.gz ../00__HUES64/01__fastq/hESC_rep1_R2.fastq.gz
