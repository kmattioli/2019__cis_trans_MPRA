#!/bin/bash
#SBATCH --job-name=count_JR2628_S3_bothlanes
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000
#SBATCH --time=3880
#SBATCH --partition=serial_requeue
#SBATCH --output=../data/02__mpra/01__counts/JR2628_S3_bothlanes.stdout
#SBATCH --error=../data/02__mpra/01__counts/JR2628_S3_bothlanes.stderr
#SBATCH --mail-type=END
#SBATCH --mail-user=kaia.mattioli@gmail.com

mkdir -p ../data/02__mpra/01__counts

module load Anaconda
source activate py3

python extract_barcodes_py3.py -f ../data/02__mpra/00__seq_data/01__trimmed/19__latest_mESCs/JR2628_S3_bothlanes_trimmed.fq.gz -i ../data/01__design/02__index/TWIST_pool4_v8_final.txt.gz -c step3 -o ../data/02__mpra/01__counts/JR2628_S3_bothlanes
