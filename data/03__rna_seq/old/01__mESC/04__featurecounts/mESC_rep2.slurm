#!/bin/bash
#SBATCH --job-name=featureCount_mESC_rep2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200
#SBATCH --time=0-0:30
#SBATCH --partition=general
#SBATCH --output=../01__mESC/04__featurecounts/mESC_rep2.stdout
#SBATCH --error=../01__mESC/04__featurecounts/mESC_rep2.stderr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kaia.mattioli@gmail.com

 /n/home04/mmele/bin/subread-1.4.6-p3-Linux-x86_64/bin/featureCounts -T 8  -a  /n/rinn_data2/users/kaia/annotation/mouse/gencode/gencode.vM13.annotation.gtf -o ../01__mESC/04__featurecounts/mESC_rep2.counts.txt ../01__mESC/03__alignments/mESC_rep2/accepted_hits.bam
