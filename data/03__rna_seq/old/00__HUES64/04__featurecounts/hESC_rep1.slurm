#!/bin/bash
#SBATCH --job-name=featureCount_hESC_rep1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8000
#SBATCH --time=0-2:0
#SBATCH --partition=general
#SBATCH --output=../00__HUES64/04__featurecounts/hESC_rep1.stdout
#SBATCH --error=../00__HUES64/04__featurecounts/hESC_rep1.stderr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kaia.mattioli@gmail.com

 /n/home04/mmele/bin/subread-1.4.6-p3-Linux-x86_64/bin/featureCounts -T 8  -a  /n/rinn_data2/users/kaia/annotation/human/gencode/hg19/gencode.v25lift37.annotation.gtf -o ../00__HUES64/04__featurecounts/hESC_rep1.counts.txt ../00__HUES64/03__alignments/hESC_rep1/accepted_hits.bam
