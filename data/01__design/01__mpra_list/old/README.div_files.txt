i noticed that several lncRNA/mRNAs that were called as divergent based on Gencode annotation (i.e., had a TSS on opposite strand w/in 1000 bp), did not have evidence for being divergent based on CAGE data. so i further required that genes called as divergent need to intersect a *cage* peak on the opposite strand w/in 1000 bp by running the following:

bedtools closest -S -d -a gencode.vM13.crossmapped_to_mm9.TSS.div_lnc.sorted.bed -b ../../../01__cage_files/mouse/mm9.cage_peak_phase1and2combined_coord.TSS.sorted.bed | awk '{{OFS="\t"} if ($14 < 1000) {print $1, $2, $3, $4, $5, $6}}' > gencode.vM13.crossmapped_to_mm9.div_lnc.TSS.cage_intersected.bed


and these files are now the "updated_divergent_genes.hg19.txt"
