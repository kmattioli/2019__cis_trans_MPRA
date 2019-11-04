to get these files, i ran the following command (note: need to run using bedtools v2.25 on centos6, as there's a groupby bug in the version in centos7):

 awk '{{OFS="\t"} if ($2>=1000000) {print $1, $2-1000000, $3+1000000, $4, $5, $6} else {print $1, 0, $3+1000000, $4, $5, $6}}' hg19_evo.bed | bedtools intersect -wo -a - -b /n/rinn_data2/users/kaia/fantom5/hg19.all_TSS_and_enh.bed | bedtools groupby -g 1,2,3,4 -c 10 -o count > ../../../misc/03__nearby_elems/hg19.num_elems_1Mb.bed


