
# snv calling
time run_midas.py species \
	MIDAS_out/`basename ${file[x]}.fastq` \
	-1 03decom/`basename ${file[x]} _paired_1.fastq`_paired_1.fastq \
	-2 03decom/`basename ${file[x]} _paired_1.fastq`_paired_2.fastq -\
	t 28 --remove_temp


time run_midas.py snps \
	MIDAS_out/`basename ${file[x]}.fastq` \
	-1 03decom/`basename ${file[x]} _paired_1.fastq`_paired_1.fastq \
	-2 03decom/`basename ${file[x]} _paired_1.fastq`_paired_2.fastq -\
	t 28 --remove_temp


time run_midas.py genes \
	MIDAS_out/`basename ${file[x]}.fastq` \
	-1 03decom/`basename ${file[x]} _paired_1.fastq`_paired_1.fastq \
	-2 03decom/`basename ${file[x]} _paired_1.fastq`_paired_2.fastq -\
	t 28 --remove_temp



# merge
time merge_midas.py snps merge_snp \
	-i ../MIDAS_out  -t dir \
	--all_sites --all_samples \
	--species_id $species_id