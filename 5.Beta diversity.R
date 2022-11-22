species = input_table

idx = rowSums(species)>0
species = na.omit(species[idx,])
print(paste0("Detected non-zero species are ", dim(species)[1], "."))



print(paste0("Samples size are:"))
colSums(species)
min = min(colSums(species))


otu = vegan::rrarefy(t(species), min)
# print(paste0("All sample rarefaction as following"))
# rowSums(otu)


library(labdsv)
library(ade4)
library(ggplot2)
library(RColorBrewer)
library(vegan)

tab.dist <- vegdist(otu,method='bray')

pcoa <- pco(tab.dist, k=2)
pcoa_eig <- (pcoa$eig)[1:2] / sum(pcoa$eig)


sample_site <- data.frame({pcoa$points})[1:2]
sample_site$names <- rownames(sample_site)
names(sample_site)[1:2] <- c('PCoA1', 'PCoA2')

sample_site = cbind(sample_site,multi_studies_meta[,c('Group','Study')])

adonis_result_group <- adonis2(tab.dist~Group, multi_studies_meta, permutations = 999)
adonis_result_study <- adonis2(tab.dist~Study, multi_studies_meta, permutations = 999)