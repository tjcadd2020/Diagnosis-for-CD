species = input_table

idx = rowSums(species)>0
species = na.omit(species[idx,])
print(paste0("Detected non-zero species are ", dim(species)[1], "."))



print(paste0("Samples size are:"))
colSums(species)
min = min(colSums(species))


# if (opts$depth==0){
#   opts$depth=min}
# print(paste0("Rarefaction depth is ", min))


set.seed(322)
otu = vegan::rrarefy(t(species), min)
# print(paste0("All sample rarefaction as following"))
# rowSums(otu)



## 2.3 Alpha diversity

library(vegan)
estimateR = t(estimateR(otu))[,c(1,2,4)]
colnames(estimateR) = c("richness", "chao1", "ACE")

shannon = diversity(otu, index = "shannon")
simpson = diversity(otu, index = "simpson")
invsimpson = diversity(otu, index = "invsimpson")

alpha_div = cbind(estimateR, shannon, simpson, invsimpson)
print(paste0("Calculate six alpha diversities by estimateR and diversity"))
head(alpha_div, n=3)



