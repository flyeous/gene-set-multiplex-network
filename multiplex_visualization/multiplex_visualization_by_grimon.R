### Reference: https://github.com/mkanai/grimon

library(grimon)
library(rgl)

df.bp <- read.csv('R_vis_GO_BP_multiplex.csv', sep = ',', header = TRUE
)

label.bp <- as.factor(c("Jaccard","CD8.T", "B", 
           "Memory.CD4.T",   "CD14.Mono","Naive.CD4.T"))

set.seed(111)
# df.bp[,2:13] -> UMAP coordinates of layers in the multiplex network
# col.structural.property -> the hue of gene sets in hex numbers, 
# which is converted from their numerical values

col.structural.property <-  df.bp$Multiplex.PageRank
grimon(x = df.bp[,2:13], label = label.bp, col = col.structural.property,
       optimize_coordinates = TRUE, maxiter = 1e3,  border_col = NULL, plane_col = NULL,
       score_function = "angle", return_coordinates = TRUE, 
       segment_alpha = 0.3, windowRect = c(0, 0, 3200, 2400), plot_2d_panels = FALSE)


# df.cc.immune <- read.csv('R_vis_GO_CC_ImmuneSig.csv', sep = ',', header = TRUE
# )

# label.cc.immune <- as.factor(c("Jaccard","BMA", "JEJEPI", 
#                                "LLN",   "LNG","SPL"))

# set.seed(111)
# grimon(x = df.cc.immune[,2:13], label = label.cc.immune, col = df.cc.immune$Gene.set.color,
#        optimize_coordinates = TRUE, maxiter = 1e3,  border_col = NULL, plane_col = NULL,
#        score_function = "angle", return_coordinates = TRUE, 
#        segment_alpha = 0.3, windowRect = c(0, 0, 3200, 2400), plot_2d_panels = FALSE)

# rgl.snapshot("lemniscus_CC_ImmuneSig.png")