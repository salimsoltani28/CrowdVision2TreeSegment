#tempdir(path="/scratch1/ssoltani/workshop/00 Code/07 Accuracy Assessment/01_yDiv_tree_species/")
.libPaths("/home/ssoltani/R/x86_64-pc-linux-gnu-library/4.2")
library(reticulate)
require(raster)
require(rgdal)
library(tidyverse)
library(rgeos)
library(ggplot2)
library(gtools)
library(gridExtra)
library(dplyr)
library(ROCR)
library(MLmetrics)
library(sf)
library(doParallel)
library(foreach)
library(terra)



# Read command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if a directory path was provided
if (length(args) == 0) {
  stop("Error: Directory path not provided")
}

# Extract the directory path from the command line arguments
path_pred <- args[1]
options(digits = 7)
message("Reading the predictions from:", paste(path_pred, collapse = ", "))
#read the orthoimage
orthodir <-   "/net/home/ssoltani/00 Workshop/04 myDive_tree_spec/02_Orthoimage_July_2022/"
#plot_type <- list.files(orthodir, recursive = T, pattern = "Plot_boundary.shp",full.names = T) %>% readOGR()
#path_pred <- "/scratch2/ssoltani/workshop/10_CNN_tree_species/1_Citzen_to_Unet_project/outdir/BestPred_googleimage_sieved50px_13Aprilfinal_pred_majorityvote_test/"

#allimgaes <- mixedsort(list.files(orthodir,pattern = "Mosaic.tif",full.names = TRUE))
ortho <- mixedsort(list.files(orthodir,pattern = "Mosaic.tif",full.names = TRUE)) %>% stack()
ref_transect <- mixedsort(list.files(orthodir,pattern = "Transect_ref_modified_exported.shp",recursive = TRUE,full.names = TRUE))



#load predictions
#path_pred <- "/scratch1/ssoltani/workshop/09_CNN_tree_species/1_Citzen_to_Unet_project/outdir/2_output_12_class_conservative_prediction/Pred2ndOrtho_iNat_Second_ortho_Unet_thresh_0.3_Mask_Output_13_googleimagepred_conservativePred_0.6_No_sieve512_Noepochs_100_softmax_ReLu/"
predictions_list <- mixedsort(list.files(path = path_pred, pattern = ".tif",full.names = TRUE)) #%>% lapply(stack)
vrtfile <- paste0(tempfile(), ".vrt")
predictions <- vrt(predictions_list, vrtfile) %>% stack() 

#condition if mono or polyculture
if(length(predictions_list)==20){
  AOI <- mixedsort(list.files(orthodir,pattern = "Plot_boundary.shp",recursive = TRUE,full.names = TRUE)) %>% st_read()#%>%readOGR()
  
  #In case you want calaculate accuracy of only monoculture
  AOI <- AOI %>% filter(Tree_speci=="Monoculture") %>% as("Spatial")
}else{
  #read all plots
  AOI <- mixedsort(list.files(orthodir,pattern = "Plot_boundary.shp",recursive = TRUE,full.names = TRUE)) %>%readOGR()
}


# #detect
cores <- 60#detectCores()-8
cl <- makePSOCKcluster(cores)
registerDoParallel(cl)
clusterEvalQ(cl, .libPaths("/home/ssoltani/R/x86_64-pc-linux-gnu-library/4.2"))
# 
# #clusterEvalQ(cl, .libPaths("Your library path"))
Acc_val <- foreach(g = seq_len(nrow(AOI)), .packages = c("terra","raster", "rgdal","tidyverse","rgeos","MLmetrics", "foreach","doParallel"), .inorder = T) %dopar% { #,.combine = rbind
  #   
  # accuracy_list <- list()
  # for(g in 1:nrow(AOI)){
  
  # Load data and set up spatial transformations
  pred_map <- stack(predictions)
  aoi <- AOI[g,] %>% spTransform(crs("+init=epsg:3398"))
  aoi <- gBuffer(aoi, byid = FALSE, width = 1)  # Inner buffer
  
  # Align CRS and filter reference data
  ref <- readOGR(ref_transect)
  aoi <- spTransform(aoi, crs(ref))
  index <- gContains(aoi, ref, byid = TRUE)
  ref <- ref[c(index),]
  
  #
  
  # Crop and mask prediction map
  pred_map <- crop(pred_map, extent(ref))
  #for resampling
  ortho_crop <- crop(ortho$X02_Orthoimage_July_2022_Mosaic_1,extent(ref) )
  pred_map <- resample(pred_map,ortho_crop,method="ngb" )
  final_pred <- mask(pred_map, ref) + 1  # Align numbering with reference data
  #if it has 12 for NAs
  final_pred[final_pred == 12] <- 11
  # Rasterize reference data
  refRaster <- rasterize(ref, final_pred, field = as.integer(ref@data[,"Class"]), fun = 'first')
  
  
  # Tile-based accuracy validation setup
  factor1 <- 1L
  res <- 512L
  
  # Column and row indexes for moving window
  ind_col <- cbind(seq(1, floor(dim(final_pred)[2] / res) * res, round(res / factor1)))
  ind_row <- cbind(seq(1, floor(dim(final_pred)[1] / res) * res, round(res / factor1)))
  
  # Combined indexes for grid
  ind_grid <- expand.grid(ind_col, ind_row)
  
  ###main foreach loop #, 
  confusion_matrix <- foreach(i = seq_len(nrow(ind_grid)), .packages = c("raster", "rgdal", "keras", "magick","stringr","dplyr"), .inorder = T) %dopar% {
    
    # Crop the tile from both rasters
    extent_tile <- extent(final_pred, ind_grid[i,2], ind_grid[i,2]+res[g]-1, ind_grid[i,1], ind_grid[i,1]+res[g]-1)
    refVector <- crop(refRaster, extent_tile) %>% 
      as.vector()
    predVector <- crop(final_pred, extent_tile) %>% 
      as.vector()
    #
    u <- c(1,2,3,4,5,6,7,8,9,10,11)#sort(base::union(refVector, predVector))
    #calculate the confusion matrix
    conmat  <- caret::confusionMatrix(data = factor(predVector, u), reference = factor(refVector, u)) 
    #conf matrix
    #save the confusion mat
    confusion_mat<- conmat$table
    #return the confusion matrix
    per_class_F1 <- conmat$byClass[,"F1"]
    # Vector of species names
    list_of_spec <- c("Acer.p", "Aesculus.h", "Betula.p", "Carpinus.b", "Fagus.s", "Fraxinus.e",
                      "Prunus.a", "Quercus.p", "Sorbus.a", "Tilia.p", "Grass")
    per_class_F1 <- tibble(F1=unlist(per_class_F1),Species=list_of_spec)
    
    #combine the data list 
    data_list <- list(ConfusionMatrix = confusion_mat, F1Scores = per_class_F1)
    return(data_list)
    
  }
  
  # Extract F1 accuracies
  f1_extracted <- do.call(rbind, lapply(confusion_matrix, function(x) x$F1Scores))
  
  # Extract reduced sum of confusion matrix
  # Sum all confusion matrices together
  summed_conf_matrix_extracted <- Reduce(`+`, lapply(confusion_matrix, function(x) x$ConfusionMatrix))
  #remove NAs and round it to digits
  per_class_F1_extractd <- f1_extracted %>% 
    na.omit() %>% 
    mutate(F1=round(F1,digits = 2))
  
  # Set column names based on the plot type
  plot_type <- AOI@data[,1][g]
  per_class_F1_extractd$Plot_type <- paste0(rep(as.character(plot_type), times = nrow(per_class_F1_extractd)))
  
  #combine the data list 
  accuracy_extracted <- list(ConfusionMatrix = summed_conf_matrix_extracted, F1Scores = per_class_F1_extractd)
  return(accuracy_extracted)
}


stopCluster(cl)
#mean(unlist(Val_metrics$OA))
#sort(unlist(Val_metrics))htop



### process the data
# Extract F1 accuracies
f1_df <- do.call(rbind, lapply(Acc_val, function(x) x$F1Scores))

# Extract reduced sum of confusion matrix
# Sum all confusion matrices together
summed_conf_matrix <- Reduce(`+`, lapply(Acc_val, function(x) x$ConfusionMatrix))
write.csv(summed_conf_matrix, paste0(path_pred,"Total_confusion_matrix.csv"))

###saved
#data_with_saved <- Acc_val
#change for plots
Acc_val <- f1_df



# Compute mean F1 score for each list of F1 scores
No_class_F1_high <- Acc_val %>% group_by(Species) %>% 
  summarize(meanf1=mean(F1)) %>% 
  filter(Species!="Grass" & meanf1>0.5) %>% count()

# Compute mean F1 score for each list of F1 scores
Treeclass_mean <- Acc_val %>% group_by(Species) %>% 
  summarize(meanf1=mean(F1)) %>% 
  filter(Species!="Grass" ) %>% pull(meanf1) %>% mean() %>% 
  round(digits = 2)



write.csv(Acc_val, paste0(path_pred,"meanTreespecF1_",Treeclass_mean,"_F1gr0.5_",No_class_F1_high,"_per_tile_forallplots.csv"))


# Here enter if comparing between the orthos are true


PerclasF1 <- Acc_val 



#add the plot type
# Function to replace patterns with values
replace_pattern <- function(data) {
  # Extract numeric values after the letters
  values <- gsub("^.*[A-Za-z]+([0-9]+)$", "\\1", data)
  
  # Convert the extracted values to numeric
  values <- as.numeric(values)
  
  return(values)
}

#complete data
PerclasF1 <- PerclasF1 %>% 
  mutate(Plot_type=replace_pattern(Plot_type))
# Generate and save the plot for the complete data

# # Replace the values
# PerclasF1$Species <- gsub("\\.", " ", PerclasF1$Species)
# PerclasF1$Species <- gsub("([^\\.])$", "\\1.", PerclasF1$Species)
# PerclasF1$Species <- ifelse(PerclasF1$Species == "Grass.", "Grass", PerclasF1$Species)



# Alternatively, in a more compact form:
PerclasF1$Species <- ifelse(gsub("([^\\.])$", "\\1.", gsub("\\.", " ", PerclasF1$Species)) == "Grass.", "Grass", gsub("([^\\.])$", "\\1.", gsub("\\.", " ", PerclasF1$Species)))





#png(paste0(pathcsv,"Box_plot_perPlot_F1.png"), width = 600, height = 600,res =600)
png(paste0(path_pred,"completeData_box_perTile_F1.png"), width = 1200, height = 1200, res = 300)



# Assume that your raster values are integers from 1 to 11
# (Adjust if your raster has a different range of values)
#color_palette <- setNames(colors, 1:11)

# Define the desired order of species
list_of_spec <- c("Acer p.", "Aesculus h.", "Betula p.", "Carpinus b.", "Fagus s.", "Fraxinus e.",
                  "Prunus a.", "Quercus p.", "Sorbus a.", "Tilia p.", "Grass")

# Define colors for each species
species_colors <- c(
  "Acer p." = "#8dd3c7",
  "Aesculus h." = "#ffffb3",
  "Betula p." = "#bebada",
  "Carpinus b." = "#fb8072",
  "Fagus s." = "#80b1d3",
  "Fraxinus e." = "#fdb462",
  "Prunus a." = "#b3de69",
  "Quercus p." = "#fccde5",
  "Sorbus a." = "#d9d9d9",
  "Tilia p." = "#bc80bd",
  "Grass" = "#ccebc5"
)

ggplot(PerclasF1, aes(x = factor(Species, levels = list_of_spec), y = F1, fill = Species)) +
  geom_boxplot( color = "#1F618D", alpha = 0.7, outlier.shape = NA, size = 0.2) +  # Adjust the size parameter
  scale_fill_manual(values = species_colors) +
  labs(title = "Per-tile F1 scores by tree species", y = "F1", x = FALSE) +
  theme_classic() +
  theme(axis.title = element_text(size = 8, face = "bold"),
        axis.text = element_text(size = 8),
        plot.title = element_text(size = 8, face = "bold"),
        legend.position = "none",
        panel.border = element_rect(color = "black", fill = NA, size = 0.9),
        axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
        axis.title.x = element_blank())




dev.off()

###remove all stored object in the environment
#rm(list=ls())



png(paste0(path_pred,"completeData_Species_composition_F1_by_class.png"), width = 1800, height = 960, res = 300)



#for vline settings
species_order <- with(PerclasF1, unique(Species[order(Plot_type)]))
breaks <- which(diff(as.numeric(factor(species_order, levels = list_of_spec))) != 1) + 0.5

#color keys for catagories
colors <- c("1" = "#002855", "2" = "#3F8FCF", "4" = "#D1E9F7")
# Get the number of unique types
num_types <- length(unique(PerclasF1$Plot_type))
ggplot(PerclasF1, aes(x = factor(Species, levels = list_of_spec), y = F1)) +
  geom_boxplot(aes(fill = factor(Plot_type, levels = sort(unique(Plot_type)))), color = "#1F618D", alpha = 0.7, outlier.shape = NA, size = 0.2) +
  geom_vline(xintercept = breaks, linetype = "dashed", color = "#1F618D", size = 0.1, alpha = 0.5) +
  labs(title = "Per-tile F1 scores by tree species", y = "F1", fill = "Species\nCount") +
  scale_fill_manual(values = colors)+
  theme_classic() +
  theme(
    axis.title = element_text(size = 8, face = "bold"),
    axis.text = element_text(size = 8),
    plot.title = element_text(size = 8, face = "bold"),
    legend.position = "right",
    legend.title = element_text(size = 8, face = "bold"),
    panel.border = element_rect(color = "black", fill = NA, size = 0.9),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)
  )

dev.off()






##########################################################################Comparing between the orthos
#exclude plots that dont overlap
No_overlapplots <- AOI@data %>% 
  filter(Ortho2_inc==0) %>% pull(Plot_No)

PerclasF1_filter <- PerclasF1 %>% 
  filter(!Plot_type %in% No_overlapplots)



# Generate and save the plot for the filtered data


#complete data
PerclasF1_filter <- PerclasF1_filter %>% 
  mutate(Plot_type=replace_pattern(Plot_type))
# Generate and save the plot for the complete data






#png(paste0(pathcsv,"Box_plot_perPlot_F1.png"), width = 600, height = 600,res =600)
png(paste0(path_pred,"Comparision_box_perTile_F1.png"), width = 1200, height = 1200, res = 300)



# Assume that your raster values are integers from 1 to 11
# (Adjust if your raster has a different range of values)
#color_palette <- setNames(colors, 1:11)





ggplot(PerclasF1_filter, aes(x = factor(Species, levels = list_of_spec), y = F1, fill = Species)) +
  geom_boxplot( color = "#1F618D", alpha = 0.7, outlier.shape = NA, size = 0.2) +  # Adjust the size parameter
  scale_fill_manual(values = species_colors) +
  labs(title = "Per-tile F1 scores by tree species", y = "F1", x = FALSE) +
  theme_classic() +
  theme(axis.title = element_text(size = 8, face = "bold"),
        axis.text = element_text(size = 8),
        plot.title = element_text(size = 8, face = "bold"),
        legend.position = "none",
        panel.border = element_rect(color = "black", fill = NA, size = 0.9),
        axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
        axis.title.x = element_blank())




dev.off()

###remove all stored object in the environment
#rm(list=ls())



png(paste0(path_pred,"Comparision_Species_composition_F1_by_class.png"), width = 1800, height = 960, res = 300)



#forplot vline
species_order <- with(PerclasF1_filter, unique(Species[order(Plot_type)]))
breaks <- which(diff(as.numeric(factor(species_order, levels = list_of_spec))) != 1) + 0.5
#catagories color
colors <- c("1" = "#002855", "2" = "#3F8FCF", "4" = "#D1E9F7")
# Get the number of unique types
num_types <- length(unique(PerclasF1_filter$Plot_type))
ggplot(PerclasF1_filter, aes(x = factor(Species, levels = list_of_spec), y = F1)) +
  geom_boxplot(aes(fill = factor(Plot_type, levels = sort(unique(Plot_type)))), color = "#1F618D", alpha = 0.7, outlier.shape = NA, size = 0.2) +
  geom_vline(xintercept = breaks, linetype = "dashed", color = "#1F618D", size = 0.1, alpha = 0.5) +
  labs(title = "Per-tile F1 scores by tree species", y = "F1", fill = "Species\nCount") +
  scale_fill_manual(values = colors)+
  theme_classic() +
  theme(
    axis.title = element_text(size = 8, face = "bold"),
    axis.text = element_text(size = 8),
    plot.title = element_text(size = 8, face = "bold"),
    legend.position = "right",
    legend.title = element_text(size = 8, face = "bold"),
    panel.border = element_rect(color = "black", fill = NA, size = 0.9),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)
  )

dev.off()






#####################



# #add the plot type
# # Function to replace patterns with values
# replace_pattern <- function(data) {
#   # Extract numeric values after the letters
#   values <- gsub("^.*[A-Za-z]+([0-9]+)$", "\\1", data)
#   
#   # Convert the extracted values to numeric
#   values <- as.numeric(values)
#   
#   return(values)
# }
# 
# #complete data
# PerclasF1 <- PerclasF1 %>% 
#   mutate(Plot_type=replace_pattern(Plot_type))
# # Generate and save the plot for the complete data
# 
# 
# 
# 
# 
# 
# #png(paste0(pathcsv,"Box_plot_perPlot_F1.png"), width = 600, height = 600,res =600)
# png(paste0(path_pred,"completeData_box_perTile_F1.png"), width = 1200, height = 1200, res = 300)
# 
# 
# 
# # Assume that your raster values are integers from 1 to 11
# # (Adjust if your raster has a different range of values)
# #color_palette <- setNames(colors, 1:11)
# 
# # Define the desired order of species
# list_of_spec <- c("Acer.p", "Aesculus.h", "Betula.p", "Carpinus.b", "Fagus.s", "Fraxinus.e",
#                   "Prunus.a", "Quercus.p", "Sorbus.a", "Tilia.p", "Grass")
# 
# # Define colors for each species
# species_colors <- c(
#   "Acer.p" = "#8dd3c7",
#   "Aesculus.h" = "#ffffb3",
#   "Betula.p" = "#bebada",
#   "Carpinus.b" = "#fb8072",
#   "Fagus.s" = "#80b1d3",
#   "Fraxinus.e" = "#fdb462",
#   "Prunus.a" = "#b3de69",
#   "Quercus.p" = "#fccde5",
#   "Sorbus.a" = "#d9d9d9",
#   "Tilia.p" = "#bc80bd",
#   "Grass" = "#ccebc5"
# )
# 
# ggplot(PerclasF1, aes(x = factor(Species, levels = list_of_spec), y = F1, fill = Species)) +
#   geom_boxplot( color = "#1F618D", alpha = 0.7, outlier.shape = NA, size = 0.2) +  # Adjust the size parameter
#   scale_fill_manual(values = species_colors) +
#   labs(title = "Per-tile F1 scores by tree species", y = "F1", x = FALSE) +
#   theme_classic() +
#   theme(axis.title = element_text(size = 8, face = "bold"),
#         axis.text = element_text(size = 8),
#         plot.title = element_text(size = 8, face = "bold"),
#         legend.position = "none",
#         panel.border = element_rect(color = "black", fill = NA, size = 0.9),
#         axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
#         axis.title.x = element_blank())
# 
# 
# 
# 
# dev.off()
# 
# ###remove all stored object in the environment
# #rm(list=ls())
# 
# 
# 
# png(paste0(path_pred,"completeData_Species_composition_F1_by_class.png"), width = 1800, height = 960, res = 300)
# 
# 
# 
# 
# 
# # Get the number of unique types
# num_types <- length(unique(PerclasF1$Plot_type))
# ggplot(PerclasF1, aes(x = factor(Species, levels = list_of_spec), y = F1)) +
#   geom_boxplot(aes(fill = factor(Plot_type, levels = sort(unique(Plot_type)))), color = "#1F618D", alpha = 0.7, outlier.shape = NA, size = 0.2) +
#   geom_vline(xintercept = seq(1.5, num_types - 0.5), linetype = "dashed", color = "#1F618D", size = 0.1, alpha = 0.5) +
#   labs(title = "Per-tile F1 scores by tree species", y = "F1", fill = "Species\nCount") +
#   theme_classic() +
#   theme(
#     axis.title = element_text(size = 8, face = "bold"),
#     axis.text = element_text(size = 8),
#     plot.title = element_text(size = 8, face = "bold"),
#     legend.position = "right",
#     legend.title = element_text(size = 8, face = "bold"),
#     panel.border = element_rect(color = "black", fill = NA, size = 0.9),
#     axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)
#   )
# 
# dev.off()
# 
# 
# 
# 
# 
# 
# ##########################################################################Comparing between the orthos
# #exclude plots that dont overlap
# No_overlapplots <- AOI@data %>% 
#   filter(Ortho2_inc==0) %>% pull(Plot_No)
# 
# PerclasF1_filter <- Acc_val %>% 
#   filter(!Plot_type %in% No_overlapplots)
# 
# 
# 
# # Generate and save the plot for the filtered data
# 
# 
# #complete data
# PerclasF1_filter <- PerclasF1_filter %>% 
#   mutate(Plot_type=replace_pattern(Plot_type))
# # Generate and save the plot for the complete data
# 
# 
# 
# 
# 
# 
# #png(paste0(pathcsv,"Box_plot_perPlot_F1.png"), width = 600, height = 600,res =600)
# png(paste0(path_pred,"Comparision_box_perTile_F1.png"), width = 1200, height = 1200, res = 300)
# 
# 
# 
# # Assume that your raster values are integers from 1 to 11
# # (Adjust if your raster has a different range of values)
# #color_palette <- setNames(colors, 1:11)
# 
# # Define the desired order of species
# list_of_spec <- c("Acer.p", "Aesculus.h", "Betula.p", "Carpinus.b", "Fagus.s", "Fraxinus.e",
#                   "Prunus.a", "Quercus.p", "Sorbus.a", "Tilia.p", "Grass")
# 
# # Define colors for each species
# species_colors <- c(
#   "Acer.p" = "#8dd3c7",
#   "Aesculus.h" = "#ffffb3",
#   "Betula.p" = "#bebada",
#   "Carpinus.b" = "#fb8072",
#   "Fagus.s" = "#80b1d3",
#   "Fraxinus.e" = "#fdb462",
#   "Prunus.a" = "#b3de69",
#   "Quercus.p" = "#fccde5",
#   "Sorbus.a" = "#d9d9d9",
#   "Tilia.p" = "#bc80bd",
#   "Grass" = "#ccebc5"
# )
# 
# ggplot(PerclasF1_filter, aes(x = factor(Species, levels = list_of_spec), y = F1, fill = Species)) +
#   geom_boxplot( color = "#1F618D", alpha = 0.7, outlier.shape = NA, size = 0.2) +  # Adjust the size parameter
#   scale_fill_manual(values = species_colors) +
#   labs(title = "Per-tile F1 scores by tree species", y = "F1", x = FALSE) +
#   theme_classic() +
#   theme(axis.title = element_text(size = 8, face = "bold"),
#         axis.text = element_text(size = 8),
#         plot.title = element_text(size = 8, face = "bold"),
#         legend.position = "none",
#         panel.border = element_rect(color = "black", fill = NA, size = 0.9),
#         axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
#         axis.title.x = element_blank())
# 
# 
# 
# 
# dev.off()
# 
# ###remove all stored object in the environment
# #rm(list=ls())
# 
# 
# 
# png(paste0(path_pred,"Comparision_Species_composition_F1_by_class.png"), width = 1800, height = 960, res = 300)
# 
# 
# 
# 
# 
# # Get the number of unique types
# num_types <- length(unique(PerclasF1_filter$Plot_type))
# ggplot(PerclasF1_filter, aes(x = factor(Species, levels = list_of_spec), y = F1)) +
#   geom_boxplot(aes(fill = factor(Plot_type, levels = sort(unique(Plot_type)))), color = "#1F618D", alpha = 0.7, outlier.shape = NA, size = 0.2) +
#   geom_vline(xintercept = seq(1.5, num_types - 0.5), linetype = "dashed", color = "#1F618D", size = 0.1, alpha = 0.5) +
#   labs(title = "Per-tile F1 scores by tree species", y = "F1", fill = "Species\nCount") +
#   theme_classic() +
#   theme(
#     axis.title = element_text(size = 8, face = "bold"),
#     axis.text = element_text(size = 8),
#     plot.title = element_text(size = 8, face = "bold"),
#     legend.position = "right",
#     legend.title = element_text(size = 8, face = "bold"),
#     panel.border = element_rect(color = "black", fill = NA, size = 0.9),
#     axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)
#   )
# 
# dev.off()
# 
