.libPaths( "/home/ssoltani/R/x86_64-pc-linux-gnu-library/4.2/")
library(reticulate)
reticulate::use_condaenv("tfr", "/opt/miniconda3/bin/conda",required = TRUE)
require(raster)
require(keras)
library(tensorflow)
require(rgdal)
require(rgeos)

#find and select the GPUs
gpus = tf$config$experimental$list_physical_devices('GPU')
tf$config$experimental$set_memory_growth(gpus[[1]], TRUE)

# presettings
res = 512L
chnl = 3L

# helper function
#range0_255 <- function(x){255*(x-min(x))/(max(x)-min(x))}  # change range to 0-255 (for orthophoto)

workdir = "/scratch1/ssoltani/workshop/09_CNN_tree_species/1_Citzen_to_Unet_project/"
setwd(workdir)
#load the customized loss function for Unet
source("scripts/utils/Customized_loss_function_Unet.R") 
#output dir
#output dir
outputfolder <- paste0(workdir, "outdir/2_output_12_class_conservative_prediction/")



pred_dir_name <- "ResamplMask_distover0.2_Under15m_11classGOOGLEimg_2Dense_256_512_No_Sieve_512_Nepoch_Combi_data_bothOrtho150_softmax_ReLu/"
#outdir = "/1_results_unet_test/"
checkpoint_dir <-  paste0("chekpoints/2_output_12_class_conservative_prediction/", pred_dir_name)

# List all files in the directory that match the pattern
model_files <- list.files(checkpoint_dir, pattern = "weights.*hdf5")
# Extract loss values from the filenames
loss_values <- sapply(model_files, function(file) {
  as.numeric(unlist(strsplit(unlist(strsplit(file, "-"))[2], ".hdf5"))[1])
})

# Get the filename of the model with the lowest loss
best_model_file <- model_files[which.min(loss_values)]



#create an output folder
pred_folder = paste0(outputfolder,"Pred2ndOrtho_iNat_Second_ortho_Unet_thresh_0.3_",pred_dir_name)
dir.create(pred_folder)

#load the model
model = load_model_hdf5(paste0(checkpoint_dir, best_model_file), compile = FALSE)
summary(model)

#check which model is loaded
print(best_model_file)

model %>% compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.0001),
  loss = bce_dice_loss,
  metrics = custom_metric("dice_coef", dice_coef)
)

# Prediction  ----------------------------------------------------------------

# load ortho+Boundary
#ortho_aoi_divided <- readOGR("/net/home/ssoltani/00 Workshop/04 myDive_tree_spec/02 High_res_uav_data_mydiv/aoi_divided_40_v2_wgs84.shp")
ortho1 <- stack("/net/home/ssoltani/00 Workshop/04 myDive_tree_spec/02_Orthoimage_July_2022/02_Orthoimage_July_2022_Mosaic.tif") 

###crop the big orth into small chunks for easy prediction
#col and row division
no_col <- 2
no_row <- 20
res_row <- floor(dim(ortho1)[1]/no_row)
res_col <- floor(dim(ortho1)[2]/no_col)

##expand the division over the ortho
#col indexes
ind_col_ortho = cbind(seq(1,floor((dim(ortho1)[2])/res_col)*res_col, res_col))
#row indexes
ind_row_ortho = cbind(seq(1,floor((dim(ortho1)[1])/res_row)*res_row, res_row))
# combined indexes
ind_grid_ortho = expand.grid(ind_col_ortho, ind_row_ortho)
dim(ind_grid_ortho)

#apply shift
# Read command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if a value  was provided
if (length(args) == 0) {
  shift_value <- 0
}else{
  shift_value <- as.integer(args[1])
}
message("Adding a prediction shif of:", paste(shift_value, collapse = ", "))

#conservative prediciton thresh
pred_threshold <- 0.3
#total= 40
for(ii in 36:40){
  
  #ortho crop
  ortho = crop(ortho1, 
               extent(ortho1, 
                      #row index
                      ind_grid_ortho[ii,2], ind_grid_ortho[ii,2]+res_row+1000L,# 1k overlap between tiles, 
                      #col index
                      ind_grid_ortho[ii,1], ind_grid_ortho[ii,1]+res_col+1000L))
  
  #col indexes
  ind_col = cbind(seq(1,floor(dim(ortho)[2]/res)*res,floor(res)))+
    shift_value #add a shift to the steps
  length(ind_col)
  #row indexes
  ind_row = cbind(seq(1,floor(dim(ortho)[1]/res)*res,floor(res)))+
    shift_value #add a shift to the steps
  length(ind_row)
  # combined indexes
  ind_grid = expand.grid(ind_col, ind_row)
  dim(ind_grid)
  
  #Set an empty raster for pred based on ortho
  predictions = ortho[[1]]
  predictions = setValues(predictions, NA)
  
  #Moving window pred
  ttt = proc.time()
  for(i in 1:nrow(ind_grid)){
    ortho_crop = crop(ortho, extent(ortho, ind_grid[i,2], ind_grid[i,2]+res-1, ind_grid[i,1], ind_grid[i,1]+res-1))
    ortho_crop = tf$convert_to_tensor(as.array(ortho_crop)/255) %>%
      tf$keras$preprocessing$image$smart_resize(size=c(res, res)) %>%
      tf$reshape(shape = c(1L, res, res, chnl))
    #plotRGB(ortho_crop1)
    #plot(as.raster(predict(model, ortho_crop)[1,,,]))
    if(length(which(is.na(ortho_crop)==TRUE))==0){
      #conservative prediction
      predicted_values <- predict(model, ortho_crop)[1,,,]
      predicted_values <- ifelse(predicted_values < pred_threshold, NA, predicted_values)
      predictions[ind_grid[i,2]:(ind_grid[i,2]+res-1), ind_grid[i,1]:(ind_grid[i,1]+res-1)] = t(as.array(k_argmax(predicted_values)))#,as.vector((predict(model, ortho_crop)))
      #predictions = setValues(predictions, as.vector(predict(model, ortho_crop)[1,,,]), index = cellFromRowColCombine(ortho, rownr=ind_grid[i,2]:(ind_grid[i,2]+res-1), colnr=ind_grid[i,1]:(ind_grid[i,1]+res-1)))
    }
    if( i %% 10 == 0){
      print(paste0(i, " of ", nrow(ind_grid), " tiles..."))
    }
    
  }
  time_taken <- proc.time()-ttt
  # Save the processing time to a text file
  write.table(time_taken, file = paste0(pred_folder,"processing_time.txt"), row.names = FALSE, col.names = TRUE, quote = FALSE)
  #Write the predction into a dir
  writeRaster(predictions, filename=paste0(pred_folder,ii, "_Unet_pred_wholeortho","shift_",shift_value,".tif"), overwrite = T)
}


# # just for testing outputs
# par(mfrow = c(1,3))
# plot(as.raster(as.array(ortho)[,,1:3], max=255))
# plot(predictions)
# 
# 
# ###############checking the predictions
# ortho_crop1 = crop(ortho, extent(ortho, ind_grid[i,2], ind_grid[i,2]+res-1, ind_grid[i,1], ind_grid[i,1]+res-1))
# ortho_crop = crop(ortho, extent(ortho, ind_grid[i,2], ind_grid[i,2]+res-1, ind_grid[i,1], ind_grid[i,1]+res-1))
# ortho_crop = array_reshape(as.array(ortho_crop/255), dim = c(1, res, res, chnl))
# 
# 
# par(mfrow=c(1,2))
# plotRGB(ortho_crop1)
# plot(as.raster(predict(model, ortho_crop)[1,,,]))
# plot(as.raster(k_argmax(predict(model, ortho_crop)[1,,,])))
