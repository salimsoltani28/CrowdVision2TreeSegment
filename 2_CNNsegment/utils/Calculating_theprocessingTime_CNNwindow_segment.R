.libPaths( "/home/ssoltani/R/x86_64-pc-linux-gnu-library/4.2/")
library(reticulate)
reticulate::use_condaenv("tfr", "/opt/miniconda3/bin/conda",required = TRUE)
require(raster)
require(keras)
library(tensorflow)
require(rgdal)
require(rgeos)
require(raster)
require(keras)
require(rgdal)
require(rgeos)
require(stringr)
library(tensorflow)
library(countcolors)
library(rgdal)
library(gtools)
#library(doParallel)
library(tfdatasets)
library(tidyverse)

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


pred_dir_name <- "Mask_Output_13_googleimagepred_conservativePred_0.6_No_sieve512_Noepochs_100_softmax/"
#outdir = "/1_results_unet_test/"
checkpoint_dir <-  paste0("checkpoints/2_output_12_class_conservative_prediction/", pred_dir_name)
models = list.files(checkpoint_dir)
models_best = which.min(as.numeric((substr(models, 12,15))))

#create an output folder
pred_folder = paste0(outputfolder,"CNNwindow_CNNsegment_prediction_time",pred_dir_name)
dir.create(pred_folder)

#load the model
model = load_model_hdf5(paste0(checkpoint_dir, models[models_best]), compile = FALSE)
summary(model)

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



#conservative prediciton thresh
pred_threshold <- 0.3
#total= 40
#code for prediction time measurement
ii <- 10
  #ortho crop
  ortho = crop(ortho1, 
               extent(ortho1, 
                      #row index
                      ind_grid_ortho[ii,2], ind_grid_ortho[ii,2]+res_row+1000L,# 1k overlap between tiles, 
                      #col index
                      ind_grid_ortho[ii,1], ind_grid_ortho[ii,1]+res_col+1000L))
  
  #col indexes
  ind_col = cbind(seq(1,floor(dim(ortho)[2]/res)*res,floor(res)))
  length(ind_col)
  #row indexes
  ind_row = cbind(seq(1,floor(dim(ortho)[1]/res)*res,floor(res)))
     #add a shift to the steps
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

  
  # Convert to minutes and hours
  minutes <- time_taken['elapsed'] / 60
  hours <- time_taken['elapsed'] / 3600
  
  # Format with labels
  formatted_time <- c(
    paste0(time_taken[1], " seconds (User)"),
    paste0(time_taken[2], " seconds (System)"),
    paste0(time_taken[3], " seconds (Elapsed)"),
    paste0(round(minutes, 3), " minutes"),
    paste0(round(hours, 3), " hours")
  )
  
  
  # Export to a text file
  write.table(formatted_time, file = paste0(pred_folder, "CNN_segment_on10ortho_processing_time.txt"), row.names = FALSE, col.names = FALSE, quote = FALSE)
  
####################################################################################Processing time measurement for the second orthoimage
  #setting the library path to the lib

  #studyarea <- mixedsort(list.files(allimg_shap,pattern = "smallcrop.shp",recursive = TRUE))
  res = 512L #window size (we can make this smaller and use tf resize function in the prediction loop to resize the data according model value)
  no_bands = 3L
  classes <- 11L
  

  # helper functions
  #range0_255 <- function(x){255*(x-min(x))/(max(x)-min(x))} 
  
  load_best_model = function(path){
    loss = as.numeric(gsub("-","", str_sub(paste0(path,"/", list.files(path)), -10, -6)))
    best = which(loss == min(loss))
    print(paste0("Loaded model of epoch ", best, "."))
    load_model_hdf5(paste0(path,"/", list.files(path)[best]), compile=FALSE)
  }
  
  # load model
  #can we specify the input shape based on our moving window ?
  model = load_best_model("/scratch1/ssoltani/workshop/09_CNN_tree_species/2_CNN_Citizen_photos/checkpoints/00_best_result_with_code/13_googleimagepred_COMPLETE_ORTHO_DONE/Output_effnet7_stamfiltering_distover0.2_Under15m_img512_11classGOOGLEimg_2Dense_256_512/")
  
  ##########################################################################################paralell computing 
  
  # 
  # cores <- detectCores()-8
  # 
  # cl <- makePSOCKcluster(cores)
  # setDefaultCluster(cl=cl)
  # registerDoParallel(cl)
  ##########################################
  #select the moving window steps
  factor1 <- 10L
  
  
  #steps for moving window
  #steps <- 128
  

    ortho <- ortho[[-4]]
    
    #plotRGB(ortho)
    #plot(shape, add=TRUE)
    ############################set the moving window steps
    ind_col = cbind(seq(1,floor(dim(ortho)[2]/res)*res,round(res/factor1))) #
    length(ind_col)
    #row indexes
    ind_row = cbind(seq(1,floor(dim(ortho)[1]/res)*res,round(res/factor1)))#
    length(ind_row)
    # combined indexes
    ind_grid = expand.grid(ind_col, ind_row)
    dim(ind_grid)
    
    
    #############################################################
    #create a matrix to stor the prediction
    preds_matrix <- matrix(NA, nrow = nrow(ind_grid), ncol = classes)
    
    ####################################################################################put the main loop
    #Moving window pred
    ttt2 = proc.time()
    
    for(i in 1:nrow(ind_grid)){
      
      
      ortho_crop = crop(ortho, extent(ortho, ind_grid[i,2], ind_grid[i,2]+res-1, ind_grid[i,1], ind_grid[i,1]+res-1))
      #plotRGB(ortho_crop)
      
      
      
      #save the xy steps
      preds_matrix[i,c(1,2)] = c((extent(ortho_crop)[2] + extent(ortho_crop)[1])/2, (extent(ortho_crop)[4] + extent(ortho_crop)[3])/2)
      
      #change the crop to tfdata
      tensor_pic = tf$convert_to_tensor(as.array(ortho_crop)/255) %>%
        
        tf$keras$preprocessing$image$smart_resize(size=c(res, res)) %>%
        #tf$image$convert_image_dtype(dtype = tf$float32) %>%
        #tf$image$resize_with_crop_or_pad(target_height = res, target_width = res) %>% #newly added function for diminsion normalization
        tf$reshape(shape = c(1L, res, res, no_bands))
      
      #for value check
      ortho_crop = as.array(ortho_crop)
      if(length(which(is.na(ortho_crop)==TRUE))==0){
        
        #preds_matrix[i,c(3:5)]= as.vector(predict(model, tensor_pic))
        preds_matrix[i,3]= as.array(k_argmax(predict(model, tensor_pic)))
        
      }
      
    }
    
    time_taken <- proc.time()-ttt2
    # Save the processing time to a text file
    
    
    # Convert to minutes and hours
    minutes <- time_taken['elapsed'] / 60
    hours <- time_taken['elapsed'] / 3600
    
    # Format with labels
    formatted_time2 <- c(
      paste0(time_taken[1], " seconds (User)"),
      paste0(time_taken[2], " seconds (System)"),
      paste0(time_taken[3], " seconds (Elapsed)"),
      paste0(round(minutes, 3), " minutes"),
      paste0(round(hours, 3), " hours")
    )
    
    
    # Export to a text file
    write.table(formatted_time2, file = paste0(pred_folder, "CNN_window_on10ortho_processing_time.txt"), row.names = FALSE, col.names = FALSE, quote = FALSE)
    