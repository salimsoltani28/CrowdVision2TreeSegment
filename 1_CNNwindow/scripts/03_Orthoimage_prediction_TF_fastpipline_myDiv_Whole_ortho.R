# #setting the library path to the lib
# library(reticulate)
# use_python("/net/home/ssoltani/.conda/envs/rtf3/bin/python")
# reticulate::use_condaenv("rtf3", "/opt/miniconda3/bin/conda",required = TRUE)
# use_virtualenv("/net/home/ssoltani/.conda/envs/rtf3")
.libPaths("/home/ssoltani/R/x86_64-pc-linux-gnu-library/4.2")
#library(magick)

library(reticulate)
reticulate::use_condaenv("tfr", required = TRUE)
#use_virtualenv("/home/ssoltani/.conda/envs/tf29")
#-----------------------------------------------------------

#load the libraries 
require(keras)
library(tensorflow)
library(tfdatasets)
library(tidyverse)
library(tibble)
library(rsample)
library(countcolors)
library(reticulate)
library(gtools)
library(raster)
#library(sampler)
library(rgdal)
library(foreach)
library(doParallel)
library(matrixStats)
#set a seed
tf$compat$v1$set_random_seed(as.integer(28))
set.seed(28)
#tfe_enable_eager_execution(device_policy = "silent") 
# enables eager execution (run directly after library(tensorflow))


# set memory growth policy
gpu1 <- tf$config$experimental$get_visible_devices('GPU')[[1]]
#gpu2 <- tf$config$experimental$get_visible_devices('GPU')[[2]]
tf$config$experimental$set_memory_growth(device = gpu1, enable = TRUE)

#tf$config$experimental$set_memory_growth(device = gpu1, enable = TRUE) #strategy <- tf$distribute$MirroredStrategy() # required for using multiple GPUs, uncomment both lines in case just one GPU is used strategy$num_replicas_in_sync
#set the work and out dir
#list the model folders
paths_toModel <- "/scratch1/ssoltani/workshop/09_CNN_tree_species/2_CNN_Citizen_photos/checkpoints/From_Server2/"
outdir = "/scratch1/ssoltani/workshop/09_CNN_tree_species/2_CNN_Citizen_photos/outdir/3_full_ortho_july_pred_with_thresh_Filter_0.6/EffNetV2L_filtstam_dist0.2_15m_512_11class_rmspropLr0.0001_L2_8kdata_iNat_PlantNetcomplete_June_1_150Ep_V2/"
dir.create(outdir)
#path to cropped ortho
workdir = "/scratch1/ssoltani/workshop/09_CNN_tree_species/2_CNN_Citizen_photos/dataset/02_myDiv_cropped_orthoimages/02_Orthoimage_July_2022_wholeortho_crop/"
setwd(workdir)


#####load orth images and shapefile for rasterizing 

ortho1 <- stack("/net/home/ssoltani/00 Workshop/04 myDive_tree_spec/02_Orthoimage_July_2022/02_Orthoimage_July_2022_Mosaic.tif")
#AOI_poly <- readOGR("/net/home/ssoltani/00 Workshop/04 myDive_tree_spec/02_Orthoimage_July_2022/MyDiv_ortho_boundary.shp")
#ref_poly <- readOGR("/net/home/ssoltani/00 Workshop/04 myDive_tree_spec/02_Orthoimage_July_2022/Ref_transects/Transect_ref_modified_exported.shp")

#get the list of dirs
ortho_path <- workdir
listof_folder <- mixedsort(list.dirs(ortho_path))[-1]
#get list of xy files
listof_imgxy <- mixedsort(list.files(ortho_path, pattern = ".csv",full.names = TRUE,recursive = TRUE))


# image size for distance filtering
res = 512L
n_bands = 3L

#create the oupt directory 
#dir.create(paste0(outdir), recursive = TRUE)


#####################################################################################################################



# tfdatasets input pipeline -----------------------------------------------
create_dataset <- function(data,
                           batch, # numeric. multiplied by number of available gpus since batches will be split between gpus
                           dataset_size){ # numeric. number of samples per epoch the model will be trained on
  
  
  # data1 <- data %>% group_by(ref) %>% sample_n(nrow(filter(data,data$ref==0))) %>% ungroup()
  # data2 <- data1[,1]
  # ref <- to_categorical(unlist(as.list(data1[,2])))
  # data <- tibble(data2, ref)
  
  dataset = data %>%
    tensor_slices_dataset()
  
  
  dataset = dataset %>%
    dataset_map(~.x %>% purrr::list_modify( # read files and decode png
      #img = tf$image$decode_png(tf$io$read_file(.x$img), channels = no_bands)
      img = tf$image$decode_png(tf$io$read_file(.x$img)
                                , channels = n_bands
      ) %>%
        tf$cast(dtype = tf$float32) %>%  
        tf$math$divide(255) %>% 
        #tf$image$convert_image_dtype(dtype = tf$float32) %>%
        tf$keras$preprocessing$image$smart_resize(size=c(res, res))))
  
  
  
  
  dataset = dataset %>%
    dataset_batch(batch, drop_remainder = TRUE) %>%
    dataset_map(unname) %>%
    #dataset_prefetch(buffer_size = tf$data$experimental$AUTOTUNE)
    dataset_prefetch_to_device(device = "/gpu:0", buffer_size =tf$data$experimental$AUTOTUNE)
}



#############################################################################prepare data
#loop over 

model_dirs <- list.dirs(paths_toModel,recursive = F)

#function to load the best model 
load_best_model = function(path){
  loss = as.numeric(gsub("-","", str_sub(paste0(path,"/", list.files(path)), -10, -6)))
  best = tail(which(loss == min(loss)),n=1)
  print(paste0("Loaded model of epoch ", best, "."))
  load_model_hdf5(paste0(path,"/", list.files(path)[best]), compile=FALSE)
}



#moving window factoer
factor1 <- 10L

#number of image per batch (choose according number of tiles)
batch_no <- 300


#specify this to set all prediciton under this value to NA
pred_threshold <- c(0.6)
#for loop prediction

for(m in 1:length(model_dirs)){
  #load the best model from selected folder
  models_path <- model_dirs[m]
  model = load_best_model(models_path)
  #create a folder 
  pred_path <- paste0(outdir,list.dirs(paths_toModel,recursive = F,full.names = FALSE)[m],"/")
  dir.create(pred_path)
  
  #####
  
  for(t in 1:length(listof_folder)){
    #read the png files
    
    #list all images in the corresponding foler
    pathtry <- list.files(listof_folder[t], pattern = ".png",full.names = TRUE)
    #sorth the images according to their number
    all_imgs1 = tibble(img=sort(pathtry))
    
    #read xy files
    imga1_xy <- read.csv(listof_imgxy[t])
    
    #create dataset function application on images
    all_imgs <- create_dataset(data = all_imgs1,batch = batch_no)
    
    #model prediction
    
    pred_prob <- predict(object = model,x=all_imgs)%>% as.tibble() 
    for(p in 1:length(pred_threshold)){
      pred_conditon <- pred_prob%>% rowwise() %>% 
        mutate(final_pred=ifelse(max(across(starts_with("V")))>pred_threshold[p],which.max(across(starts_with("V")))-1,NA)) %>% 
        pull(final_pred)
      
      #####export the data
      
      #combine the prediction and xy
      pred_xy <- cbind(imga1_xy[1:length(pred_conditon),], pred= pred_conditon )
      
      
      ############crop the orthoimage to get bounding box
      #load the shape according to their id(which corresponds to the ortho number)
      #shape <-  ref_poly[ref_poly$id==t,]
      
      #crop the image
      ortho = ortho1
      ortho <- ortho[[-4]]
      
      #create the grid for prediction rasterization
      ind_col = cbind(seq(1,floor(dim(ortho)[2]/res)*res,round(res/factor1))) #
      length(ind_col)
      #row indexes
      ind_row = cbind(seq(1,floor(dim(ortho)[1]/res)*res,round(res/factor1)))#
      length(ind_row)
      # combined indexes
      ind_grid = expand.grid(ind_col, ind_row)
      dim(ind_grid)
      
      ########################################## another way of rasterizing
      #creat extent
      dat = data.frame( x = pred_xy[,2], y = pred_xy[,3],var = pred_xy[,4])
      e <- extent(ortho[[1]])
      #refrence grid for now /this can be replaced by ortho
      ref_grid =raster(e,length(ind_row), length(ind_col), crs="+proj=longlat +datum=WGS84 +no_defs    +ellps=WGS84 +towgs84=0,0,0")
      
      
      ################################### put the coordinate
      coordinates(dat) = ~x + y
      projection(dat)<-CRS("+proj=longlat +datum=WGS84 +no_defs    +ellps=WGS84 +towgs84=0,0,0")
      predicte_raster = rasterize(dat, ref_grid, field = "var", fun = "first")
      #plot(predicte_raster)
      
      
      #########################################more plots
      crs(predicte_raster) = crs (ortho)
      #write the prediction to the file
      #writeRaster(predicte_raster, paste0("cnn_prediction_",gsub(".*01 Orthoimages//","",allimgaes[[t]])))
      writeRaster(predicte_raster, paste0(pred_path,t,"_img",res,"_cnn_prediction_mydiff_spec","pred_thresh_Wholeortho",pred_threshold[p],".tif"))
      # ##############################################t
    }
    
    
  }
  
  
}







