#Note read before executing the code
#1- make sure that prediction is resampled to the ortho resolution 
#2- make sure that number of classes correspond to the actual number
#3- 


require(raster)
require(keras)
require(rgdal)
require(rgeos)
require(stringr)
library(tensorflow)
library(countcolors)
library(raster)
library(rgdal)
library(gtools)
library(doParallel)
library(tfdatasets)
library(tidyverse)
library(foreach)
library(magick)
library(terra)
#library(sf)
#library(gpclib)



#allimg_shap =  "/net/home/ssoltani/00 Workshop/04 myDive_tree_spec/01 Orthoimages/"

#sepcify image crop
res <- 512L#c(256L,512L,1024)


no_bands = 3L




ortho1 <- stack("/net/home/ssoltani/00 Workshop/04 myDive_tree_spec/02_Orthoimage_July_2022/02_Orthoimage_July_2022_Mosaic.tif")#|>projectRaster(crs = crs("+init=epsg:32735"))

ref_poly <- readOGR("/net/home/ssoltani/00 Workshop/04 myDive_tree_spec/02_Orthoimage_July_2022/MyDiv_ortho_boundary.shp")
pred_folder<- "/scratch1/ssoltani/workshop/09_CNN_tree_species/2_CNN_Citizen_photos/outdir/00_best_result_with_code/Pred_0.6filter/13_googleimagepred/Output_effnet7_stamfiltering_distover0.2_Under15m_img512_11classGOOGLEimg_2Dense_256_512/"
# helper functions

label_dir <- "/scratch1/ssoltani/workshop/09_CNN_tree_species/1_Citzen_to_Unet_project/dataset/July_ortho_training_Data/"

factor1 <- 1L
cores <- 80#detectCores()-20

cl <- makePSOCKcluster(cores)
registerDoParallel(cl)
# #loop over resolution
for(g in 1:length(res)){
  img_folder <- paste0(label_dir,"img_finalcheck_",res[g],"/")
  mask_folder <- paste0(label_dir,"Mask_Output_effnet7_stamfiltering_distover0.2_Under15m_img512_11classGOOGLEimg_2Dense_256_512_No_sieve",res[g],"/")
  lapply(c(img_folder, mask_folder), dir.create)

  #list all pred files
  allpred<- list.files(pred_folder,full.names = T,recursive = T)[1]
 
  #load the ortho
  for (t in 1:1){ #4,5 is donelength(allimgaes)
    #load the image and shp
    #ortho <- stack(allimgaes[[t]])
    shape <-  ref_poly
    
    # load reference data
    #shape = gBuffer(shape,  width=-1)
    #shape = gUnaryUnion(shape)
    #shape = spTransform(shape, crs(ortho))
    pred_raster<- stack(allpred[t]) %>% mask(shape) 
    
    
    # load reference data
    #shape = gBuffer(shape, byid=TRUE, width=0)
    #shape = gUnaryUnion(shape)
    #shape = spTransform(shape, crs(ortho))
    
    
    ortho = mask(ortho1, shape)
    ortho <- ortho[[-4]]
    pred_raster <- resample(pred_raster, ortho, method="ngb")
    #plotRGB(ortho)
    #plot(shape, add=TRUE)
    
    
    ############################set the moving window steps
    ind_col = cbind(seq(1,floor(dim(ortho)[2]/res[g])*res[g],round(res[g]/factor1))) #
    length(ind_col)
    #row indexes
    ind_row = cbind(seq(1,floor(dim(ortho)[1]/res[g])*res[g],round(res[g]/factor1)))#
    length(ind_row)
    # combined indexes
    ind_grid = expand.grid(ind_col, ind_row)
    dim(ind_grid)
    
    
    ###main foreach loop #, 
    orthocrop_list <- foreach(i = seq_len(nrow(ind_grid)), .packages = c("raster", "rgdal", "keras", "magick","stringr"),.combine = rbind, .inorder = T) %dopar% {
      # #for(i in 1:nrow(ind_grid)){
      # #crop image
      # ortho_crop = crop(ortho, extent(ortho, ind_grid[i,2], ind_grid[i,2]+res[g]-1, ind_grid[i,1], ind_grid[i,1]+res[g]-1))
      # #plotRGB(ortho_crop)
      # 
      # #calculate mean xy
      # #Spreds_matrix = c((extent(ortho_crop)[2] + extent(ortho_crop)[1])/2, (extent(ortho_crop)[4] + extent(ortho_crop)[3])/2)
      # #convert it to png
      # ortho_crop <- as.array(ortho_crop)
      # ortho_crop <- image_read(ortho_crop / 255)
      # image_write(ortho_crop,format = "png",path =  paste0(img_folder,"mydiv_",sprintf("%7d",i),".png"))
      # #Crop the Moving window prediction
      ortho_pred = crop(pred_raster, extent(ortho, ind_grid[i,2], ind_grid[i,2]+res[g]-1, ind_grid[i,1], ind_grid[i,1]+res[g]-1))
      
      
      
      #get the pixel value of the mask
      imagevalue <- getValues(ortho_pred) |> table()
      # #convert the pred to png
      # ortho_pred <- as.array(ortho_pred)
      # ortho_pred <- image_read(ortho_pred/255)
      # image_write(ortho_pred,format = "png",path =  paste0(mask_folder,"mydiv_",sprintf("%7d",i),".png"))
      
      
      # #RUN INCASE FOR LOOP:calculate the number of pixels for each class
      # pixel_count[i,1] <- paste0("mydiv_",sprintf("%7d",i),".png")
      # #we put +1 because the first column is not our class
      # pixel_count[i,c(as.numeric(names(imagevalue)))+2] <- c(as.numeric(imagevalue))
      
      #pixel count data
      pixel_count <- data.frame(matrix(data=0, nrow=1, ncol=12))
      colnames(pixel_count)<- c("image", str_c("px_no_",c(0:10)))
      #RUN INCASE PARALLEL RUN
      pixel_count[,1] <- paste0("mydiv_",sprintf("%7d",i),".png")
      #we put +1 because the first column is not our class
      pixel_count[,c(as.numeric(names(imagevalue)))+2] <- c(as.numeric(imagevalue))
      
      return(pixel_count)
      
    }
    
    #export the pixel info as cv
    orthocrop_list[is.na(orthocrop_list)] <- 0
    write.csv(orthocrop_list, file=paste0(mask_folder,"pixelcount_Mask",".csv"))
    
    
    
    #write.csv(orthocrop_list,paste0(currentfolder,t,ref_poly@data$Species[t],"_","_xy.csv") )
    #removeTmpFiles(h=0.17)
    
  }
  
  
}






#stopCluster(cl)
# cores <- 30#detectCores()-8
# 
# cl <- makePSOCKcluster(cores)
# registerDoParallel(cl)
# 
# 
# ###main foreach loop #, 
# orthocrop_list <- foreach(i = seq_len(nrow(ind_grid)), .packages = c("raster", "rgdal", "keras", "magick"),.combine = rbind, .inorder = T) %dopar% {
#   #crop image
#   ortho_crop = crop(ortho, extent(ortho, ind_grid[i,2], ind_grid[i,2]+res-1, ind_grid[i,1], ind_grid[i,1]+res-1))
#   #plotRGB(ortho_crop)
#   
#   #calculate mean xy
#   preds_matrix = c((extent(ortho_crop)[2] + extent(ortho_crop)[1])/2, (extent(ortho_crop)[4] + extent(ortho_crop)[3])/2)
#   #convert it to png
#   ortho_crop <- as.array(ortho_crop)
#   ortho_crop <- image_read(ortho_crop / 255)
#   image_write(ortho_crop,format = "png",path =  paste0(currentfolder,sprintf("%7d",i),".png"))
#   #return the datafram of xy 
#   return(data.frame(x=preds_matrix[1],y=preds_matrix[2]))
# }
# 
# write.csv(orthocrop_list,paste0(currentfolder,t,ref_poly@data$Species[t],"_","_xy.csv") )
# removeTmpFiles(h=0.17)
