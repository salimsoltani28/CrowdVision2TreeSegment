.libPaths("/home/ssoltani/R/x86_64-pc-linux-gnu-library/4.2")
library(tidyverse)
library(raster)
library(snow)
library(raster)
library(terra)
#ls('package:raster')

# Read command line arguments
args <- commandArgs(trailingOnly = TRUE)


message("Eexecuing the majority vote function on the :", paste(args))
predpath <- args

allimages <- list.files(predpath, pattern = ".tif", full.names = T)



for (i in seq(1, length(allimages), 3)) {
  rasterstack <-  raster::stack(allimages[i:(i+2)])
  # do something with chunk
  # majority vote
  beginCluster(n=20)
  finalPrediction = clusterR(rasterstack, calc, args = list(modal, na.rm = T, ties = "random"))
  endCluster()
  #crop the raster around the margin by 512 pix
  div_factor <- 8192* res(finalPrediction)[1]
  finalPrediction <- crop(finalPrediction, extent(finalPrediction)/div_factor)
  writeRaster(finalPrediction, paste0(predpath,"/",i, "_8kcrop_final_pred_majorityvote_test.tif"))
}




