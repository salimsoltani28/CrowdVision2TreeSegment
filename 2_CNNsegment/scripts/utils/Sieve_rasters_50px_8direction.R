
.libPaths("/home/ssoltani/R/x86_64-pc-linux-gnu-library/4.2")
library(reticulate)
library(tidyverse)
library(raster)
library(terra)
library(foreach)
library(doParallel)
#library(snow)


# Read command line arguments
args <- commandArgs(trailingOnly = TRUE)


message("Eexecuing the Sieve vote function on the :", paste(args))
predpath <- args


allimages <- list.files(predpath, pattern = ".tif", recursive = T, full.names = T)

cores <- 10#detectCores()-20
cl <- makePSOCKcluster(cores)
registerDoParallel(cl)
clusterEvalQ(cl,.libPaths("/home/ssoltani/R/x86_64-pc-linux-gnu-library/4.2"))

no_pix=500
foreach(i = 1:length(allimages), .packages = c("raster", "rgdal", "terra"), .inorder = T) %dopar% {
    
  raster1 <- terra::sieve(rast(allimages[i]), threshold=no_pix, directions=8)
  raster1[raster1<0] <- NA
  writeRaster(raster1, paste0(predpath,"/",i, "5kpixel_final_pred_majorityvote_test_SIEVED_",no_pix,"PX.tif"))
}

stopCluster(cl)
