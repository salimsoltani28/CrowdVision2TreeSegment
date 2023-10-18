library(reticulate)
reticulate::use_condaenv("tf29", required = TRUE)
#Libraries
require(keras)
library(tensorflow)
library(tfdatasets)
library(tidyverse)
library(tibble)
library(rsample)
library(countcolors)
library(reticulate)
library(gtools)
#library(sampler)
library(rgdal)

#set a seed
tf$random$set_seed(25L)
set.seed(28)

# set memory growth policy
gpu1 <- tf$config$experimental$get_visible_devices('GPU')[[1]]
#gpu2 <- tf$config$experimental$get_visible_devices('GPU')[[2]]
tf$config$experimental$set_memory_growth(device = gpu1, enable = TRUE)



#set the work and out dir
workdir = "/scratch1/ssoltani/workshop/09 CNN tree species/"
outdir = "/scratch1/ssoltani/workshop/09 CNN tree species/Tree_species_output/"
setwd(workdir)



# image size for distance filtering
xres = 256L
yres = 256L
n_bands = 3L
no_class = 11L #11 class the forest floor is exluded
#create the oupt directory 
dir.create(paste0(outdir), recursive = TRUE)


# Loading list of photographs
#----------------------------------------------------------------
#get the photos path for other species along their species label for sampling (to ensure we have photographs from all species)

#list the species folder
iNat_folders <- list.dirs(path = paste0(getwd(),"/01 myDiv_tree_spec_training_photos/" ),recursive = FALSE)

#read all files
iNat_img <- as.data.frame(matrix(nrow = 1,ncol = 2))[-1,]
#give columns name
colnames(iNat_img) <- c("img","ref")

for(g in 1:length(iNat_folders)){
  images <- mixedsort(list.files(iNat_folders[g], full.names = T, pattern = ".jpg", recursive = T))
  ref <- rep(as.integer(g),length(images))
  findata <- tibble(img=images,ref=ref)
  iNat_img <- rbind(iNat_img,findata)
}

#add iNat column
iNat_img <- iNat_img|> mutate(data_source= "iNat")
table(iNat_img$ref)
###plntNet data
#list the species folder
plannetdata_path <- "/scratch1/ssoltani/workshop/09 CNN tree species/04_PlantNet_data/"
PlantNet_folders <- list.dirs(path = plannetdata_path,recursive = FALSE)#[-1]

#read all files
planNet_path_img2 <- as.data.frame(matrix(nrow = 1,ncol = 2))[-1,]


#image pattern

for(g in 1:length(PlantNet_folders)){
  images1 <- mixedsort(list.files(PlantNet_folders[g], full.names = T, pattern = ".jpg", recursive = T))
  ref1 <- rep(as.integer(g),length(images1))
  findata1 <- tibble(img=images1,ref=ref1)
  planNet_path_img2 <- rbind(planNet_path_img2,findata1)
}
table(planNet_path_img2$ref)

#add data source column
planNet_path_img2<- planNet_path_img2|> mutate(data_source= "plantNet")
#combine both data from iNat+PlantNet
iNat_PlanNet<- bind_rows(iNat_img,planNet_path_img2 )
######################################################################################### Filter the data according to Angle and distance
#first run the create datasets in the below 
all_imgs1 <- iNat_PlanNet#tibble(img=iNat_PlanNet$img,ref= iNat_PlanNet$ref)
table(all_imgs1$ref)
# tfdatasets input pipeline -----------------------------------------------
create_dataset <- function(data,
                           train, # logical. TRUE for augmentation of training data
                           batch, # numeric. multiplied by number of available gpus since batches will be split between gpus
                           epochs,
                           shuffle, # logical. default TRUE, set FALSE for test data
                           dataset_size){ # numeric. number of samples per epoch the model will be trained on
  
  
  # data1 <- data %>% group_by(ref) %>% sample_n(nrow(filter(data,data$ref==0))) %>% ungroup()
  # data2 <- data1[,1]
  # ref <- to_categorical(unlist(as.list(data1[,2])))
  # data <- tibble(data2, ref)
  if(shuffle){
    #sample subset of data in each epoch
    dataset = data %>%  
      tensor_slices_dataset() %>%
      
      dataset_shuffle(buffer_size = length(data$img), reshuffle_each_iteration = TRUE)
  } else {
    dataset = data %>%
      tensor_slices_dataset()
  }
  
  
  dataset = dataset %>%
    dataset_map(~.x %>% purrr::list_modify( # read files and decode png
      #img = tf$image$decode_png(tf$io$read_file(.x$img), channels = no_bands)
      img = tf$image$decode_jpeg(tf$io$read_file(.x$img)
                                 , channels = n_bands
                                 #, ratio = down_ratio
                                 , try_recover_truncated = TRUE
                                 , acceptable_fraction=0.5
      ) %>%
        tf$cast(dtype = tf$float32) %>%  
        tf$math$divide(255) %>% 
        #tf$image$convert_image_dtype(dtype = tf$float32) %>%
        tf$keras$preprocessing$image$smart_resize(size=c(xres, yres))))
  
  
  
  #you can use this function incase you dont want to use data augmentation
  # if(train) {
  #   
  #   dataset = dataset %>%
  #     dataset_repeat(count = ceiling(epochs *(dataset_size/length(train_data$img))))}
  
  if(train) {
    
    dataset = dataset %>%
      dataset_map(~.x %>% purrr::list_modify( # randomly flip up/down
        img = tf$image$random_flip_up_down(.x$img) %>%
          tf$image$random_flip_left_right() %>%
          tf$image$random_brightness(max_delta = 0.1, seed = 1L) %>%
          tf$image$random_contrast(lower = 0.9, upper = 1.1) %>%
          tf$image$random_saturation(lower = 0.9, upper = 1.1) %>% # requires 3 chnl -> with useDSM chnl = 4
          tf$clip_by_value(0, 1) # clip the values into [0,1] range.
        
      )) %>% #,num_parallel_calls = tf$data$experimental$AUTOTUNE
      #),num_parallel_calls = NULL) %>%
      dataset_repeat(count = ceiling(epochs *(dataset_size/length(train_data$img))))
  }
  dataset = dataset %>%
    dataset_batch(batch, drop_remainder = TRUE) %>%
    dataset_map(unname) %>%
    #dataset_prefetch(buffer_size = tf$data$experimental$AUTOTUNE)
    dataset_prefetch_to_device(device = "/gpu:0", buffer_size =tf$data$experimental$AUTOTUNE)
}





#take the data to the pipeline
all_imgs <- create_dataset(data = all_imgs1,train = FALSE,batch = 1,shuffle = FALSE)

######################angle prediction and denormalize
angle_model <- load_model_hdf5("00 Angle_Dist_stam_filter_models/Angle_January_2022_ResNet50_v2_littledatatest_weights.49-0.03.hdf5")
#make prediction
angle_pred <- predict(object = angle_model,x=all_imgs)

#denormalize the predictions
minofdata_angle <- -90
maxofdata_angle <- 90
#function
denormalize <- function(x,minofdata_angle,maxofdata_angle) {
  x*(maxofdata_angle-minofdata_angle) + minofdata_angle
}

angle_pred_denormalized <- denormalize(angle_pred,minofdata_angle,maxofdata_angle)

###################################Distance prediction and denormalize
#load the trained model
Dist_model <- load_model_hdf5("00 Angle_Dist_stam_filter_models/Log_transform_Distweights.49-0.01.hdf5")

#Distance predictions
Dist_imgs_pred <- predict(object = Dist_model,x=all_imgs)


#denormalize the predictions
#logtransformation
minofdata <- -2.302585
maxofdata <- 5.010635
##############################normal transformation
# minofdata <- 0.1
# maxofdata <- 150

#denormalize function
denormalize <- function(x,minofdata,maxofdata) {
  x*(maxofdata-minofdata) + minofdata
}

Dist_pred_denormalized <- exp(denormalize(Dist_imgs_pred,minofdata,maxofdata))

###########################################put the images, ref, angle and distance prediction together


###model for stam no stam 
stam_model <- load_model_hdf5("00 Angle_Dist_stam_filter_models/stam_no_stam_weights.39-0.00.hdf5")
stam_no_stam <- as.array(k_argmax(predict(object = stam_model,x=all_imgs)))

###join the data
all_imgs_pred_join <- tibble(all_imgs1,dist= Dist_pred_denormalized[,1],stam_nostam=stam_no_stam,angle=angle_pred_denormalized )#,angle=angle_pred_denormalized[,1]



############################################sample the number of images to combine with PlanNet data

#filter out images with distance
all_imags_filtered <- all_imgs_pred_join %>%
  #set all dist for grass to 5
  #filter(!ref==11) %>%
  filter(stam_nostam==0 , dist>0.2 ,dist<15) #%>%

table(all_imags_filtered %>% filter(data_source=="iNat") %>% pull(ref))
table(all_imags_filtered %>% filter(data_source=="plantNet") %>% pull(ref))
table(all_imags_filtered$ref)
############################################sample the number of images to combine with PlanNet data

#sample number
num_sample <- 8000
#Check the data
PlanNetdata <- all_imags_filtered %>%
  filter(data_source=="plantNet", !ref==11) #%>%
# group_by(ref) %>%
# sample_n(length(.))
table(PlanNetdata$ref)- table(planNet_path_img2$ref)[-11]

#remaining sample sizes
sample_sizes <- c(num_sample-table(PlanNetdata$ref))[-c(3,8,10)]
sample_sizes_replace <- c(num_sample-table(PlanNetdata$ref))[c(3,8,10)]

#sample the iNat data
iNatData <- all_imags_filtered %>%
  group_by(ref) %>%
  filter(data_source=="iNat", !n()<num_sample) %>%
  nest() %>%
  ungroup() %>%
  mutate(n=sample_sizes) %>%
  mutate(samp=map2(data, n,sample_n)) %>%
  select(-c(data,n)) %>%
  unnest(cols = c(samp))
#sample_n(n()-num_sample)

#sample with replacment
iNatData_withreplac <- all_imags_filtered %>%
  group_by(ref) %>%
  filter(data_source=="iNat", !n()>num_sample) %>%
  nest() %>%
  ungroup() %>%
  mutate(n=sample_sizes_replace) %>%
  mutate(samp=map2(data, n,sample_n,replace=T)) %>%
  select(-c(data,n)) %>%
  unnest(cols = c(samp))
#sample grass data with replacement
grassclass <- all_imags_filtered %>%
  filter(ref==11) %>%
  group_by(ref) %>%
  sample_n(num_sample,replace = T)

#combine all data



################################################################combine together iNat + planNet
#
# #####integrate the Grass class
all_imags_filtered <- rbind(PlanNetdata,iNatData,iNatData_withreplac,grassclass)
table(all_imags_filtered$ref)
##########################
#creat new var for img ref and img
path_img_n <- all_imags_filtered
path_img_n$ref <- path_img_n$ref-1L 

##########################################################################

path_img <- path_img_n$img 
ref2 <- path_img_n$ref
unique(ref2)
ref <-to_categorical(ref2)
# split test data (10%) and save to disk 


#split test data (10%) and save to disk
testIdx = sample(x = 1:length(path_img), size = floor(length(path_img)/10), replace = F)
test_img = path_img[testIdx]
save(test_img, file = paste0(outdir, "test_img.RData"), overwrite = T)
test_ref = ref[testIdx,]
save(test_ref, file = paste0(outdir, "test_ref.RData"), overwrite = T)
# split training and validation data
path_img = path_img[-testIdx]
ref = ref[-testIdx,]
valIdx = sample(x = 1:length(path_img), size = floor(length(path_img)/5), replace = F)
val_img = path_img[valIdx] 
val_ref = ref[valIdx,] 
train_img = path_img[-valIdx]
train_ref = ref[-valIdx,]

train_data = tibble(img = train_img, ref = train_ref) 
val_data = tibble(img = val_img, ref = val_ref)

###redefine the image size
xres = 512L
yres = 512L


#######################################################################################Parameters###################################################
# Parameters----------------------------------------------------------------
batch_size <-15 # 12 (multi gpu, 512 a 2cm --> rstudio freeze) 
n_epochs <- 50
dataset_size <- length(train_data$img) # if ortho is combined with DSM = 4 (RGB + DSM), if not = 3 (RGB)

#create training data
training_dataset <- create_dataset(
  train_data, 
  train = TRUE, 
  batch = batch_size, 
  epochs = n_epochs, 
  dataset_size = dataset_size,shuffle = TRUE) 

#create validation data
validation_dataset <- create_dataset(
  val_data, 
  train = FALSE, 
  batch = batch_size, 
  epochs = n_epochs,shuffle = TRUE)

# with the following lines you can test if your input pipeline produces meaningful tensors. You can also use as.raster, etc... to visualize the frames.
dataset_iter = reticulate::as_iterator(training_dataset)
example = dataset_iter %>% reticulate::iter_next() 
example
plotArrayAsImage(as.array(example[[1]][1,,,]))
example[[2]][1,]

dataset_iter = reticulate::as_iterator(validation_dataset)
example = dataset_iter %>% reticulate::iter_next() 
example
plotArrayAsImage(as.array(example[[1]][4,,,]))
example[[2]][1,]

###########################################################clr


# Defining Model----------------------------------------------------------------

#model backbone
base_model <- tf$keras$applications$EfficientNetV2L(
  input_shape = c(xres, yres, n_bands),
  include_top = FALSE,
  include_preprocessing = FALSE,
  weights = NULL
  #weights = "imagenet",# 0.2 is default
  #include_preprocessing=True,
  #pooling = NULL
)
# base_model <- application_efficientnet_b7(
#   include_top = FALSE,
#   input_shape = c(xres, yres, n_bands)) #,drop_connect_rate=0.1, pooling = NULL

# add our custom layers
# add our custom layers
predictions <- base_model$output %>%
  
  layer_global_average_pooling_2d()%>% 
  layer_dropout(rate = 0.5) %>% 
  #layer_batch_normalization() %>% 
  #layer_flatten() %>% 
  #layer_dense(units = 1024, kernel_regularizer = regularizer_l2(0.001),activation = 'relu') %>% 
  #layer_dropout(rate = 0.5) %>% 
  # layer_dense(units = 512,kernel_regularizer = regularizer_l2(0.001),activation = 'relu') %>% #
  # layer_dropout(rate = 0.5) %>% 
  # layer_dense(units = 1024, kernel_regularizer = regularizer_l2(0.0001),activation = 'relu') %>% #
  # layer_dropout(0.5) %>% 
  layer_dense(units = 512, kernel_regularizer = regularizer_l2(0.001),activation = 'relu') %>% #
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = no_class, activation = 'softmax')
# this is the model we will train
model <- keras_model(inputs = base_model$input, outputs = predictions)

#compile the backbone
model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.0001),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

#}) incase of parallel GPU
####################################################################################cyclic learning
#
# library(KerasMisc)
#
# iter_per_epoch <- nrow(train_data) / 50
# callback_clr <- new_callback_cyclical_learning_rate(
#   step_size = iter_per_epoch * 2,
#   base_lr = 0.0001,
#   max_lr = 0.0006,
#   mode = "triangular",
#   patience = 3,
#   factor = 0.9,
#   cooldown = 2,
#   verbose = 0
# )

#############################################################################################



checkpoint_dir <- paste0(outdir, "Output_effNetV2L_stamfiltering_distover0.2_Under15m_img512_11class_Avgpool_drop0.5_Dense512_Drop0.5_AdamLr0.0001_8kdata_iNat_Batch15_PlanNet_Dec18_poolActive")
unlink(checkpoint_dir, recursive = TRUE)
dir.create(checkpoint_dir, recursive = TRUE)
filepath = file.path(checkpoint_dir,
                     "weights.{epoch:02d}-{val_loss:.2f}.hdf5")

cp_callback <- callback_model_checkpoint(filepath = filepath,
                                         monitor = "val_loss",
                                         save_weights_only = FALSE,
                                         save_best_only = TRUE,
                                         verbose = 1,
                                         mode = "auto",
                                         save_freq = "epoch")

#put class weight
#class_weight <- list("0"=11.99884, "1"=1, "2"=11.23185)
history <- model %>% fit(x = training_dataset,
                         epochs = n_epochs,
                         steps_per_epoch = dataset_size/batch_size,
                         callbacks = list(cp_callback,
                                          callback_terminate_on_naan()),
                         #class_weight=class_weight,
                         validation_data = validation_dataset)

####
setwd(checkpoint_dir)
dev.off()

pdf(width=8,height=8,paper='special')
plot(history)


#plot(history)
dev.off()



####
saveRDS(objects,"Environment.RData")
setwd(workdir)
#####################
#### EVALUTATION ####
#####################
#outdir = "results/"

checkpoint_dir <- paste0( outdir, "checkpoints/")
load(paste0(outdir, "test_img.RData"))
load(paste0(outdir, "test_ref.RData"))
testdata = tibble(img = test_img,
                  ref = test_ref)
test_dataset <- create_dataset(testdata, train = FALSE, batch = 1, 
                               shuffle = FALSE)

model = load_model_hdf5('weights.49-0.06.hdf5', compile = TRUE)

eval <- evaluate(object = model, x = test_dataset)
eval

plot(history)
dev.off()
pdf( width = 8, height = 8, paper = 'special')
plot(history)
dev.off()
history_metrics = history$metrics
save(history_metrics, file = paste0(workdir, outdir, "model_history.RData"))


test_pred = predict(model, test_dataset)
dim(test_pred)

test_pred

sum(test_pred[10,])

