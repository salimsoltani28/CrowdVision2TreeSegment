# Set Library Paths
library(reticulate)
# Setup Environment
reticulate::use_condaenv(condaenv = "tfr")
.libPaths("/home/ssoltani/R/x86_64-pc-linux-gnu-library/4.2")

# Load Libraries
libraries_to_load <- c(
  "reticulate", 
  "keras", 
  "tensorflow", 
  "tidyverse", 
  "tibble", 
  "rsample", 
  "magick", 
  "ggplot2", 
  "gtools"
)
lapply(libraries_to_load, library, character.only = TRUE)

# WandB Initialization
wandb <- import("wandb")
wandb$login(key = "753058fac9941b142125245452790d1caf6fa227")



# GPU Configuration
gpu_indices <- c(1, 2)
gpu_devices <- tf$config$list_physical_devices()[gpu_indices]
tf$config$set_visible_devices(gpu_devices)

gpu1 <- tf$config$experimental$get_visible_devices('GPU')[[1]]
tf$config$experimental$set_memory_growth(device = gpu1, enable = TRUE)

# Uncomment if you want to enable eager execution
# tfe_enable_eager_execution(device_policy = "silent")

# Uncomment if you are using MirroredStrategy
# strategy <- tf$distribute$MirroredStrategy()
# strategy$num_replicas_in_sync

# Set Parameters
tilesize <- 512L
chnl <- 3L
no_epochs <- 150L
no_classes <- 11L # one more class for NAs
batch_size <- 20L

# Set Working Directory
workdir <- "/scratch1/ssoltani/workshop/09_CNN_tree_species/1_Citzen_to_Unet_project/"
setwd(workdir)

# Load Utility Functions
source("scripts/utils/Create_dataset_function.R")
source("scripts/utils/Unet_model_ReLu.R")



# Function to load data
load_data <- function(base_path, model_dir_name) {
  full_path <- file.path(base_path, model_dir_name)
  
  csv_files <- list.files(path = full_path, 
                          pattern = ".csv", 
                          full.names = TRUE,
                          recursive = TRUE)
  
  all_csv <- list()
  
  for (q in seq_along(csv_files)) {
    all_csv[[q]] <- read.csv(csv_files[q]) %>% 
      mutate(image = str_c(str_remove(csv_files[q], "pixelcount_Mask.csv"), image))
  }
  
  return(all_csv)
}

# Base paths
base_path_ortho1 <- "/scratch1/ssoltani/workshop/09_CNN_tree_species/1_Citzen_to_Unet_project/dataset/02_myDiv_Second_Orthoimage_July_2022_Citizen_to_Unet_12classes/3_conservation_model/"
base_path_ortho2 <- "/scratch1/ssoltani/workshop/09_CNN_tree_species/1_Citzen_to_Unet_project/2_Citizen_to_Unet_project_Ortho_15malt/dataset/2_conservative_trainingdata/"

# Model directory names
model_dir_name_ortho1 <- "Mask_Output_effnet7_stamfiltering_distover0.2_Under15m_img512_11classGOOGLEimg_2Dense_256_512_No_sieve512/"
model_dir_name_ortho2 <- "Resampled_Mask_Output_13_googleimagepred_conservativePred_0.6_NoSieve512/"

# Load data
all_csv_ortho1 <- load_data(base_path_ortho1, model_dir_name_ortho1)
all_csv_ortho2 <- load_data(base_path_ortho2, model_dir_name_ortho2)



# Combine all data frames in the list into a single data frame
pixel_no_mask_ortho1 <- do.call("rbind", all_csv_ortho1)
pixel_no_mask_ortho2 <- do.call("rbind", all_csv_ortho2)
pixel_no_mask <- rbind(pixel_no_mask_ortho1,pixel_no_mask_ortho2)
# Filter and preprocess image data
Img_filter <- pixel_no_mask[,-1] %>% 
  pivot_longer(!image, names_to = "class", values_to = "count") %>% 
  mutate(count = round((count / (512 * 512)) * 100)) %>% 
  filter(count > 30)

# Explore the data using ggplot
ggplot(Img_filter, aes(x = count)) +
  geom_histogram() +
  facet_wrap(~ class) +
  ylim(0, 400)

# Check number of observations and duplicates
table(Img_filter$class)
table(duplicated(Img_filter$image))

# Sample images with replacement
sample_no <- 4000
Img_filter <- Img_filter %>%
  group_by(class) %>%
  sample_n(size = sample_no, replace = length(.) < sample_no)

# Verify sampling
table(Img_filter$class)



# Update image paths conditionally################################################################
path_img <- ifelse(
  grepl("02_myDiv_Second_Orthoimage_July_2022_Citizen_to_Unet_12classes", Img_filter$image), # Condition for ortho1
  str_replace(
    Img_filter$image,
    paste0( model_dir_name_ortho1),
    "img_02_Orthoimage_July_2022_512"
  ),
  str_replace(
    Img_filter$image,
    paste0( model_dir_name_ortho2),
    "Resampled_img_ortho_sept_finalcheck_512"
  )
)

#put the mask as it is
path_msk <- Img_filter$image

# Uncomment if you wish to remove duplicated records
# path_img <- path_img[!duplicated(path_img)]
# path_msk <- path_msk[!duplicated(path_msk)]

# Check for duplicates again
table(duplicated(path_img))
table(duplicated(path_msk))


# Loading Data ----------------------------------------------------------------


valIdx = sample(x = 1:length(path_img), size = floor(length(path_img)/8), replace = F)
val_img = path_img[valIdx]; val_msk = path_msk[valIdx]
train_img = path_img[-valIdx]; train_msk = path_msk[-valIdx]

train_data = tibble(img = train_img,
                    msk = train_msk)
val_data = tibble(img = val_img,
                  msk = val_msk)
dataset_size <- length(train_data$img)




# Parameters ----------------------------------------------------------------


dataset_size <- length(train_data$img)

training_dataset <- create_dataset(train_data, train = TRUE, batch = batch_size, epochs = no_epochs, dataset_size = dataset_size)
validation_dataset <- create_dataset(val_data, train = FALSE, batch = batch_size, epochs = no_epochs)


dataset_iter = reticulate::as_iterator(training_dataset)
example = dataset_iter %>% reticulate::iter_next()
example[[1]]
example[[2]]
par(mfrow=c(1,2))
plot(as.raster(as.array(example[[1]][1,,,1:3]), max = 1))
plot(as.raster(as.array(example[[2]][1,,,1]), max = 1))



#with(strategy$scope(), {
model <- get_unet_128()
#}))

model_dir_name <- "ResamplMask_distover0.2_Under15m_11classGOOGLEimg_2Dense_256_512_No_Sieve_512"
#output model name
output_model_name <- paste0(str_replace(model_dir_name,"/",""),"_Nepoch_Combi_data_bothOrtho",no_epochs,"_softmax_ReLu")
######################Monitoring
# Initialize the run
run <- wandb$init(
  # Set the project where this run will be logged
  #project = "citizen_science",
  project = output_model_name,
  # Track hyperparameters and run metadata
  config = list(
    learning_rate = lr,
    epochs = no_epochs
  )
)
# # Model fitting ----------------------------------------------------------------
# 
# checkpoint_dir <- paste0(workdir, "chekpoints/2_output_12_class_conservative_prediction/",output_model_name)
# unlink(checkpoint_dir, recursive = TRUE)
# dir.create(checkpoint_dir, recursive = TRUE)
# filepath = file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
# 
# cp_callback <- callback_model_checkpoint(filepath = filepath,
#                                          monitor = "dice_coef",
#                                          save_weights_only = FALSE,
#                                          save_best_only = TRUE,
#                                          verbose = 1,
#                                          #mode = "auto",
#                                          mode = "max",
#                                          save_freq = "epoch")
# 
# history <- model %>% fit(x = training_dataset,
#                          epochs = no_epochs,
#                          steps_per_epoch = dataset_size/(batch_size),
#                          callbacks = list(cp_callback, callback_terminate_on_naan(),wandb$keras$WandbCallback()),
#                          validation_data = validation_dataset)
# # 
# # 
# # 
# 
# # ###########################################Run this incase you want to resume the model training
# 
# 
# # Model fitting ----------------------------------------------------------------

checkpoint_dir <- paste0(workdir, "chekpoints/2_output_12_class_conservative_prediction/",output_model_name)
#unlink(checkpoint_dir, recursive = TRUE)
dir.create(checkpoint_dir, recursive = TRUE)
filepath = file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")

cp_callback <- callback_model_checkpoint(filepath = filepath,
                                         monitor = "dice_coef",
                                         save_weights_only = FALSE,
                                         save_best_only = TRUE,
                                         verbose = 1,
                                         #mode = "auto",
                                         mode = "max",
                                         save_freq = "epoch")

###resume training
# Load the latest checkpoint

#function to load the best model
load_best_model = function(path){
  loss = length(list.files(path))
  best = max(loss)
  print(paste0("Loaded model of epoch ", best, "."))
  load_model_hdf5(paste0(path,"/", list.files(path)[best]), compile=FALSE)
}

latest <- load_best_model(checkpoint_dir)

completed_epoch <- 70L
# Load the previously saved weights
weights <- get_weights(latest)
model %>% set_weights(weights)

history <- model %>% fit(x = training_dataset,
                         epochs = no_epochs,
                         initial_epoch = completed_epoch,
                         steps_per_epoch = dataset_size/(batch_size),
                         callbacks = list(cp_callback, callback_terminate_on_naan(),wandb$keras$WandbCallback()),
                         validation_data = validation_dataset)


###finish tracking
wandb$finish()
