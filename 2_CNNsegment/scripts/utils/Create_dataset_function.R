
# tfdatasets input pipeline -----------------------------------------------

create_dataset <- function(data,
                           train, # logical. TRUE for augmentation of training data
                           batch, # numeric. multiplied by number of available gpus since batches will be split between gpus
                           epochs,
                           shuffle = TRUE, # logical. default TRUE, set FALSE for test data
                           tile_size = as.integer(tilesize),
                           dataset_size) { # numeric. number of samples per epoch the model will be trained on
  require(tfdatasets)
  require(purrr)
  
  if(shuffle){
    dataset = data %>%
      tensor_slices_dataset() %>%
      dataset_shuffle(buffer_size = length(data$img), reshuffle_each_iteration = TRUE)
  } else {
    dataset = data %>%
      tensor_slices_dataset() 
  } 
  
  dataset = dataset %>%
    dataset_map(~.x %>% list_modify( # read files and decode png
      img = tf$image$decode_jpeg(tf$io$read_file(.x$img), channels = chnl),
      msk = tf$image$decode_png(tf$io$read_file(.x$msk)) #%>%
      #tf$subtract(tf$constant(value = 1L, dtype = "uint8", shape = c(128L,128L,1L)))
      #tf$image$resize(size = c(tile_size, tile_size), method = "nearest")# %>%
      # tf$cast(dtype = "uint8") %>%
    )) %>% 
    dataset_map(~.x %>% list_modify(
      img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32),
      # convert datatype
      msk = tf$one_hot(.x$msk, depth = as.integer(no_classes), dtype = tf$float32) %>%
        tf$squeeze() # removes dimensions of size 1 from the shape of a tensor
    )) %>% 
    dataset_map(~.x %>% list_modify( # set shape to avoid error at fitting stage "tensor of unknown rank"
      img = tf$reshape(.x$img, shape = c(tile_size, tile_size, chnl)),
      msk = tf$reshape(.x$msk, shape = c(tile_size, tile_size, no_classes))
    ))
  
  if(train) {
    dataset = dataset %>%
      dataset_map(~.x %>% list_modify( # randomly flip up/down
        img = tf$image$random_flip_up_down(.x$img, seed = 1L),
        msk = tf$image$random_flip_up_down(.x$msk, seed = 1L)
      )) %>%
      dataset_map(~.x %>% list_modify( # randomly flip left/right
        img = tf$image$random_flip_left_right(.x$img, seed = 1L) %>%
          tf$image$random_flip_up_down(seed = 1L),
        msk = tf$image$random_flip_left_right(.x$msk, seed = 1L) %>%
          tf$image$random_flip_up_down(seed = 1L)
      )) %>%
      dataset_map(~.x %>% list_modify( # randomly assign brightness, contrast and saturation to images
        img = tf$image$random_brightness(.x$img, max_delta = 0.1, seed = 1L) %>%
          tf$image$random_contrast(lower = 0.9, upper = 1.1, seed = 2L) %>%
          tf$image$random_saturation(lower = 0.9, upper = 1.1, seed = 3L) %>% # requires 3 chnl -> with useDSM chnl = 4
          tf$clip_by_value(0, 1) # clip the values into [0,1] range.
      )) %>%
      dataset_repeat(count = ceiling(epochs * (dataset_size)) )
  }
  
  dataset = dataset %>%
    dataset_batch(batch, drop_remainder = TRUE) %>%
    dataset_map(unname) %>%
    dataset_prefetch_to_device("/gpu:0", buffer_size = tf$data$AUTOTUNE)
}


# 
# msk = tf$image$decode_png(tf$io$read_file(x$msk)) %>%
#   tf$subtract(tf$constant(value = 1L, dtype = "uint8", shape = c(128L,128L,1L)))  %>%
#   tf$one_hot(depth = as.integer(no_classes), dtype = tf$float32) %>%
#   #tf$one_hot(depth = no_classes, dtype = tf$uint8) %>% # one-hot encode masks
#   tf$squeeze() %>%
#   tf$reshape(shape = c(tilesize, tilesize, 1L))

# dataset_iter = reticulate::as_iterator(dataset)
# example = dataset_iter %>% reticulate::iter_next()
# example