# Load data for each locality ---------------------------------------------

load("sandyTur_data.RData")
model_name <- "sandyTur"


# Load user-defined functions for AD in time-series data ------------------

# functions for AD using ANN, both for supervised (sRNN_fit) and semi-supervised classification (uRNN_fit)

source("AD_fitFuns.R")
source("AD_performance.R")



# Example of semi-supervised classification -------------------------------

data_flags <- list(n_features = 1, n_split = 2, p_split = 0.5)

# User-defined set of hyper-parameters

RNN_flags <- data.frame(n_layers = 1, n_units = 16, dropout = 0.5, activation = "linear", weight_constraint = 1, optimizer_type = "sgd", momentum = 0.5,
                        learning_rate = 0.3, kernel_initializer = "zero", batch_size = 1024, sld_window = 1, thrsld_class = 0.5)
  
uRNN_exmpl <- uRNN_fit(data_vars, data_label, data_flags, RNN_flags, paste(model_name, "_uModel"))
uRNN_exmpl$scores

#uModel_exmpl <- load_model_hdf5(paste0(model_name, "_uModel.h5"))

for (cc in unlist(ClassAll)) {
  RatiosClass[1,cc] <- round(sum(data_label[data_type == cc]))
  RatiosClass[2,cc] <- round(sum(uRNN_exmpl$classify[data_type == cc], na.rm = T))
  RatiosClass[3,cc] <- RatiosClass[2,cc] / RatiosClass[1,cc]
}



# Example of supervised classification ------------------------------------

data_flags <-  list(n_features = 1, n_split = 1, p_split = 0.5)

sRNN_exmpl <- sRNN_fit(data_vars, data_label, data_flags, RNN_flags, paste(model_name, "_sModel"))
sRNN_exmpl$scores

#sModel_exmpl <- load_model_hdf5(paste0(model_name, "_sModel.h5"))

for (cc in unlist(ClassAll)) {
  RatiosClass[1,cc] <- round(sum(data_label[data_type == cc]))
  RatiosClass[2,cc] <- round(sum(sRNN_exmpl$classify[data_type == cc], na.rm = T))
  RatiosClass[3,cc] <- RatiosClass[2,cc] / RatiosClass[1,cc]
}