# Artificial Neural Networks for Anomaly detection in time-series data (AD_ANN)

Here we provide tools for using **Artificial Neural Networks** (ANN) for detecting anomalies in high-frequency water-quality data. Latter tools could be also applied to numerous applications for detecting anomalies in time-series data using ANNs, not only for environmental data. We applied **Recurrent Neural Networks** (RNN) as a particular case of ANNs which is designed to handle sequence-dependent strings of data. In particular, the **Long Short-Term Memory networks** (LSTM) are a type of RNN commonly used in deep learning for time-series forescasting and are specially useful to include memory during the learning process.   

Given the large number of hyper-parameters values to be tested, we tuned ANN models using a sampling optimization methods for obtaining the best model performance. To do so we perform a **Bayesian optimization** as a class of optimisation methods for hyper-parameter tuning commonly used in machine-learning methods.


## Getting Started

There a large number of libraries available in *R* for fitting ANN, and here we computed and fitted ANNs with [Keras](https://keras.rstudio.com/), a model-level library which provides high-level building for programming for developing deep-learning models. **Keras** is a high level wrapper of **TensorFlow** and help to provide simplified way to build neural networks from standard types of layers and facilitate a reproducible platform for developing deep learning approaches.

For optimization of ANN models, we applied the toolbox [mlrMBO](https://mlrmbo.mlr-org.com/) implemented in R, which provide excellent performance for expensive optimization scenarios for single- and multi-objective tasks, with continuous or mixed parameter spaces.


### Prerequisites

Before running models, it is necesary to install library dependencies for both [Keras](https://keras.rstudio.com/articles/getting_started.html) and [mlrMBO](https://mlrmbo.mlr-org.com/). Of special importance is the installation of **Keras**, given the large amount of dependencies necessary for properly user **Keras** under R interface: To do so please to follow carefully the installation process.


## Running a single model

First load data for the selected locality. Here We provide an example of Turbidity data log-transformed (*data_vars*) and the labelled anomalies (*data_label*) defined as "normal" (0) and anomalies (1). Finally we provide an array defining the sequence of the sliding window (*sld_window.seq*) which defines the lenght of the matrix (in number of colums) of the time-series dataset. In addition, we provide an array of the types of anomalies (*data_type*) and the types of anomalous classes defined by users (*RatiosClass*, *ClassAll*), which depend on each type of data. In our case, types of anomalies were based on those defined by Leigh et al. (2019) in water-quality data.

```
load("sandyTur_data.RData")
data_name <- "sandyTur"
```

Second, load functions for anomaly detection in time-series. It includes functions both for semi-supervised (*uRNN_fit*) and supervised (*sRNN_fit*) classification using ANNs. In addition there are also auxiliary functions to re-shape the data during model fitting.

```
source("AD_fitFuns.R")
source("AD_performance.R")
```

Third, we setup hyperparameters for running a single model.  

- Parameters to split data (*data_flags*). It includes the number fitted variables (*n_features*), the number of splits (*n_split*) and the split location relative of the whole time-series (*p_split*). There are pre-defined values depending on the type of classification, but can be modified if necessary.

- Hyperparameters to fit RNN models (*RNN_flags*). It includes a large number of hyperparameters related to (i) the network structure and (ii) the training algorithm. It is necessary to define the complete set of hyperparameters for model running, except *sld_window* (1) and *thrsld_class* (0.5).

```
data_flags <- list(n_features = 1, n_split = 2, p_split = 0.5)

RNN_flags <- data.frame(n_layers = 1, n_units = 16, dropout = 0.5, activation = "linear", weight_constraint = 1, optimizer_type = "sgd", momentum = 0.5,
                        learning_rate = 0.3, kernel_initializer = "zero", batch_size = 1024, sld_window = 1, thrsld_class = 0.5)
```

We provide an example of semi-supervised classification and we obtain the performance scores. If available it is possible to compare the anomaly detection for different types of anomalies.

```
uRNN_exmpl <- uRNN_fit(data_vars, data_label, data_flags, RNN_flags, paste(data_name, "_uRnd"))
uRNN_exmpl$scores

for (cc in unlist(ClassAll)) {
  RatiosClass[1,cc] <- round(sum(data_label[data_type == cc]))
  RatiosClass[2,cc] <- round(sum(uRNN_exmpl$classify[data_type == cc], na.rm = T))
  RatiosClass[3,cc] <- RatiosClass[2,cc] / RatiosClass[1,cc]
}
```

We can additionaly calculate the performance for a supervised classifcation with the same set of hyperparameter values defined above.

```
data_flags <-  list(n_features = 1, n_split = 1, p_split = 0.5)

sRNN_exmpl <- sRNN_fit(data_vars, data_label, data_flags, RNN_flags, paste(data_name, "_sRnd"))
sRNN_exmpl$scores

for (cc in unlist(ClassAll)) {
  RatiosClass[1,cc] <- round(sum(data_label[data_type == cc]))
  RatiosClass[2,cc] <- round(sum(sRNN_exmpl$classify[data_type == cc], na.rm = T))
  RatiosClass[3,cc] <- RatiosClass[2,cc] / RatiosClass[1,cc]
}
```


## Optimizing hyperparameters to get the best model

Before MBO can really start, it necessary to provide a initial design of values to be evaluated. In our case we defined a set of hyper-parameter values and the values to be possible to be included in the model.

```
library('mlrMBO')

ParamSet <- makeParamSet(
  makeIntegerParam("n_layers", lower = 1, upper = 3),
  makeDiscreteParam("n_units", values = 2^(1:6)),
  makeDiscreteParam("dropout", values = c(0.001, seq(0.1, 0.9, 0.2))),
  makeDiscreteParam("activation", values = c("relu","tanh","sigmoid", "hard_sigmoid", "linear")),
  makeIntegerParam("weight_constraint", lower = 1, upper = 5),
  
  makeDiscreteParam("optimizer_type", values = c("sgd", "RMSProp", "adagrad", "adam", "adamax")),
  makeDiscreteParam("momentum", values = c(0.001, seq(0.1, 0.9, 0.2))),
  makeDiscreteParam("learning_rate", values = c(0.001, 0.01, 0.1, 0.2, 0.3)),
  makeDiscreteParam("kernel_initializer", values = c("uniform", "normal", "zero", "lecun_uniform", "lecun_normal")),
  makeDiscreteParam("batch_size", values = 4^(1:5)), 
  
  makeDiscreteParam("sld_window", values = sld_window.seq),
  makeDiscreteParam("thrsld_class", values = c(seq(0.4, 0.9, 0.1), 0.95, 0.999)))
```

It is necessary to generate a sufficient number of samples to get the best model. Here we used an strategy of 250 samples of random search and 250 samples of model optimization. Those values will depend on each database. We additionally defined the corresponding objective value. In our case we used the **balanced accuracy** as a classification metrics for imbalanced data, as that occurred with detection of anomalies.  

```
RNN_reps <- list(init = 250, opt = 250)

fit_scores <- "b_acc"

```

We start start with random search of hyperparameter values. 

```
uRNN_rnd.reps <- RNN_reps$init 
ParamDesign <- generateDesign(n = uRNN_rnd.reps, par.set = ParamSet)
uRNN_scores <- data.frame()

for (j in 1:uRNN_rnd.reps) {
  cat("Running starting values // Model: uRNN // Vars:", selectVars, "// Complete:", j/dim(ParamDesign)[1]*100, "%"); print(j)
  
  RNN_flags <- ParamDesign[j,]
  uRNN_rnd <- uRNN_fit(data_vars, data_label, data_flags, RNN_flags, paste0(data_name, "uRnd"))
  uRNN_scores <- rbind(uRNN_scores, uRNN_rnd$scores)
  ParamDesign$y[[j]] <- uRNN_rnd$scores[[fit_scores]]
  
  write.csv(ParamDesign, file = paste0(data_name, "_uBRndPoints.csv"))
  write.csv(uRNN_scores, file = paste0(data_name, "_uBRndscores.csv"))
}

```

And we continue with optimization of hyperparameters to get the best model based on maximizing the objective value.

```
uRNN_ctrl = makeMBOControl()
uRNN_ctrl = setMBOControlInfill(uRNN_ctrl, crit = crit.ei)

uRNN_opt.state = initSMBO(
  par.set = ParamSet,
  design = ParamDesign,
  control = uRNN_ctrl,
  minimize = TRUE,
  noisy = FALSE)

uRNN_opt.reps <- RNN_reps$opt
uRNN_proposePoints <- NULL
uRNN_scores <- data.frame()

start.time <- Sys.time()

for (j in 1:uRNN_opt.reps) {
  cat("Optimizing values // Model: uRNN // Data:", data_name, "// Complete:", j/uRNN_opt.reps*100, "%"); print(j)
  
  RNN_flags = proposePoints(uRNN_opt.state)$prop.points
  uRNN_opt = uRNN_fit(data_vars, data_label, data_flags, RNN_flags, paste0(data_name, "_uBOpt"))
  uRNN_scores <- rbind(uRNN_scores, uRNN_opt$scores)
  y <- uRNN_opt$scores[[fit_scores]]
  uRNN_proposePoints <- rbind(uRNN_proposePoints, cbind(j, RNN_flags, y))
  
  write.csv(uRNN_scores, file = paste0(data_name, "_uBOptscores.csv"))
  write.csv(uRNN_proposePoints, file = paste0(data_name, "_uBOptproposePoints.csv"))
  
  updateSMBO(uRNN_opt.state, 
             x = RNN_flags,
             y = y)
}

time.taken <- data.frame(time.taken, OptPoints = Sys.time() - start.time)

proposePoints(uRNN_opt.state)
finalizeSMBO(uRNN_opt.state)
uRNN_opt.state$opt.result$mbo.result$x
uRNN_opt.state$opt.result$mbo.result$y
plot(finalizeSMBO(uRNN_opt.state))

uRNN_proposePoints <- read.csv(file = paste0(data_name, "_uBOptproposePoints.csv"))
ParamBest <- uRNN_proposePoints[order(uRNN_proposePoints$y, decreasing = T),][1,][3:14]

modelBest <- load_model_hdf5(paste0(data_name, "_uBOptBest.h5"))

for (cc in unlist(ClassAll)) {
  RatiosClass[1,cc] <- round(sum(data_label[data_type == cc]))
  RatiosClass[2,cc] <- round(sum(uRNN_best$classify[data_type == cc], na.rm = T))
  RatiosClass[3,cc] <- RatiosClass[2,cc] / RatiosClass[1,cc]
}
```


## References

Bischl, B., Richter, J., Bossek, J., Horn, D., Thomas, J., & Lang, M. (2017). mlrMBO: A modular framework for model-based optimization of expensive black-box functions. arXiv preprint arXiv:1703.03373.

Leigh, C., Alsibai, O., Hyndman, R. J., Kandanaarachchi, S., King, O. C., McGree, J. M., ... & Mengersen, K. (2019). A framework for automated anomaly detection in high frequency water-quality data from in situ sensors. Science of The Total Environment, 664, 885-898.


## Deployment

Add additional notes about how to deploy this on a live system


## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Javier Rodriguez-Perez** - *Initial work* - [PurpleBooth](https://github.com/jvrrodriguez)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
