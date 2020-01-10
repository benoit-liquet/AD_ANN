# Artificial Neural Networks for Anomaly detection in time-series data (AD_ANN)

Here we provide tools for using **Artificial Neural Networks** (ANN) for detecting anomalies in high-frequency water-quality data. Latter tool could be also applied to to numerous applications for detecting anomalies in time-series data using ANNs. To do so, we applied Recurrent Neural Networks (RNN) as a particular case of ANNs which is designed to handle sequence-dependent strings of data. In particular, the **Long Short-Term Memory networks** (LSTM) are a type of RNN commonly used in deep learning for time-series forescasting and are specially useful dut to include memory during the learning process.   

Given the large number of hyper-parameters values to be tested to get the best model performance, we tuned ANN models using a sampling optimization methods. To do so we perform a**Bayesian optimization** as a class of optimisation methods for hyper-parameter tuning commonly used in machine-learning methods.


## Getting Started

There a large number of libraries available in R for fitting ANN, but here we computed and fitted ANNs with [Keras](https://keras.rstudio.com/) , a model-level library which provides high-level building for programming for developing deep-learning models. **Keras** is a high level wrappers of **TensorFlow** and help to provide simplified way to build neural networks from standard types of layers and facilitate a reproducible platform for developing deep learning approaches.

For optimization of ANN models, we applied the toolbox [mlrMBO](https://mlrmbo.mlr-org.com/) implemented in R, which provide excellent performance for expensive optimization scenarios for single- and multi-objective tasks, with continuous or mixed parameter spaces.


### Prerequisites

Before running models, it is necesary to install library dependencies of both [Keras](https://keras.rstudio.com/articles/getting_started.html) and [mlrMBO](https://mlrmbo.mlr-org.com/). Of special importance is the installation of **Keras**, given the large amount of dependencies necessary for properly running under R interface and it is thus necessary to follow carefully the installation process.


## Running a single model

First load data for the selected locality. Here We provide an example of Turbidity data log-transformed (*data_vars*). It also includes labelled anomalies (*data_label*) and the types of anomalies (*data_type*), if available. Finally we provide an array defining the sequence of the sliding window (*sld_window.seq*) which defines the lenght of the matrix (in number of colums) of the time-series dataset.

```
load("sandyTur_data.RData")
model_name <- "sandyTur"
```

Second, load user-defined functions for anomaly detection in time-series. It includes functions both for semi-supervised (*uRNN_fit*) and supervised (*sRNN_fit*) classification using ANNs. In addition there are also auxiliary functions to re-shape the data during model fitting.

```
source("AD_fitFuns.R")
source("AD_performance.R")
```

Third, we setup hyperparameters for running a single model.  

- Parameters to split data (*data_flags*). It includes the number fitted variables (n_features), the number of splits (n_split) and the split location relative of the whole time-series (p_split). By default we use n_features=1

- Hyperparameters to fit RNN models (*RNN_flags*). Hyperparameters to fit RNN models. It included a large number of hyperparameters related to (i) the network structure and (ii) the training algorithm. 

```
data_flags <- list(n_features = 1, n_split = 2, p_split = 0.5)

# User-defined set of hyper-parameters

RNN_flags <- data.frame(n_layers = 1, n_units = 16, dropout = 0.5, activation = "linear", weight_constraint = 1, optimizer_type = "sgd", momentum = 0.5,
                        learning_rate = 0.3, kernel_initializer = "zero", batch_size = 1024, sld_window = 1, thrsld_class = 0.5)
```

We provide an example of semi-supervised classification and we obtain the performance scores. If available it is possible to compare the anomaly detection for different types of anomalies.

```
uRNN_exmpl <- uRNN_fit(data_vars, data_label, data_flags, RNN_flags, paste(model_name, "_uRnd"))
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

sRNN_exmpl <- sRNN_fit(data_vars, data_label, data_flags, RNN_flags, paste(model_name, "_sRnd"))
sRNN_exmpl$scores

for (cc in unlist(ClassAll)) {
  RatiosClass[1,cc] <- round(sum(data_label[data_type == cc]))
  RatiosClass[2,cc] <- round(sum(sRNN_exmpl$classify[data_type == cc], na.rm = T))
  RatiosClass[3,cc] <- RatiosClass[2,cc] / RatiosClass[1,cc]
}
```


## References

Bischl, B., Richter, J., Bossek, J., Horn, D., Thomas, J., & Lang, M. (2017). mlrMBO: A modular framework for model-based optimization of expensive black-box functions. arXiv preprint arXiv:1703.03373.



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
