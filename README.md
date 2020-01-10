# AD_ANN

Here we provide tools for using Artificial Neural Networks (ANN) for detecting anomalies in high-frequency water-quality data. Latter tool could be also applied to to numerous applications for detecting anomalies in time-series data using ANNs. To do so, we applied Recurrent Neural Networks (RNN) as a particular case of ANNs which is designed to handle sequence-dependent strings of data. In particular, the Long Short-Term Memory networks (LSTM) are a type of RNN commonly used in deep learning for time-series forescasting and are specially useful dut to include memory during the learning process. To do so we applied the keras   

Given the large number of hyper-parameters to be tested in each ANN model, we tuned ANN models using a sampling optimization methods based on **Bayesian optimization**. Here we applied the toolbox [mlrMBO](https://mlrmbo.mlr-org.com/) implemented in R, which provide excellent performance for expensive optimization scenarios for single- and multi-objective tasks, with continuous or mixed parameter spaces.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## References

Bischl, B., Richter, J., Bossek, J., Horn, D., Thomas, J., & Lang, M. (2017). mlrMBO: A modular framework for model-based optimization of expensive black-box functions. arXiv preprint arXiv:1703.03373.



## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Javier Rodriguez-Perez** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
