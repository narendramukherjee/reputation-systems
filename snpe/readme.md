Code for SNPE based inference of hidden parameters in online review generation
==============================================================================

The main classes for running the generative model are in `snpe/simulator_class.py` and for training the inference network are in `snpe/inference_class.py`. Currently, we can:

+ Simulate online ratings with a single rho value (`SingleRhoSimulator`) or two rho values (`DoubleRhoSimulator`), one for positive and another for negative deviations of the consumer's actual experience from what they expected.

+ Simulate online ratings with two rho values and a product-level herding parameter (`HerdingSimulator`).

+ These simulations can be saved in the form of histograms or timeseries. Correspondingly, we have two types of inference networks in `HistogramInference` and `TimeSeriesInference`


