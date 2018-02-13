# Params and other settings are set here
# Settings are for the generative model as well as the inference engine


# The generative model settings

params = {'product_tracked': 0,
              'prices': [10, 11, 12, 13],
              'product_features': [1, 2, 3, 4],
              'total_number_of_reviews': 20,
              #'input_type': 'kurtosis',
              # 'input_type': 'averages',
              'input_type': 'histograms',
              # 'input_histograms_are_normalized': True,
              'number_of_rating_levels': 5,
              }


# The inference engine settings

if params['input_type'] == 'histograms':
    number_of_features = 5# each point in the input time series is a oistogram comprised of
# five number for each of the five review levels (1,2,3,4,5)
    assert number_of_features == 5, 'wrong number of features'
else:
    number_of_features = 1# each point in the input time series is an average review
    assert number_of_features == 1, 'wrong number of features'

n_hidden = 16  # number of units in each layer of the recurrent unit

NUM_LAYERS = 3  # number of layers in each recurrent unit

OUTPUT_SIZE = 2 # output of the fully connected linear module at the end before the softmax
