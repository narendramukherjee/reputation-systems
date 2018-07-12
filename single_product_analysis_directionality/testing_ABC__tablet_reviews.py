# Processes the observed tomes series of reviews for one product and feeds the processed data to ABC for
# inference

from inference_ABC import *
from models_single_product import *
import settings
import os

RD.seed()

if __name__ == '__main__':

    gen_model = ABC_GenerativeModel(params=settings.params, conditioning = False, direction = None)
    # conditioning = False, direction = None
    # conditioning=True, direction='above'

    current_working_directory = os.getcwd()

    observed_timeseries = pd.read_csv(current_working_directory + '/data/' + settings.tracked_product_ID +
                                      '_time_series.txt', sep='\t')

    data = list(observed_timeseries['Rating'])
    # all_ratings = list(raw_timeseries['Rating'])
    # processed_timeseries = gen_model.process_raw_timeseries()

    print(len(data))

    (posterior, distances,
     accepted_count, trial_count,
     epsilon) = basic_abc(gen_model, data, epsilons=[0.05], min_samples=20)

    print('posterior', posterior)
    print('distances', distances)
    print('accepted_count', accepted_count)
    print('trial_count', trial_count)
    print('epsilon', epsilon)
    print(np.mean(posterior))