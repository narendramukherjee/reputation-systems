from models_single_product_mcmc_replacedv2 import *
import settings
import matplotlib.pyplot as plt
import seaborn as sns

repeats = 50
d1 = np.zeros([repeats,5])

for repeat in range(repeats):
    dynamics = market(settings.params)
    #dynamics.params['value_of_outside_option'] = 5.0
    data, avg_reviews, perceived_qual = dynamics.generateTimeseries()
    print(dynamics.params["rate_decision_threshold_above"])
    d1[repeat] = np.array(data.tail(1))

frame1 = pd.DataFrame(d1, columns = [str(i+1) for i in range(5)])
figure = plt.figure()
sns.barplot(data = frame1)
figure.savefig("threshold_one.png", bbox_inches = "tight")

repeats = 50
d0 = np.zeros([repeats,5])

for repeat in range(repeats):
    dynamics = market(settings.params)
    #dynamics.params['value_of_outside_option'] = 5.0
    dynamics.params['rate_decision_threshold_above'] = 0.0
    dynamics.params['rate_decision_threshold_below'] = dynamics.params['rate_decision_threshold_above']
    print(dynamics.params["rate_decision_threshold_above"])
    data, avg_reviews, perceived_qual = dynamics.generateTimeseries()
    d0[repeat] = np.array(data.tail(1))



frame0 = pd.DataFrame(d0, columns = [str(i+1) for i in range(5)])
figure = plt.figure()
sns.barplot(data = frame0)
figure.savefig("threshold_zero.png", bbox_inches = "tight")


