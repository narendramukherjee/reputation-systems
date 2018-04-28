# simple test file for the classes defined in models.py

#from models_single_product import *
from models_single_product_mcmc_replaced import *
import settings
import matplotlib.pyplot as plt


# Goal: set variable outside option values - here, we fix threshold = 1 for now

# Note:

value_of_outside_option = [0,2,4,6,8]
repeats = 10
avg_reviews_list = []

for outside_option in value_of_outside_option:
    avg_reviews_at_this_outside_op = []
    for repeat in range(repeats):
        dynamics = market(settings.params)
        dynamics.params['value_of_outside_option'] = outside_option
        data, avg_reviews, perceived_qual = dynamics.generateTimeseries()
        avg_reviews_at_this_outside_op.append(avg_reviews[-1])
        print(repeat)
    avg_reviews_list.append([np.mean(avg_reviews_at_this_outside_op), np.std(avg_reviews_at_this_outside_op)])
    print(outside_option)

print(avg_reviews_list)
avg_reviews_list = np.array(avg_reviews_list)
#plt.scatter(value_of_outside_option,avg_reviews_list[:, 0])
#plt.ion()
plt.figure(1)
plt.errorbar(value_of_outside_option,avg_reviews_list[:, 0], yerr=2*avg_reviews_list[:,1]/np.sqrt(10))
plt.xlabel("Outside Option", fontsize = 14)
plt.ylabel("Final average rating (averaged across 10 samples)", fontsize = 12)
plt.savefig("Fig1{}.png".format(settings.tracked_product_ID),bbox_inches = "tight")

# Evolution of avg ratings over time

avg_rev_rolling = []
for outside_option in value_of_outside_option:
    dynamics = market(settings.params)
    dynamics.params['value_of_outside_option'] = outside_option
    data, avg_reviews, perceived_qual = dynamics.generateTimeseries()
    avg_rev_rolling.append(avg_reviews)

avg_rev_rolling = np.array(avg_rev_rolling)
plt.figure(2)
plt.xlabel("Time", fontsize = 14)
plt.ylabel("Average rating at each timepoint (single trial)", fontsize = 14)

for i in range(avg_rev_rolling.shape[0]):
    plt.plot(avg_rev_rolling[i,:], label = "{}".format(value_of_outside_option[i]))
plt.legend()
plt.savefig("Fig2{}.png".format(settings.tracked_product_ID), bbox_inches = "tight")


# How do fraction of 1,2,3 vs 4,5 stars change over time

frac_diff = []
for outside_option in value_of_outside_option:
    frac_diff_for_this_outside_op = []

    for repeat in range(repeats):
        dynamics = market(settings.params)
        dynamics.params['value_of_outside_option'] = outside_option
        data, avg_reviews, perceived_qual = dynamics.generateTimeseries()
        frac_diff_for_this_outside_op.append(((data[4] + data[3]) - (data[2] + data[1] + data[0]))/(np.arange(40) + 1))
        print(repeat)
    frac_diff.append(frac_diff_for_this_outside_op)
frac_diff = np.array(frac_diff)
plt.figure(3)
plt.xlabel("Time", fontsize = 14)
plt.ylabel("Difference in frac. high reviews (4,5)" + "\n" + "and low reviews (1,2,3) over time", fontsize = 14)
for i in range(frac_diff.shape[0]):
    plt.errorbar(np.arange(40),np.mean(frac_diff[i,:,:], axis=0), label = "{}".format(value_of_outside_option[i]),yerr=2*np.std(frac_diff[i,:,:], axis=0)/np.sqrt(frac_diff.shape[1]))
plt.legend()
plt.savefig("Fig3{}.png".format(settings.tracked_product_ID), bbox_inches = "tight")












#######################
##### CODE SANDBOX ###


#dynamics = market(settings.params)
#print(dynamics.params)


#print('A Torch Sample')
#label_of_data, data = dynamics.genTorchSample()
#data, avg_reviews, perceived_qual = dynamics.generateTimeseries()
#print(label_of_data)
#testing = dynamics.testing_what
#print(dynamics.params)

#print(data,avg_reviews[-1],perceived_qual)


# print('A Torch Dataset with two samples')
# dataset = dynamics.genTorchDataset(2)
# print(dataset)





