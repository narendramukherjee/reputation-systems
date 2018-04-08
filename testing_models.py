# simple test file for the classes defined in models.py


from model_multiple_products import *
import matplotlib.pyplot as plt


if __name__ == '__main__':




    dynamics = market()
    dynamics.params['total_number_of_reviews'] = 20
    print(dynamics.generateTimeseries())
    print(dynamics.regret)
    print(dynamics.disappointment)
    print(dynamics.best_choice_made)


    fraction_of_best_choices = []
    print(range(len(dynamics.best_choice_made)))

    for i in range(len(dynamics.best_choice_made)):
        # print(i)
        # print(dynamics.best_choice_made[0:i])
        fraction_of_best_choices += [sum(dynamics.best_choice_made[:i])/(float(i)+1)]

    plt.plot(fraction_of_best_choices)
    plt.ylabel('fraction of best choices')
    plt.xlabel('time (consumer ordinals)')
    plt.show()

    plt.plot(dynamics.disappointment)
    plt.ylabel('disappointment')
    plt.xlabel('time (consumer ordinals)')
    plt.show()

    plt.plot(dynamics.regret)
    plt.ylabel('regret')
    plt.xlabel('time (consumer ordinals)')
    plt.show()