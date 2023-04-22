import time
from data import *
from model import *


if __name__ == '__main__':
    start = time.time()

    train_data = prep_data('data/weatherAUS_train.csv')
    train(train_data)

    test_data = prep_data('data/weatherAUS_test.csv')
    evaluate(test_data)

    prod(5)

    end = time.time()

    print(f'Finished in: {(end - start)/60} min')