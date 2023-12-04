import argparse
import multiprocessing
import os
import sys
import time
from collections import OrderedDict

import dlib

PROCS = multiprocessing.cpu_count()
MAX_FUNC_CALLS = 100

train_path, test_path = None, None
shape_predictor_path = None
func_call_count = 0


def test_shape_predictor_params(
        treeDepth, nu, cascadeDepth, featurePoolSize, numTestSplits, 
        oversamplingAmount, oversamplingTransJitter, padding, lambdaParam):
    global func_call_count

    # grab the default options for dlib's shape predictor and then
    # set the values based on our current hyperparameter values,
    # casting to ints when appropriate
    options = dlib.shape_predictor_training_options()
    options.tree_depth = int(treeDepth)
    options.nu = nu
    options.cascade_depth = int(cascadeDepth)
    options.feature_pool_size = int(featurePoolSize)
    options.num_test_splits = int(numTestSplits)
    options.oversampling_amount = int(oversamplingAmount)
    options.oversampling_translation_jitter = oversamplingTransJitter
    options.feature_pool_region_padding = padding
    options.lambda_param = lambdaParam
    # tell dlib to be verbose when training and utilize our supplied
    # number of threads when training
    options.be_verbose = True
    options.num_threads = PROCS
     
    func_call_count += 1
    print(f"[INFO] starting training {func_call_count}...")
    print(options)
    sys.stdout.flush()

    start_time = time.time()

    # train the model using the current set of hyperparameters
    dlib.train_shape_predictor(TRAIN_PATH, SHAPE_PREDICTOR_PATH, options)

    print(f'[INFO] training took: {(time.time() - start_time) / 60:.2f} mins')

    # take the newly trained shape predictor model and evaluate it on
    # both our training and testing set
    trainingError = dlib.test_shape_predictor(TRAIN_PATH, SHAPE_PREDICTOR_PATH)
    testingError = dlib.test_shape_predictor(TEST_PATH, SHAPE_PREDICTOR_PATH)

    # display the training and testing errors for the current trial
    print(f"[INFO] train error: {trainingError}")
    print(f"[INFO] test error: {testingError}")

    # return the error on the testing set
    return testingError


def tune(args):
    global TRAIN_PATH, TEST_PATH, SHAPE_PREDICTOR_PATH
    TRAIN_PATH = args.train_xml_path
    TEST_PATH = args.test_xml_path
    SHAPE_PREDICTOR_PATH = 'temp.dat'

    params = OrderedDict([
        ("tree_depth", (2, 5, True)),  # 2^tree_depth leaves in each tree
        ("nu", (0.001, 0.2, False)),  # regularisation param
        ("cascade_depth", (4, 25, True)),  # tunes initial predictions; more cascades -> larger model
        ("feature_pool_size", (100, 1000, True)),  # no. pixels to generate features for; more -> slower model
        ("num_test_splits", (20, 100, True)),  # more splits -> accurate model (careful, this will explode train time)
        ("oversampling_amount", (1, 10, True)),  # data aug; e.g. 2 = 2x training data
        ("oversampling_translation_jitter",  (0.0, 0.3, False)),  # amount of jitter applied
        ("feature_pool_region_padding", (-0.2, 0.2, False)),
        ("lambda_param", (0.01, 0.99, False))
    ])
    lower, upper, isint = zip(*[v for v in params.values()])

    (bestParams, bestLoss) = dlib.find_min_global(
        test_shape_predictor_params,
        bound1=list(lower),
        bound2=list(upper),
        is_integer_variable=list(isint),
        num_function_calls=MAX_FUNC_CALLS
    )

    print("[INFO] optimal parameters: {}".format(bestParams))
    print("[INFO] optimal error: {}".format(bestLoss))

    # delete the temporary model file
    os.remove(SHAPE_PREDICTOR_PATH)


def train(args):
    global TRAIN_PATH, TEST_PATH, SHAPE_PREDICTOR_PATH
    TRAIN_PATH = args.train_xml_path
    TEST_PATH = args.test_xml_path
    SHAPE_PREDICTOR_PATH = 'custom_shape_predictor.dat'

    test_shape_predictor_params(
        treeDepth=args.tree_depth, 
        nu=args.nu, 
        cascadeDepth=args.cascade_depth, 
        featurePoolSize=args.feature_pool_size, 
        numTestSplits=args.num_test_splits, 
        oversamplingAmount=args.oversampling_amount, 
        oversamplingTransJitter=args.oversampling_trans_jitter, 
        padding=args.padding, 
        lambdaParam=args.lambda_param
    )


def main(args):
    f = {
        'tune': tune,
        'train': train
    }
    f[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parser.add_parser('tune')
    parser_1.add_argument('train_xml_path')
    parser_1.add_argument('test_xml_path')

    parser_2 = sub_parser.add_parser('train')
    parser_2.add_argument('train_xml_path')
    parser_2.add_argument('test_xml_path')
    parser_2.add_argument('tree_depth', type=float)
    parser_2.add_argument('nu', type=float)
    parser_2.add_argument('cascade_depth', type=float)
    parser_2.add_argument('feature_pool_size', type=float)
    parser_2.add_argument('num_test_splits', type=float)
    parser_2.add_argument('oversampling_amount', type=float)
    parser_2.add_argument('oversampling_trans_jitter', type=float)
    parser_2.add_argument('padding', type=float)
    parser_2.add_argument('lambda_param', type=float)

    main(parser.parse_args())
