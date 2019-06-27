from model_run import model_train, model_test
from simulate_data import generate_ny_flow
from model import AnoModel

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='experiment settings.')

parser.add_argument('--weight', type=int, default=5, help='anomaly weight')
parser.add_argument('--gen_data', type=bool, default=False, help='if generate data')

if __name__ == "__main__":
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["weight"] = str(args.weight)

    if args.gen_data:
        generate_ny_flow(args.weight)

    print("create model.")
    model = AnoModel(tf_units=[128, 16],
                     sf_units=[],
                     st_units=[128, 256, 64],
                     out_dim=4,
                     sf_dim=16,
                     tf_dim=36,
                     batch_size=128)

    model_train(model)
    model_test(model)