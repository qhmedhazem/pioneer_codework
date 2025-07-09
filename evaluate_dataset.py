import argparse
import numpy as np

from pose_estimator.estimator import Estimator
from datasets.euroc_dataset import DataType, EuRoCDataset
from config.vio_parameters import Parameters


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the framework by YAML config file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./data/euroc_mh_01_easy_evaluation.yaml",
        help="Directory to the YAML file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    parameters = Parameters(args.config)
    estimator = Estimator(parameters=parameters)
    dataset = EuRoCDataset(parameters.input_path)

    print(parameters.input_path)

    for type, frame in dataset:
        if type == DataType.CAMERA:
            estimator.img_callback(frame)
        elif type == DataType.IMU:
            estimator.imu_callback(frame)


if __name__ == "__main__":
    main()
