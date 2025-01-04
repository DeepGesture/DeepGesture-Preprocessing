import argparse
import yaml
from pprint import pprint
from easydict import EasyDict


def process_bvh_files(source_path, target_path, args):
    pass


def process_zeggs_dataset(source_path, target_path, args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepGesturePreprocessing')
    parser.add_argument('--config', default='./configs/DeepGesturePreprocessing.yml')
    parser.add_argument('--bvh_path', type=str, default='../data/bvh')
    parser.add_argument('--bvh_target_path', type=str, default='../output/bvh')
    parser.add_argument('--wav_path', type=str, default='../data/wav')
    parser.add_argument('--wav_target_path', type=str, default='../output/wav')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    # for k, v in vars(args).items():
    #     config[k] = v
    pprint(config)

    config = EasyDict(config)

    process_bvh_files(args.bvh_path, args.bvh_target_path, config)
    # process_zeggs_dataset(source_path, target_path, config)
    print("Finish!")
