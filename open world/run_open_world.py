import argparse
from os import path
from map_train import train_mapper
from tester import run_tester

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, help='benchmark or predict or train')
parser.add_argument('--file', type=str, required=False, help='path to the description')
args = parser.parse_args()

if args.mode == 'benchmark':
    run_tester()
elif args.mode == 'predict':
    if path.exists(args.file):
        run_tester(single=True, file=args.file)
    else:
        exit('Description file not exists...')
elif args.mode == 'train':
    train_mapper()
else:
    exit('Mode is not supported...')