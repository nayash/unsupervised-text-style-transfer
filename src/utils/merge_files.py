"""
Utility script to merge multiple files with 'source/target' sentences into
single source/target file.

python utils/merge_files.py --srcfiles '../inputs/yelp-reviews-preprocessed/sentiment.0.all.org.txt' '../inputs/amzn-reviews-preprocessed/sentiment.0.txt' --tgtfiles '../inputs/yelp-reviews-preprocessed/sentiment.1.all.org.txt' '../inputs/amzn-reviews-preprocessed/sentiment.1.txt'
python utils/merge_files.py --srcfiles '../inputs/amzn-reviews-preprocessed/sentiment.0.txt' '../inputs/yelp-reviews-preprocessed/sentiment.0.all.txt' --tgtfiles '../inputs/amzn-reviews-preprocessed/sentiment.1.txt' '../inputs/yelp-reviews-preprocessed/sentiment.1.all.txt'
"""

import sys
sys.path.append('../src')
import argparse
from pathlib import Path
import re
import os
import time
import shutil

ROOT_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
INPUT_PATH = ROOT_PATH / 'inputs'
OUTPUT_PATH = INPUT_PATH / 'combined'
src_output_file_name = 'sentiment.0.txt'
tgt_output_file_name = 'sentiment.1.txt'
start_time = time.time()


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


argparser = argparse.ArgumentParser()
argparser.add_argument('--srcfiles',
                       default=[INPUT_PATH/'yelp-reviews-preprocessed/sentiment.0.all.org.txt',
                                INPUT_PATH/'amzn-reviews-preprocessed/sentiment.0.txt'],
                       type=argparse.FileType('r'), nargs='*')
argparser.add_argument('--tgtfiles',
                       default=[INPUT_PATH/'yelp-reviews-preprocessed/sentiment.1.all.org.txt',
                                INPUT_PATH/'amzn-reviews-preprocessed/sentiment.1.txt'],
                       type=argparse.FileType('r'), nargs='*')
argparser.add_argument('--out', default=OUTPUT_PATH, type=dir_path)


args = argparser.parse_args()
out_src_file = Path(args.out)/src_output_file_name
out_tgt_file = Path(args.out)/tgt_output_file_name

print('source and target output files:', out_src_file, out_tgt_file)

out_src_file.parent.mkdir(parents=True, exist_ok=True)
out_tgt_file.parent.mkdir(parents=True, exist_ok=True)


out_src_file.unlink(missing_ok=True)
out_tgt_file.unlink(missing_ok=True)

with open(out_src_file, 'w') as outfile:
    for file in args.srcfiles:
        print('merging', file)
        shutil.copyfileobj(file, outfile)

with open(out_tgt_file, 'w') as outfile:
    for file in args.tgtfiles:
        print('merging', file)
        shutil.copyfileobj(file, outfile)

print('source files and target files are merged into their respective output '
      'files', out_src_file, out_tgt_file, 'in', time.time()-start_time, 'secs')