from nltk.tokenize import sent_tokenize, word_tokenize
import argparse
from pathlib import Path
import re
import os
from tqdm.auto import tqdm
from utils import clean_text_yelp
import numpy as np
import sys
import multiprocessing as mp

"""
https://www.kaggle.com/yelp-dataset/yelp-dataset?select=yelp_academic_dataset_review.json
"""

ROOT_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parent
file_name = 'yelp_academic_dataset_review.json'
INPUT_PATH = ROOT_PATH / ('./inputs/'+file_name)
OUTPUT_PATH = ROOT_PATH / './inputs/yelp-reviews-preprocessed'
src_output_file_name = 'sentiment.0.all.txt'
tgt_output_file_name = 'sentiment.1.all.txt'

argparser = argparse.ArgumentParser()

argparser.add_argument('--infile', default=INPUT_PATH)
argparser.add_argument('--outsrc', default=OUTPUT_PATH/src_output_file_name)
argparser.add_argument('--outtgt', default=OUTPUT_PATH/tgt_output_file_name)


args = argparser.parse_args()
input_file = args.infile
out_src_file = args.outsrc
out_tgt_file = args.outtgt

print(out_src_file, out_tgt_file)
"""
sample json :
"stars":2.0,"useful":5,"funny":0,"cool":0,"text":"As someone who has worked with many museums, 
I was eager to visit this gallery on my most recent trip to Las Vegas. When I 
saw they would be showing infamous eggs of the House of Faberge from the Virginia
 Museum of Fine Arts (VMFA), I knew I had to go!\n\nTucked away near the 
 gelateria and the garden, the Gallery is pretty much hidden from view. It's ","date":"2015-04-15 05:21:16"
"""

reg_pattern_text = r'\"text\"\:\"(.*)\"\,'
reg_pattern_stars = r'\"stars\"\:(\d+\.?\d*)\,'
with open(input_file, 'r') as infile:
    content = infile.read()
    matches_text = re.findall(reg_pattern_text, content)
    matches_stars = np.array(re.findall(reg_pattern_stars, content)).astype(float)

del content
assert len(matches_text) == len(matches_stars), 'size of texts and review ' \
                                                 'stars parsed don\'t match'


def append_file(file_path, sent_list):
    with open(file_path, 'a') as file:
        file.write('\n'.join(sent_list))
    

src_sents = []
tgt_sents = []
for text, star in tqdm(zip(matches_text, matches_stars)):
    if star <= 3:
        src_sents.extend(sent_tokenize(clean_text_yelp(text)))
    else:
        tgt_sents.extend(sent_tokenize(clean_text_yelp(text)))

    if len(src_sents) >= 100000:
        append_file(out_src_file, src_sents)
        src_sents.clear()

    if len(tgt_sents) >= 100000:
        append_file(out_tgt_file, tgt_sents)
        tgt_sents.clear()

# flush out remaining sentences
append_file(out_src_file, src_sents)
append_file(out_tgt_file, tgt_sents)

print('appended src and tgt files with new sentences')
