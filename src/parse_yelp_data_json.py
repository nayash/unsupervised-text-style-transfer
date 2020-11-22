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
import time
import fcntl

"""
https://www.kaggle.com/yelp-dataset/yelp-dataset?select=yelp_academic_dataset_review.json
"""

ROOT_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parent
file_name = 'yelp_academic_dataset_review.json'
INPUT_PATH = ROOT_PATH / ('./inputs/'+file_name)
OUTPUT_PATH = ROOT_PATH / './inputs/yelp-reviews-preprocessed'
src_output_file_name = 'sentiment.0.all_.txt'
tgt_output_file_name = 'sentiment.1.all_.txt'
start_time = time.time()

argparser = argparse.ArgumentParser()

argparser.add_argument('--infile', default=INPUT_PATH)
argparser.add_argument('--outsrc', default=OUTPUT_PATH/src_output_file_name)
argparser.add_argument('--outtgt', default=OUTPUT_PATH/tgt_output_file_name)
argparser.add_argument('--parts', default=10000)


args = argparser.parse_args()
input_file = args.infile
out_src_file = args.outsrc
out_tgt_file = args.outtgt
parts = args.parts

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
matches_text = []
matches_stars = []
with open(input_file, 'r') as infile:
    count = 0
    for content in infile:
        matches_text.extend(re.findall(reg_pattern_text, content))
        matches_stars.extend(re.findall(reg_pattern_stars, content))
        count += 1

# print(matches_text, matches_stars)

assert len(matches_text) == len(matches_stars), 'size of texts and review ' \
                                                 'stars parsed don\'t match'


src_sents = []
tgt_sents = []


def append_file(file_path, sent_list):
    with open(file_path, 'a+') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.write('\n'.join(sent_list))
        fcntl.flock(file, fcntl.LOCK_UN)


def append_list(texts, stars):
    src_sents_temp = []
    tgt_sents_temp = []
    for text, star in zip(texts, stars):
        text = text.replace('\\n', ' ')  # without this sentence split is worse
        # print(text, star)
        if float(star) <= 3:
            src_sents_temp.extend(sent_tokenize(text))
        else:
            tgt_sents_temp.extend(sent_tokenize(text))
    # print('list lens', len(src_sents_temp), len(tgt_sents_temp))
    if len(src_sents_temp) >= 1000:
        q_src.put(src_sents_temp.copy())
        # print('sent q_src', len(src_sents_temp))
        src_sents_temp.clear()

    if len(tgt_sents_temp) >= 1000:
        q_tgt.put(tgt_sents_temp.copy())
        # print('sent q_tgt', len(tgt_sents_temp))
        tgt_sents_temp.clear()


def listener_src(q, file_path):
    """listens for messages on the q, writes to file. """
    with open(file_path, 'a+') as file:
        while 1:
            m = q.get()
            # print('received q_src', len(m))
            if m == 'kill':
                file.flush()
                break
            file.write('\n'.join(m))


def listener_tgt(q, file_path):
    """listens for messages on the q, writes to file. """
    with open(file_path, 'a+') as file:
        while 1:
            m = q.get()
            if m == 'kill':
                file.flush()
                break
            file.write('\n'.join(m))


manager = mp.Manager()
q_src = manager.Queue()
q_tgt = manager.Queue()
pool = mp.Pool(mp.cpu_count() + 2)

watcher_src = pool.apply_async(listener_src, (q_src, out_src_file))
watcher_tgt = pool.apply_async(listener_tgt, (q_tgt, out_tgt_file))

results = []
# workers = mp.cpu_count()
offset = len(matches_text) // parts
print('writing to file with multiple process...')
for i in range(parts):
    s = i*offset
    e = s+offset-1 if i < parts-1 else -2
    results.append(pool.apply_async(append_list, args=[matches_text[s:e+1],
                                                       matches_stars[s:e+1]]))

[res.get() for res in results]
[res.wait() for res in results]

# flush out remaining sentences
# append_file(out_src_file, src_sents)
# append_file(out_tgt_file, tgt_sents)

q_src.put('kill')
q_tgt.put('kill')
pool.close()
pool.join()

print('appended src and tgt files with new sentences')
print('finished in ', (time.time()-start_time), 'secs')
