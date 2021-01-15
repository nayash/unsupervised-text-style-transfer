"""
utility script to parse, segregate into positive and negative sentiments and
save into separate files with each sentence in a new line.

samples of amazon reviews data:
__label__2 Great CD: My lovely Pat has one of the GREAT voices of her generation. I have listened to this CD for YEARS and I still LOVE IT. When I'm in a good mood it makes me feel better. A bad mood just evaporates like sugar in the rain. This CD just oozes LIFE. Vocals are jusat STUUNNING and lyrics just kill. One of life's hidden gems. This is a desert isle CD in my book. Why she never made it big is just beyond me. Everytime I play this, no matter black, white, young, old, male, female EVERYBODY says one thing "Who was that singing ?"
__label__2 One of the best game music soundtracks - for a game I didn't really play: Despite the fact that I have only played a small portion of the game, the music I heard (plus the connection to Chrono Trigger which was great as well) led me to purchase the soundtrack, and it remains one of my favorite albums. There is an incredible mix of fun, epic, and emotional songs. Those sad and beautiful tracks I especially like, as there's not too many of those kinds of songs in my other video game soundtracks. I must admit that one of the songs (Life-A Distant Promise) has brought tears to my eyes on many occasions.My one complaint about this soundtrack is that they use guitar fretting effects in many of the songs, which I find distracting. But even if those weren't included I would still consider the collection worth it.

multiprocess writing reference: https://stackoverflow.com/questions/13446445/python-multiprocessing-safely-writing-to-a-file
"""

import sys
sys.path.append('../src')
import argparse
from pathlib import Path
import re
import os
from tqdm.auto import tqdm
from utils import clean_text_yelp
import numpy as np
import multiprocessing as mp
import time
import fcntl
from nltk.tokenize import sent_tokenize, word_tokenize

ROOT_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
file_name = 'amazon_reviews_raw.txt'
INPUT_PATH = ROOT_PATH / ('./inputs/'+file_name)
OUTPUT_PATH = ROOT_PATH / './inputs/amzn-reviews-preprocessed'
src_output_file_name = 'sentiment.0.txt'
tgt_output_file_name = 'sentiment.1.txt'
start_time = time.time()

argparser = argparse.ArgumentParser()

argparser.add_argument('--infile', default=INPUT_PATH, type=argparse.FileType('r'))
argparser.add_argument('--outsrc', default=OUTPUT_PATH/src_output_file_name,
                       type=argparse.FileType('r'))
argparser.add_argument('--outtgt', default=OUTPUT_PATH/tgt_output_file_name,
                       type=argparse.FileType('r'))
argparser.add_argument('--parts', default=10000)
argparser.add_argument('--append', default=False)


args = argparser.parse_args()
input_file = Path(args.infile)
out_src_file = Path(args.outsrc)
out_tgt_file = Path(args.outtgt)
parts = args.parts

print('source and target output files:', out_src_file, out_tgt_file)

out_src_file.parent.mkdir(parents=True, exist_ok=True)
out_tgt_file.parent.mkdir(parents=True, exist_ok=True)

if not args.append:
    out_src_file.unlink(missing_ok=True)
    out_tgt_file.unlink(missing_ok=True)

reg_pattern_stars = r'__label__(\d)'
matches_text = []
matches_stars = []
with open(input_file, 'r') as infile:
    count = 0
    for content in infile:
        matches_stars.extend(re.findall(reg_pattern_stars, content))
        matches_text.append(re.sub(reg_pattern_stars, '', content).strip())
        count += 1

assert len(matches_text) == len(matches_stars), 'size of texts and review ' \
                                                 'stars parsed don\'t match'

print('number of samples', len(matches_text))

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
        if float(star) == 1:
            src_sents_temp.extend(sent_tokenize(text))
        else:
            tgt_sents_temp.extend(sent_tokenize(text))

    q_src.put(src_sents_temp.copy())
    src_sents_temp.clear()

    q_tgt.put(tgt_sents_temp.copy())
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
print('writing to file with multiple processes. Batch size=', offset)
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
