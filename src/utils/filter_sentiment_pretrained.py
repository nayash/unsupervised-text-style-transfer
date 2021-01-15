"""
utility script to read sentences from one or more files and classify them as
positive or negative sentiments and save in separate files. The input files with
sentences should have one sentence per line.

Each review has a mix of positive, negative and neutral sentences. This script
can be used to segregate sentences based on it's own polarity and not the
overall rating of the review. The polarity of each sentence is determined by a
pre-trained sentiment classifier. The output of this script can then be used for
 any downstream style transfer task.

This script was used on outputs of "yelp_parser.py" and "amazon_reviews.parser.py"
scripts.

python utils/filter_sentiment_pretrained.py --infiles '../inputs/yelp-reviews-preprocessed/sentiment.0.all.org.txt' '../inputs/yelp-reviews-preprocessed/sentiment.1.all.org.txt' '../inputs/amzn-reviews-preprocessed/sentiment.0.txt' '../inputs/amzn-reviews-preprocessed/sentiment.1.txt'
"""

import sys
sys.path.append('../src')
import argparse
from pathlib import Path
import re
import os
import multiprocessing as mp
import time
import fcntl
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
# torch.multiprocessing.set_start_method('spawn', force=True)
# import flair
from textblob import TextBlob
from utils import clean_text_yelp

ROOT_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
INPUT_PATH = ROOT_PATH / 'inputs'
OUTPUT_PATH = ROOT_PATH / 'inputs/sentiment_classified'
src_output_file_name = 'sentiment.0.txt'
tgt_output_file_name = 'sentiment.1.txt'
start_time = time.time()


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

argparser = argparse.ArgumentParser()

argparser.add_argument('--infiles',
                       default=[INPUT_PATH/'yelp-reviews-preprocessed/sentiment.0.all.org.txt',
                                INPUT_PATH/'yelp-reviews-preprocessed/sentiment.1.all.org.txt',
                                INPUT_PATH/'amzn-reviews-preprocessed/sentiment.0.txt',
                                INPUT_PATH/'amzn-reviews-preprocessed/sentiment.1.txt'],
                       type=argparse.FileType('r'), nargs='*')
argparser.add_argument('--out', default=OUTPUT_PATH, type=dir_path)
argparser.add_argument('--append', default=False)
argparser.add_argument('--parts', default=1000)


args = argparser.parse_args()
input_files = args.infiles
out_src_file = Path(args.out)/src_output_file_name
out_tgt_file = Path(args.out)/tgt_output_file_name
parts = args.parts

print('source and target output files:', out_src_file, out_tgt_file)

out_src_file.parent.mkdir(parents=True, exist_ok=True)
out_tgt_file.parent.mkdir(parents=True, exist_ok=True)

if not args.append:
    out_src_file.unlink(missing_ok=True)
    out_tgt_file.unlink(missing_ok=True)

# flair_sentiment = flair.models.TextClassifier.load('en-sentiment')


def append_file(file_path, sent_list):
    with open(file_path, 'a+') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.write('\n'.join(sent_list))
        fcntl.flock(file, fcntl.LOCK_UN)


def get_sentiment_textblob(text):
    # TextBlob's performance is not good on sentiment analysis
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return 1
    elif polarity < -0.05:
        return -1
    else:
        return 0


# def get_sentiment_flair(text):
#     # TODO flair model loading on multiple processes is not working.
#     #  Needs more work!
#     s = flair.data.Sentence(text)
#     flair_sentiment.predict(s)
#     s.labels[0]


def append_list(texts):
    src_sents_temp = []
    tgt_sents_temp = []
    for text in texts:
        text = text.replace('\\n', ' ')  # without this sentence split is worse
        sents = sent_tokenize(text)
        for sent in sents:
            sent = clean_text_yelp(sent)
            if not sent:
                continue
            sentiment = get_sentiment_textblob(sent)
            if sentiment == -1:
                src_sents_temp.append(sent)
            elif sentiment == 1:
                tgt_sents_temp.append(sent)

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

for file in input_files:
    print('processing file:', file)
    texts = file.readlines()
    results = []
    # workers = mp.cpu_count()
    offset = len(texts) // parts
    print('writing to file with multiple processes. Batch size=', offset)
    for i in range(parts):
        s = i*offset
        e = s+offset-1 if i < parts-1 else -2
        results.append(pool.apply_async(append_list, args=[texts[s:e+1]]))

    file.close()

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


