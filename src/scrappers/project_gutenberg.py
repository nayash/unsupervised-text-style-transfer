#
# Copyright (c) 2020 - present. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import sys
import argparse
from pathlib import Path
import os
import re
import urllib.request
import zipfile

'''
--linkspath
--outpath
--lang
--filetype
--limit
'''

ROOT_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parent
INPUT_PATH = ROOT_PATH / 'inputs'
OUTPUT_PATH = ROOT_PATH / 'outputs'

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--linkspath',
                        help='path to project gutenberg links files which holds'
                             'urls to txt files')
arg_parser.add_argument('--outpath',
                        help='path to project gutenberg links files which holds'
                             'urls to txt files')
arg_parser.add_argument('--lang', help='language for which books are to be'
                                       'downloaded')
arg_parser.add_argument('--filetype', default='txt', help='file format to be'
                                                          'downloaded')
arg_parser.add_argument('--limit', default='500',
                        help='approx download size limit in MB. Once "limit" MB'
                             ' of data is downloaded, script will stop.')

args = arg_parser.parse_args()
lang = args.lang
links_dump_path = os.path.abspath(args.linkspath)
download_path = Path(os.path.abspath(args.outpath))/lang
filetype = args.filetype

link_files = [f for f in links_dump_path.glob('**/*') if f.is_file()]
for lf in link_files:
    with open(lf, 'r') as f:
        lines = f.readlines()
        links = re.findall('href=\"(.*)\"', lines)
        file_name, _ = urllib.request.urlretrieve(links[0], download_path/links[0].split('/')[-1])
        with zipfile.ZipFile('./' + file_name, 'r') as zip_ref:
            zip_ref.extractall('.')