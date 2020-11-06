import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from pandas import DataFrame
from time import time
from multihopr.longformerUtils import LongformerTensorizer, LongformerEncoder
from multihopr.hotpotqaIOUtils import loadWikiData

full_wiki_path = '../data/hotpotqa/fullwiki_qa'
abs_full_wiki_path = os.path.abspath(full_wiki_path)