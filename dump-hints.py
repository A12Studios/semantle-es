import io
import pickle
import re
import time
from zipfile import ZipFile

import gensim.models.keyedvectors as word2vec
from tqdm import tqdm
import secret_words

SIMPLE_WORD_REGEX = re.compile("^[a-z]*")
DISALLOWED_ENDINGS = ['rlo', 'rlos', 'rla', 'rlas', 'rle', 'rles', 'ando', 'iendo', 'ía', 'ían', 'ías', 'ísima', 'ísimo', 'ará', 'aré' \
        'arse', 'erse', 'irse', 'ió', 'ié', 'etí', 'ola', 'ole', 'ote', 'ose', 'ándolo', 'ándola', 'ándole']
TOP_N = 1000
"""
Should we exclude this word from the common word list?
"""
def is_disallowed_word(word):
    return any(map(lambda ending: word.endswith(ending), DISALLOWED_ENDINGS)) or \
            len(word) < 4

"""
Reads the common words list.
We filter the hints by this list to reduce "noise"/improve gradient in the hints.
"""
def get_common_words_set():
    result = set()
    with ZipFile("data/CREA_total.ZIP", 'r') as crea_zip:
        with crea_zip.open("CREA_total.TXT") as crea_txt:
            # skip header
            next(crea_txt)
            for line in tqdm(io.TextIOWrapper(crea_txt, 'iso8859'), 'reading common words file'):
                # cols: 0: word number, 1: word, 2: absolute freq, 3: normalized freq.
                word = line.strip().split('\t')[1].strip()
                if SIMPLE_WORD_REGEX.match(word) and not is_disallowed_word(word):
                    result.add(word)
    return result

common_words_set = get_common_words_set()
# Set to None to read all words in the model. Useful to set a low number to test script.
WORD_LIMIT = None
t_word2vec = time.process_time()
print("loading word2vec file...")
model = word2vec.KeyedVectors.load_word2vec_format("data/SBW-vectors-300-min5.bin", binary=True, limit=WORD_LIMIT)
print(f'done in {time.process_time() - t_word2vec} seconds')

hints = {}
for secret in tqdm(secret_words.read(), desc='generating hints (takes 1~2 minutes to start)'):
    # secret might not be in the model vocabulary if we loaded a subset
    # of the model. Skip generating hints if that's the case
    if secret not in model.vocab:
        continue
    # Calculate nearest using KeyedVectors' `most_similar`.
    # It calculates cosine similarity, which is  what
    # the original Semantle does.
    # The first call to `most_similar` is s l o w: the progress
    # indicator will start moving after a minute or so.
    # This is _way_ faster than doing a nested "secret x vocab" loop.
    # Slice up to TOP_N -1 to leave room for the secret word.
    most_similar = [it for it in model.most_similar(secret, topn=100 * TOP_N) if it[0] in common_words_set][0:TOP_N - 1]
    # Nearest must include the secret. `most_similar` doesn't, so we need to add it manually.
    most_similar.extend([(secret, 1)])
    if len(most_similar) < TOP_N:
        raise RuntimeError(f'most_similar has too few common words: {len(most_similar)} after filtering, needs {TOP_N}')
    # store-hints.py expects a (score, word) tuple
    nearest = [(item[1], item[0]) for item in most_similar]
    # store-hints.py relies on nearest's order to get the closest, 10th and 1000th nearby element.
    nearest.sort()
    hints[secret] = nearest

with open(b"nearest.pickle", "wb") as pickled:
    pickle.dump(hints, pickled)
