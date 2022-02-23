import gensim.models.keyedvectors as word2vec

import math

import heapq

from numpy import dot
from numpy.linalg import norm

from tqdm import tqdm
import re
import time

import code, traceback, signal

def debug(sig, frame):
    """Interrupt running process, and provide a python prompt for
    interactive debugging."""
    d={'_frame':frame}         # Allow access to frame object.
    d.update(frame.f_globals)  # Unless shadowed by global
    d.update(frame.f_locals)

    i = code.InteractiveConsole(d)
    message  = "Signal received : entering python shell.\nTraceback:\n"
    message += ''.join(traceback.format_stack(frame))
    i.interact(message)

signal.signal(signal.SIGUSR1, debug)  # Register handler

# Set to None to read all words in the model. Useful to set a low number to test script.
word_limit = None
t_word2vec = time.process_time()
print("loading word2vec file...")
model = word2vec.KeyedVectors.load_word2vec_format("../GoogleNews-vectors-negative300.bin", binary=True, limit=word_limit)
print(f'done in {time.process_time() - t_word2vec} seconds')

def mag(v):
    return math.sqrt(sum(x * x for x in v))

def similarity(v1, v2):
    return abs(sum(a * b for a, b in zip(v1, v2)) / (mag(v1)*mag(v2)))

def similarity(a, b):
    return abs(dot(a, b)/(norm(a)*norm(b)))

# synonyms = {}

# with open("moby/words.txt") as moby:
#     for line in moby.readlines():
#         line = line.strip()
#         words = line.split(",")
#         word = words[0]
#         synonyms[word] = set(words)

print("loaded moby...")

t = tqdm(desc='loading words_alpha.txt')
allowable_words = set()
with open("words_alpha.txt") as walpha:
    for line in walpha.readlines():
        allowable_words.add(line.strip())
        t.update()
t.close()

simple_word = re.compile("^[a-z]*")
words = []
for word in tqdm(iterable=model.vocab, desc='loading words from model'):
#    if simple_word.match(word) and word in allowable_words:
    words.append(word)

hints = {}
with open("static/assets/js/secretWords.js") as f:
    for line in tqdm(iterable=f.readlines(), desc='generating hints'):
        line = line.strip()
        if not '"' in line:
            continue
        secret = line.strip('",')
        # secret might not be in the model vocabulary if we loaded a subset
        # of the model. Skip generating hints if that's the case
        if secret not in model.vocab:
            continue
        target_vec = model[secret]

        start = time.time()
#        syns = synonyms.get(secret) or []
        nearest = []
        for word in tqdm(iterable=words, desc='looking for hints', leave=False, position=1):
#            if word in syns:
#                continue
#            if secret in (synonyms.get(word) or []):
#                # yow, asymmetrical!
#                continue
#            if word in secret or secret in word:
#                continue
            vec = model[word]
            s = similarity(vec, target_vec)
            if len(nearest) > 1000:
                heapq.heappushpop(nearest, (s, word))
            else:
                heapq.heappush(nearest, (s, word))
        nearest.sort()
        hints[secret] = nearest

import pickle
with open(b"nearest.pickle", "wb") as f:
    pickle.dump(hints, f)

