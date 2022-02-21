import gensim.models.keyedvectors as word2vec

import heapq

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
    for line in tqdm(iterable=f.readlines(), desc='generating hints (takes 1~2 minutes to start)'):
        line = line.strip()
        if not '"' in line:
            continue
        secret = line.strip('",')
        # secret might not be in the model vocabulary if we loaded a subset
        # of the model. Skip generating hints if that's the case
        if secret not in model.vocab:
            continue
        # Calculate nearest using KeyedVectors' `most_similar`.
        # It calculates cosine similarity, which is _exactly_ what
        # this module's `similarity` does.
        # The first call to `most_similar` is s l o w: the progress
        # indicator will start moving after a minute or so.
        # This is _way_ faster than doing a nested "secret x vocab" loop.
        nearest = []
        # TODO: figure out a pythonic way to swap the tuples (map?)
        for most_similar in model.most_similar(secret, topn=1000):
            heapq.heappush(nearest, (most_similar[1], most_similar[0]))
        # The old way has `nearest` include the similarity with itself.
        # `most_similar` doesn't, so we need to add it manually.
        if len(nearest) >= 1000:
            heapq.heappushpop(nearest, (1, secret))
        else:
            heapq.heappush(nearest, (1, secret))

        nearest.sort()
        hints[secret] = nearest

import pickle
with open(b"nearest.pickle", "wb") as f:
    pickle.dump(hints, f)

