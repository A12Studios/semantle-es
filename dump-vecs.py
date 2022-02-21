import time
import gensim.models.keyedvectors as word2vec

import sqlite3
from tqdm import tqdm

t_word2vec = time.process_time()
print("loading word2vec file...")
model = word2vec.KeyedVectors.load_word2vec_format("../GoogleNews-vectors-negative300.bin", binary=True)
print(f'done in {time.process_time() - t_word2vec} seconds')

con = sqlite3.connect('word2vec.db')
cur = con.cursor()
cur.execute("""create table if not exists word2vec (word text, vec blob)""")
con.commit()
cur.execute("""create unique index if not exists word2vec_word on word2vec (word)""");
con.commit()

import pdb;pdb.set_trace()

total_words = len(model.vocab)
t_load = time.process_time()
t = tqdm(total=total_words)
for i, word in enumerate(model.vocab):
    if (i % 1111 == 0):
        con.commit()
    vec = model[word].tostring()
    cur.execute("insert into word2vec values(?,?)", (word,vec))
    t.update()

con.commit()
t.close()
t_total = time.process_time() - t_load
print(f"done in {t_total} secs, {total_words/t_total} words/sec")
