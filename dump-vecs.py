import time
import sqlite3
import gensim.models.keyedvectors as word2vec

from tqdm import tqdm

t_word2vec = time.process_time()
print("loading word2vec file...")
model = word2vec.KeyedVectors.load_word2vec_format("data/SBW-vectors-300-min5.bin", binary=True)
print(f'done in {time.process_time() - t_word2vec} seconds')

print("creating word2vec table...")
con = sqlite3.connect('word2vec.db')
cur = con.cursor()
cur.execute("""create table if not exists word2vec (word text, vec blob)""")
con.commit()
cur.execute("""create unique index if not exists word2vec_word on word2vec (word)""")
con.commit()
print("done")

total_words = len(model.vocab)
t_load = time.process_time()
for i, word in tqdm(enumerate(model.vocab), desc='inserting model values into sqlite database', total=total_words):
    if i % 1111 == 0:
        con.commit()
    vec = model[word].tostring()
    cur.execute("insert into word2vec values(?,?)", (word,vec))

con.commit()
t_total = time.process_time() - t_load
print(f"done in {t_total} secs, {total_words/t_total} words/sec")
