import sqlite3

import pickle

from tqdm import tqdm

con = sqlite3.connect('word2vec.db')
cur = con.cursor()
cur.execute("""drop table if exists similarity_range;""")
cur.execute("""drop table if exists nearby;""")
con.commit()

cur.execute("""create table if not exists nearby (word text, neighbor text, similarity float, percentile integer)""")
con.commit()

cur.execute("""create unique index if not exists nearby_words on nearby (word, neighbor)""")
con.commit()

cur.execute("""create table if not exists similarity_range (word text, top float, top10 float, rest float)""")

cur.execute("""create unique index if not exists similarity_range_word on similarity_range (word)""")
con.commit()


with open(b"nearest.pickle", "rb") as f:
    nearest = pickle.load(f)

t = tqdm(desc='inserting hints into db', total=len(nearest))
for i, (secret, neighbors) in enumerate(nearest.items()):
    if i % 1111 == 0:
        con.commit()
    for idx, (score, neighbor) in enumerate(neighbors):
        con.execute ("insert into nearby (word, neighbor, similarity, percentile) values (?, ?, ?, ?)", (secret, neighbor, "%s" % score, (1 + idx)))
    try:
        top = neighbors[-2][0]
        top10 = neighbors[-12][0]
        rest = neighbors[0][0]
        con.execute ("insert into similarity_range (word, top, top10, rest) values (?, ?, ?, ?)", (secret, "%s" % top, "%s" % top10, "%s" % rest))
    except IndexError:
        print(f'{secret} does not have enough neighbors: {len(neighbors)}')
    t.update()

con.commit()

# Validate hints are sane
max_percentile_query = cur.execute("select max(percentile) from (select word, max(percentile) as percentile from nearby group by word)")
max_percentile = list(max_percentile_query.fetchone())[0]
if max_percentile != 1000:
    raise RuntimeError(f'maximum percentile is {max_percentile}, should be 1000')
con.close()
