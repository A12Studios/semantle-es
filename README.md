# Semantle-es

This is a spanish version of [Semantle](https://semantle.novalis.org).

## Running locally
### One-time setup
1. Get spanish Word2vec dataset from [Spanish Billion Word Corpus and Embeddings](https://crscardellino.github.io/SBWCE/). Download the word2vec binary format to the `data` directory. _Unzip it_
1. Download the "Lista total de frecuencias" data file (`CREA_total.ZIP`) from [Corpus de Referencia del Español Actual (CREA) - Listado de frecuencias](http://corpus.rae.es/lfrecuencias.html) to the `data` directory. _Do not unzip it_
1. Create a python virtual environment: `python3 -m venv .`
1. Activate the environment: `source bin/activate`
1. Install all dependencies: `python3 -m pip install -r requirements.txt`
1. Load model into sqlite db: `python3 dump-vecs.py`. Takes ~5min in a 2.4 GHz Intel Core i5 MacBook Pro
1. Dump hints into pickle file: `python3 dump-hints.py`. Takes ~30mins in a 2.4 GHz Intel Core i5
1. Load hints into sqlite db: `python3 store-hints.py`. Fast.
1. I don't think we need/use the respelling feature of Semantle-en, so no need to run `british.py`

### Running it
1. Run web server: `python3 semantle.py`

## Attribution
Original Semantle code by [David Turner](https://novalis.org). Changes:
  - Improved `dump-hints.py` performance
  - Add progress indicator to dump and store scripts
  - Localization

Word2vec data set by Cristian Cardellino. Citation:
> Cristian Cardellino: Spanish Billion Words Corpus and Embeddings (March 2016), https://crscardellino.github.io/SBWCE/

Frequent words data set from [Corpus de referencia del español actual](http://corpus.rae.es/lfrecuencias.html). Citation:
> REAL ACADEMIA ESPAÑOLA: Banco de datos (CREA) [en línea]. Corpus de referencia del español actual. <http://www.rae.es> [2022-02-25]

