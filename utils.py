from nbconvert import PythonExporter
import nbformat
import glob
import os
import sys
import re
import numpy as np
import gensim.downloader as api


def convert_nb(file_name):
    export_path = f'{os.getcwd()}/{os.path.split(file_name)[1][:-5]}py'
    print(export_path)
    with open(file_name, encoding="utf8") as fn:
        nb = nbformat.read(fn, as_version=4)

    exporter = PythonExporter()

    # source is a tuple of python source code
    # meta contains metadata
    source, meta = exporter.from_notebook_node(nb)

    with open(export_path, 'w+', encoding="utf8") as fh:
        fh.writelines(source)


def clean_text(text: str) -> str:
    """    
    Removes html tags, whitespaces (e.g tags), and non alphabet characters.
    Returns the lower-cased result.
    Parameters: 
        text: an input string, e.g a sentence or document
    """
    text = re.sub(r"<[^>]*>", " ", str(text))  # Removes html tags
    text = re.sub(r"\s+", " ", text)  # Removes unnecesary white space
    text = re.sub(r'[^a-zA-Z'' ]', ' ', text)  # Removes non alphabetics
    text = re.sub(r"\b[a-zA-Z]\b", "", text)  # Removes single letter words
    return re.sub(r'\s+', ' ', text).lower()  # Return lower-cased result


def embed_matrix_gen(word_index, vocab_size, embedding_dim):
    # Prepare embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    # wv = api.load('glove-twitter-100')
    wv_str = {300: "word2vec-google-news-300",
              200: "glove-wiki-gigaword-200", 100: 'glove-twitter-100', 50: 'glove-twitter-50'}
    wv = api.load(wv_str[embedding_dim])

    hits = 0
    misses = 0
    for word, i in word_index.items():
        try:
            embedding_vector = wv.get_vector(word.decode("utf-8"))
            embedding_matrix[i] = embedding_vector
            hits += 1
        except:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    return embedding_matrix


if __name__ == "__main__":
    convert_nb(sys.argv[1])
