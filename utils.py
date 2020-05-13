from nbconvert import PythonExporter
import nbformat
import glob
import os
import sys
import re

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
    text = re.sub(r"<[^>]*>", " ", text)  # Removes html tags
    text = re.sub(r"\s+", " ", text)  # Removes unnecesary white space
    text = re.sub(r'[^a-zA-Z'' ]', ' ', text)  # Removes non alphabetics
    text = re.sub(r"\b[a-zA-Z]\b", "", text)  # Removes single letter words
    return re.sub(r'\s+', ' ', text).lower()  # Return lower-cased result


if __name__ == "__main__":
    convert_nb(sys.argv[1])
