from nbconvert import PythonExporter
import nbformat
import glob
import os
import sys
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

if __name__ == "__main__":
    convert_nb(sys.argv[1])