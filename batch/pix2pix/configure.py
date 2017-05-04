import sys
import os


def get_root_path():
    spliter = "/"
    if os.name == "nt": spliter = "\\"
    cells = os.path.abspath(os.path.dirname(__file__)).split(spliter)
    cells.pop()
    cells.pop()
    return "/".join(cells)

sys.path.append(get_root_path() + "/include/")
