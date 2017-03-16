import os
import sys


def get_root_path():
    cells = os.path.abspath(os.path.dirname(__file__)).split("\\")
    cells.pop()
    return "/".join(cells)

def get_data_path():
    return get_root_path() + "/data/"