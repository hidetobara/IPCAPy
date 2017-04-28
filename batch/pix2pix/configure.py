import sys
import os


def get_root_path():
    cells = os.path.abspath(os.path.dirname(__file__)).split("\\")
    cells.pop()
    cells.pop()
    return "/".join(cells)

sys.path.append(get_root_path() + "/include/")
