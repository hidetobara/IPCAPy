import os
import sys

def get_block_size():
    if os.name == "nt": return 64 # windowsの場合は開発
    return 128 # その他は本番

def get_root_path():
    cells = os.path.abspath(os.path.dirname(__file__)).split("\\")
    cells.pop()
    return "/".join(cells)

def get_data_path():
    return get_root_path() + "/data/"