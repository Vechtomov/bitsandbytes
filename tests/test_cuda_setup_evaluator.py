import os
from typing import List, NamedTuple
import platform
import pytest
import torch
from pathlib import Path

# hardcoded test. Not good, but a sanity check for now
def test_manual_override():
    manual_cuda_path = str(Path('/mmfs1/home/dettmers/data/local/cuda-12.2'))

    pytorch_version = torch.version.cuda.replace('.', '')

    assert pytorch_version != 122

    os.environ['CUDA_HOME']='{manual_cuda_path}'
    os.environ['CUDA_VERSION']='122'
    assert str(manual_cuda_path) in os.environ['LD_LIBRARY_PATH']
    import bitsandbytes as bnb
    loaded_lib = bnb.cuda_setup.main.CUDASetup.get_instance().binary_name
    assert loaded_lib == 'libbitsandbytes_cuda122.so'





    # if CONDA_PREFIX exists, it has priority before all other env variables
    # but it does not contain the library directly, so we need to look at the a sub-folder

    # not testing windows platform
    if(platform.system()=="Windows"):
        return

    version = ""
    if "CONDA_PREFIX" in os.environ:
        ls_output, err = bnb.utils.execute_and_return(f'ls -l {os.environ["CONDA_PREFIX"]}/lib/libcudart.so.11.0')
        major, minor, revision = (ls_output.split(" ")[-1].replace("libcudart.so.", "").split("."))
        version = float(f"{major}.{minor}")



