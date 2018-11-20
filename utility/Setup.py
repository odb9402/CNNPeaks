import sys
from cx_Freeze import setup, Executable

build_exe_options = {"includes": [""],
                     "include_files": []}

setup(name="labelManager",
      version="1.0",
      author="pin",
      options = {"build_exe": build_exe_options},
      executables=[Executable("labelManagerRun.py")],
      )