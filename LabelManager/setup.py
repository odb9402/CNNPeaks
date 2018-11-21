import sys
from cx_Freeze import setup, Executable

build_exe_options = {"packages": ["os","sys"],
                     "excludes": ["tkinter","pandas","numpy","matplotlib","scipy"],
                     "includes": ["labelManager"],
                     'include_msvcr': True}

base = None
if sys.platform == "win32":
    base = "Win32GUI"

target = Executable(
    script="labelManagerRun.py",
    base=base,
    icon="peakLabelManager.ico",
    targetName="labelManager.exe"
    )

setup(name="labelManager",
      version="1.0",
      author="pin",
      description = "Label data manager application for CNN-peaks",
      options = {"build_exe": build_exe_options},
      executables=[target]
      )