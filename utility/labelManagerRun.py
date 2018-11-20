from labelManager import labelManager
from os.path import abspath
from tkinter.filedialog import askdirectory

filename = askdirectory(title="Select Your Directory")

labelManager(abspath(filename))