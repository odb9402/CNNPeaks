from labelManager import labelManager
from os.path import abspath
from tkinter.filedialog import askdirectory
from tkinter import Tk



root = Tk()
root.withdraw()

dirPath = askdirectory(title="Select Your Directory")

root.destroy()

labelManager(abspath(dirPath))