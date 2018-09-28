from tkinter import *
import os
import glob
import buildModel.buildModel as buildModel
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import matplotlib

def main():
    data_list, label_list = fileNameLoad("../data_clean_with_ref")
    file_index = 0

    root = Tk()
    root.title('Label data checker')
    root.geometry('800x600')

    drop_button = Button(root, text="Drop", command=None, height=2, width=4)
    drop_button.place(x=720, y=40)
    pass_button = Button(root, text="Pass", command=None, height=2, width=4)
    pass_button.place(x=720, y=100)
    
    fig = Figure(figsize=(5, 5), dpi=100)
    subplt = fig.add_subplot(111)
    subplt.plot(data_list[file_index])
    subplt.plot(label_list[file_index])

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.show()
    canvas._tkcanvas.pack()



    

    root.mainloop()


def dropLabels():
    pass


def fileNameLoad(dir_name, num_grid=12000):
    PATH = os.path.abspath(dir_name)
    dir_list = os.listdir(PATH)

    for dir in dir_list:
        dir = PATH + '/' + dir

    input_list = {}
    for dir in dir_list:
        dir = PATH + '/' + dir
        input_list[dir] = buildModel.extractChrClass(dir)

    data_list = []
    label_list = []

    for dir in input_list:
        for chr in input_list[dir]:
            for cls in input_list[dir][chr]:
                input_file_name = "{}/{}_{}_grid{}.ct".format(dir, chr, cls, num_grid)
                ref_file_name = "{}/ref_{}_{}_grid{}.ref".format(dir, chr, cls, num_grid)
                label_file_name = "{}/label_{}_{}_grid{}.lb".format(dir, chr, cls, num_grid)
                
                reads = (pd.read_csv(input_file_name))['readCount'].values.reshape(num_grid)
                label = (pd.read_csv(label_file_name))['peak'].values.transpose()
                label = buildModel.expandingPrediction(label)
                
                data_list.append(reads)
                label_list.append(label)

    return (data_list, label_list)


if __name__ == '__main__':
    main()