from tkinter import *
import tkinter.messagebox as messagebox
import os
import glob
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from math import pi, sqrt, exp

from scipy.signal import gaussian

import matplotlib

import utility.utilities


class labelManager():

    def __init__(self, directory):
        self.startLoc = 0
        self.endLoc = 0
        self.fileIndex = 0
        self.smoothing = False
        self.smoothing_window = 31
        self.smoothing_var = 1

        loads = self.fileNameLoad(directory)

        self.data_list = loads['data']
        self.file_list = loads['file_name']

        self.root = Tk()
        self.root.title('Label data checker')
        #self.root.geometry('600x600')

        self.fig = Figure(figsize=(4,4), dpi=100)
        self.subplt = self.fig.add_subplot(111)

        self.drop_button = Button(self.root, text="Drop", command=self.dropLabels)
        self.drop_button.grid(row=0, column = 1, columnspan=2, sticky=W+E+N+S)
        self.prev_button = Button(self.root, text="Prev", command=self.prevData)
        self.prev_button.grid(row=1, column = 1,sticky=W+E+N+S)
        self.next_button = Button(self.root, text="Next", command=self.nextData)
        self.next_button.grid(row=1, column = 2,sticky=W+E+N+S)
        self.noPeak_button = Button(self.root, text="noPeak", command=lambda: self.adjustData(peak=False))
        self.noPeak_button.grid(row=5, column = 1, columnspan=2,sticky=W+E+N+S)
        self.peak_button = Button(self.root, text="peak", command=lambda : self.adjustData(peak=True))
        self.peak_button.grid(row=6, column = 1, columnspan=2,sticky=W+E+N+S)
        self.smoothParam = Text(self.root, height=2, width=12)
        self.smoothParam.grid(row=7, column=1)
        self.smooth_button = Button(self.root, text="Smoothing", command=self.smoothingDepth)
        self.smooth_button.grid(row=7, column = 2,sticky=W+E+N+S)

        self.moveFileLabel = Label(self.root, text=" {}/{} ` th label.".format(self.fileIndex,len(self.data_list[0])))
        self.moveFileLabel.grid(row=2, column=2)
        self.moveFileEntry = Entry(self.root, width=12)
        self.moveFileEntry.bind("<Return>", self.moveFile)
        self.moveFileEntry.grid(row=2, column=1)

        self.startLabel = Label(self.root, text="Start point :")
        self.startLabel.grid(row=3, column=1)
        self.startAxis = Text(self.root, height = 2, width = 12)
        self.startAxis.insert(END, self.startLoc)
        self.startAxis.grid(row=3, column = 2)

        self.endLabel = Label(self.root, text="End point   :")
        self.endLabel.grid(row=4, column=1)
        self.endAxis = Text(self.root, height = 2, width = 12)
        self.endAxis.insert(END, self.endLoc)
        self.endAxis.grid(row=4, column=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=0, column= 0, rowspan=8, sticky=W+E+N+S, padx=20, pady=20)

        self.drawPlot()

        self.fig.canvas.mpl_connect('button_press_event', self.dragStart)
        self.fig.canvas.mpl_connect('button_release_event', self.dragEnd)

        self.root.mainloop()


    def moveFile(self, event):
        self.fileIndex = int(self.moveFileEntry.get()) + 1
        self.drawPlot()

    def smoothingDepth(self):
        self.smoothing = not self.smoothing
        value = self.smoothParam.get('1.0',END)
        values = value.split(',')
        self.smoothing_window = int(values[0])
        self.smoothing_var = int(values[1])
        self.drawPlot()


    def dragStart(self, event):
        self.startLoc = int(event.xdata)
        self.startAxis.delete('1.0', END)
        self.startAxis.insert(END, self.startLoc)


    def dragEnd(self, event):
        self.endLoc = int(event.xdata)
        self.endAxis.delete('1.0', END)
        self.endAxis.insert(END, self.endLoc)


    def dropLabels(self):
        os.remove(self.file_list[2][self.fileIndex])
        os.remove(self.file_list[1][self.fileIndex])
        os.remove(self.file_list[0][self.fileIndex])

        print("<{}> is removed ".format(self.file_list[1][self.fileIndex]))

        self.file_list[0].pop(self.fileIndex)
        self.file_list[1].pop(self.fileIndex)
        self.file_list[2].pop(self.fileIndex)

        self.data_list[0].pop(self.fileIndex)
        self.data_list[1].pop(self.fileIndex)
        self.data_list[2].pop(self.fileIndex)

        if self.fileIndex > len(self.data_list[0]):
            self.fileIndex -= 1
        self.drawPlot()


    def nextData(self):
        if(len(self.data_list[0]) - 1 < self.fileIndex + 1):
            print("next_index")
            messagebox.showinfo("Announce", "It is end of the label. [{}/{}]".format(len(self.data_list[0]),len(self.data_list[0])))
        else:
            self.fileIndex += 1
            self.drawPlot()


    def prevData(self):
        if(0 > self.fileIndex - 1):
            print("prev_index")
            messagebox.showinfo("Announce", "It is start of the label. [1/{}]".format(len(self.data_list[0])))
        else:
            self.fileIndex -= 1
            self.drawPlot()


    def adjustData(self, peak=True):
        filename = self.file_list[2][self.fileIndex]
        length = len(self.data_list[2][self.fileIndex])

        i = min(int(self.startAxis.get('1.0',END)), int(self.endAxis.get('1.0',END)))

        while i < max(int(self.startAxis.get('1.0',END)), int(self.endAxis.get('1.0',END))):
            if peak:
                self.data_list[2][self.fileIndex][i] = 1
            else:
                self.data_list[2][self.fileIndex][i] = 0
            i += 1
        self.drawPlot()

        compressed_label = []

        ###### Save new label ########
        for i in range(length):
            if i % 5 == 0:
                compressed_label.append(self.data_list[2][self.fileIndex][i])

        noPeakColumn = []
        for i in range(length//5):
            if compressed_label[i] == 1:
                noPeakColumn.append(0)
            else:
                noPeakColumn.append(1)

        new_label_df = pd.DataFrame({'peak': compressed_label,
            'noPeak' : noPeakColumn } , dtype=int, index=range(length//5))
        print(new_label_df)

        os.remove(filename)
        new_label_df.to_csv(filename)
        print("{} saved.".format(filename))




    def drawPlot(self):
        self.subplt.cla()

        ### Draw read input data
        if self.smoothing:
            depths = np.array(self.data_list[0][self.fileIndex])
            smoothing_filter = gaussian(self.smoothing_window,self.smoothing_var)/\
                               np.sum(gaussian(self.smoothing_window,self.smoothing_var))
            #smoothing_filter = [1/31 for x in range(31)]
            conv_depths = np.convolve(depths, smoothing_filter, mode='same')
            #self.subplt.plot(np.maximum(depths, conv_depths).tolist(),'k',markersize=2, linewidth=1)
            self.subplt.plot(conv_depths, 'k', markersize=2, linewidth=1)
        else:
            self.subplt.plot(self.data_list[0][self.fileIndex],'k', markersize=2, linewidth=1)

        ### Highlight on label
        onPositive = False
        start = 0
        end = 0
        for i in range(len(self.data_list[2][self.fileIndex])):
            if self.data_list[2][self.fileIndex][i] == 1 and not onPositive:
                start = i
                onPositive = True
            elif self.data_list[2][self.fileIndex][i] == 0 and onPositive:
                end = i
                onPositive = False
                self.subplt.axvspan(start, end, color='red', alpha=0.3)

        ### Draw refSeq
        refSeq_index = []
        for i in range(len(self.data_list[1][self.fileIndex])):
            if self.data_list[1][self.fileIndex][i] == 1:
                refSeq_index.append(i)
        self.subplt.plot(refSeq_index, [0 for x in range(len(refSeq_index))], 'bo', markersize=6)

        self.moveFileLabel.configure(text="{}/{} ` th label.".format(self.fileIndex,len(self.data_list[0])))

        self.canvas.show()


    def fileNameLoad(self, dir_name, num_grid=12000):
        PATH = os.path.abspath(dir_name)
        dir_list = os.listdir(PATH)

        for dir in dir_list:
            dir = os.path.join(PATH,dir)

        input_list = {}
        for dir in dir_list:
            dir = os.path.join(PATH,dir)
            input_list[dir] = utility.utilities.extractChrClass(dir)

        data_list = []
        ref_list = []
        label_list = []

        input_file_list = []
        ref_file_list = []
        label_file_list = []

        for dir in input_list:
            for chr in input_list[dir]:
                for cls in input_list[dir][chr]:
                    input_file_name = os.path.join(dir, "{}_{}_grid{}.ct".format(chr, cls, num_grid))
                    ref_file_name = os.path.join(dir,"ref_{}_{}_grid{}.ref".format(chr, cls, num_grid))
                    label_file_name = os.path.join(dir,"label_{}_{}_grid{}.lb".format(chr, cls, num_grid))

                    input_file_list.append(input_file_name)
                    ref_file_list.append(ref_file_name)
                    label_file_list.append(label_file_name)

                    reads = (pd.read_csv(input_file_name))['readCount'].values.reshape(num_grid)
                    refs = (pd.read_csv(ref_file_name))['refGeneCount'].values.reshape(num_grid)
                    label = (pd.read_csv(label_file_name))['peak'].values.transpose()
                    label = utility.utilities.expandingPrediction(label)

                    data_list.append(reads)
                    label_list.append(label)
                    ref_list.append(refs)

        return {'data':(data_list,ref_list, label_list), 'file_name':(input_file_list, ref_file_list, label_file_list)}
