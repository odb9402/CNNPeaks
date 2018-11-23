from tkinter import *
import tkinter.messagebox as messagebox
import os
import glob
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ntpath
from matplotlib.figure import Figure
from scipy.signal import gaussian
from scipy.ndimage.filters import maximum_filter1d
import matplotlib.pyplot as plt

def expandingPrediction(input_list, multiple=5):
    """

    :param input_list:
    :param multiple:
    :return:
    """
    expanded_list = []
    for prediction in input_list:
        for i in range(multiple):
            expanded_list.append(prediction)

    return expanded_list


def extractChrClass(dir):
    """
    Extract a chromosome number and a class number from label file names.

    :param dir:
    :return:
    """

    chr_list = set()

    for ct_file in glob.glob(dir + "/*.ct"):
        chr_list.add(path_leaf(ct_file).split('_')[0])

    data_direction = {}
    for chr in chr_list:
        cls_list = []
        for ct_file in glob.glob(dir + "/" + chr + "_*.ct"):
            cls_list.append(path_leaf(ct_file).split('_')[1])
        data_direction[chr] = cls_list

    return data_direction


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(path)


class labelManager():

    def __init__(self, directory):
        self.startLoc = 0
        self.endLoc = 0
        self.thresholdLoc = 0
        self.fileIndex = 0
        self.smoothing = False
        self.smoothing_window = 31
        self.smoothing_var = 1

        loads = self.fileNameLoad(directory)

        self.data_list = loads['data']
        self.file_list = loads['file_name']

        self.root = Tk()
        self.root.title('Label data checker')
        self.root.iconbitmap("peakLabelManager.ico")
        #self.root.geometry('600x600')

        self.fig = Figure(figsize=(4,4), dpi=100)
        self.peak_plot = self.fig.add_subplot(111)

        self.prev_button = Button(self.root, text="Prev(Q)", command=self.prevData)
        self.prev_button.grid(row=0, column = 1,sticky=W+E+N+S)
        self.next_button = Button(self.root, text="Next(W)", command=self.nextData)
        self.next_button.grid(row=0, column = 2,sticky=W+E+N+S)
        self.drop_button = Button(self.root, text="Drop(E)", command=self.dropLabels)
        self.drop_button.grid(row=1, column = 1, columnspan=2, sticky=W+E+N+S)
        self.noPeak_button = Button(self.root, text="noPeak(A)", command=lambda: self.adjustData(peak=False,criteria='region'))
        self.noPeak_button.grid(row=5, column = 1, columnspan=2, sticky=W+E+N+S)
        self.peak_region_button = Button(self.root, text="peak = region(S)", command=lambda : self.adjustData(peak=True, criteria='region'))
        self.peak_region_button.grid(row=6, column = 1, sticky=W+E+N+S)
        self.peak_threshold_button = Button(self.root, text="peak > threshold(D)",command=lambda: self.adjustData(peak=True,criteria='threshold'))
        self.peak_threshold_button.grid(row=6, column=2, sticky=W+E+N+S)

        self.smoothParam = Text(self.root, height=2, width=12)
        self.smoothParam.grid(row=7, column=1)
        self.smooth_button = Button(self.root, text="Smoothing", command=self.smoothingDepth)
        self.smooth_button.grid(row=7, column = 2,sticky=W+E+N+S)

        self.moveFileLabel = Label(self.root, text=" {}/{} ` th label.".format(self.fileIndex + 1,len(self.data_list[0])))
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

        ### Mouse drag events for region selection
        self.fig.canvas.mpl_connect('button_press_event', self.dragStart)
        self.fig.canvas.mpl_connect('button_release_event', self.dragEnd)

        ### Keyboard events
        self.root.bind('<Key>', self.keyPressed)

        self.root.mainloop()

    def keyPressed(self, event):
        if event.char == 'q':
            self.prevData()
        elif event.char == 'w':
            self.nextData()
        elif event.char.lower() == 'a':
            self.adjustData(peak=False,criteria='region')
        elif event.char.lower() == 's':
            self.adjustData(peak=True, criteria='region')
        elif event.char.lower() == 'd':
            self.adjustData(peak=True, criteria='threshold')
        elif event.char.lower() == 'e':
            self.dropLabels()
        else:
            pass

    def moveFile(self, event):
        self.fileIndex = int(self.moveFileEntry.get())
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
        self.thresholdLoc = int(event.ydata)
        self.startAxis.delete('1.0', END)
        self.startAxis.insert(END, self.startLoc)

    def dragEnd(self, event):
        self.endLoc = int(event.xdata)
        self.endAxis.delete('1.0', END)
        self.endAxis.insert(END, self.endLoc)
        self.drawPlot()

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

    def adjustData(self, peak=True, criteria=None):
        filename = self.file_list[2][self.fileIndex]
        length = len(self.data_list[2][self.fileIndex])

        if criteria == 'region':
            i = min(int(self.startAxis.get('1.0',END)), int(self.endAxis.get('1.0',END)))

            while i < max(int(self.startAxis.get('1.0',END)), int(self.endAxis.get('1.0',END))):
                if peak:
                    self.data_list[2][self.fileIndex][i] = 1
                else:
                    self.data_list[2][self.fileIndex][i] = 0
                i += 1
            self.drawPlot()

        elif criteria == 'threshold':
            if self.smoothing:
                read_depth = self.smoothingInput()
            else:
                read_depth = self.data_list[0][self.fileIndex]

            for i in range(len(read_depth)):
                if read_depth[i] >= self.thresholdLoc:
                    self.data_list[2][self.fileIndex][i] = 1
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
        self.peak_plot.cla()

        ### Draw read input data
        if self.smoothing:
            read_depth = self.smoothingInput()
            ##final_depth = (final_depth - final_depth.mean())/(final_depth.std() + 0.0001)
            self.peak_plot.plot(read_depth, 'k', markersize=2, linewidth=1)
        else:
            self.peak_plot.plot(self.data_list[0][self.fileIndex], 'k', markersize=2, linewidth=1)

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
                self.peak_plot.axvspan(start, end, color='red', alpha=0.3)

        ### Draw refSeq
        refSeq_index = []
        for i in range(len(self.data_list[1][self.fileIndex])):
            if self.data_list[1][self.fileIndex][i] == 1:
                refSeq_index.append(i)

self.peak_plot.plot(refSeq_index, [0 for x in range(len(refSeq_index))], 'b|', markersize=8)

        ### Draw Threshold setting
        height = self.thresholdLoc
        self.peak_plot.plot([0, len(self.data_list[0][self.fileIndex])],[height,height], 'y--')

        self.moveFileLabel.configure(text="{}/{} ` th label.".format(self.fileIndex,len(self.data_list[0])))
        self.canvas.show()

    def smoothingInput(self):
        depths = np.array(self.data_list[0][self.fileIndex])
        smoothing_filter = gaussian(self.smoothing_window, self.smoothing_var) / \
                           np.sum(gaussian(self.smoothing_window, self.smoothing_var))
        union_depths = maximum_filter1d(depths, 51)  ## MAX POOL to extract boarder lines
        final_depth = np.convolve(union_depths, smoothing_filter, mode='same')  ## Smoothing boarder lines
        return final_depth

    def fileNameLoad(self, dir_name, num_grid=12000):
        PATH = os.path.abspath(dir_name)
        dir_list = os.listdir(PATH)

        for dir in dir_list:
            dir = os.path.join(PATH,dir)

        input_list = {}
        for dir in dir_list:
            dir = os.path.join(PATH,dir)
            input_list[dir] = extractChrClass(dir)

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
                    label = expandingPrediction(label)

                    data_list.append(reads)
                    label_list.append(label)
                    ref_list.append(refs)

        return {'data':(data_list,ref_list, label_list), 'file_name':(input_file_list, ref_file_list, label_file_list)}
