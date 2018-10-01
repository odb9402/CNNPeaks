from tkinter import *
import os
import glob
import buildModel.buildModel as buildModel
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import matplotlib

class labelManager():

    def __init__(self, directory):
        self.startLoc = 0
        self.endLoc = 0
        self.fileIndex = 0

        loads = self.fileNameLoad(directory)

        self.data_list = loads['data']
        self.file_list = loads['file_name']

        self.root = Tk()
        self.root.title('Label data checker')
        self.root.geometry('750x550')

        self.fig = Figure(figsize=(5,5), dpi=100)
        self.subplt = self.fig.add_subplot(111)
        self.subplt.plot(self.data_list[0][self.fileIndex],'k')
        self.subplt.plot(self.data_list[1][self.fileIndex],'r.')
    
        self.drop_button = Button(self.root, text="Drop", command=None)#, height=3, width=16)
        self.drop_button.grid(row=0, column = 1, columnspan=2, sticky=W+E+N+S)
        #self.drop_button.place(x=530, y=40)
        self.prev_button = Button(self.root, text="Prev", command=self.prevData)#, height=3, width=6)
        self.prev_button.grid(row=1, column = 1,sticky=W+E+N+S)
        #self.prev_button.place(x=530, y=100)
        self.next_button = Button(self.root, text="Next", command=self.nextData )#, height=3, width=6)
        self.next_button.grid(row=1, column = 2,sticky=W+E+N+S)
        #self.next_button.place(x=610, y=100)
        self.noPeak_button = Button(self.root, text="noPeak", command=lambda: self.adjustData(peak=False))#, height=3, width=16)
        self.noPeak_button.grid(row=4, column = 1, columnspan=2,sticky=W+E+N+S)
        #self.noPeak_button.place(x=530, y=290)
        self.peak_button = Button(self.root, text="peak", command=lambda : self.adjustData(peak=True) )# , height=3, width=16)
        self.peak_button.grid(row=5, column = 1, columnspan=2,sticky=W+E+N+S)
        #self.peak_button.place(x=530, y=350)

        self.startLabel = Label(self.root, text="Start point :")
        self.startLabel.grid(row=2, column=1)
        self.startAxis = Text(self.root, height = 2, width = 12)
        self.startAxis.insert(END, self.startLoc)
        self.startAxis.grid(row=2, column = 2)
        #self.startAxis.place(x=530, y=200)
        self.endLabel = Label(self.root, text="End point   :")
        self.endLabel.grid(row=3, column=1)
        self.endAxis = Text(self.root, height = 2, width = 12)
        self.endAxis.insert(END, self.endLoc)
        self.endAxis.grid(row=3, column = 2 )
        #self.endAxis.place(x=530, y=250)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=0, column= 0, rowspan=6, sticky=W+E+N+S, padx=20, pady=20)
        #self.canvas.get_tk_widget().pack(anchor='w',padx=20, pady=20)
        self.fig.canvas.mpl_connect('button_press_event', self.dragStart)
        self.fig.canvas.mpl_connect('button_release_event', self.dragEnd)

        self.root.mainloop()


    def dragStart(self, event):
        self.startLoc = int(event.xdata)
        self.startAxis.delete('1.0', END)
        self.startAxis.insert(END, self.startLoc)


    def dragEnd(self, event):
        self.endLoc = int(event.xdata)
        self.endAxis.delete('1.0', END)
        self.endAxis.insert(END, self.endLoc)


    def dropLabels(self):
        pass


    def nextData(self):
        if(len(self.data_list[0]) < self.fileIndex + 1):
            print("next_index")
        else:
            self.fileIndex += 1
            self.drawPlot()


    def prevData(self):
        if(0 > self.fileIndex - 1):
            print("prev_index")
        else:
            self.fileIndex -= 1
            self.drawPlot()


    def adjustData(self, peak=True):
        filename = self.file_list[2][self.fileIndex]
        length = len(self.data_list[1][self.fileIndex])

        i = min(int(self.startAxis.get('1.0',END)), int(self.endAxis.get('1.0',END)))
        
        while i < max(int(self.startAxis.get('1.0',END)), int(self.endAxis.get('1.0',END))):
            if peak:
                self.data_list[1][self.fileIndex][i] = 1 
            else:
                self.data_list[1][self.fileIndex][i] = 0
            i += 1
        self.drawPlot()

        noPeakColumn = []
        for i in range(length):
            if self.data_list[1][self.fileIndex][i] == 1:
                noPeakColumn.append(0)
            else:
                noPeakColumn.append(1)

        new_label_df = pd.DataFrame({'peak': self.data_list[1][self.fileIndex],
            'noPeak' : noPeakColumn } , dtype=int, index=range(length))

        new_label_df.to_csv(filename)
        print("{} saved.".format(filename))




    def drawPlot(self):
        self.subplt.cla()
        self.subplt.plot(self.data_list[0][self.fileIndex],'k')
        self.subplt.plot(self.data_list[1][self.fileIndex],'r.')
        self.canvas.show()


    def fileNameLoad(self, dir_name, num_grid=12000):
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

        input_file_list = []
        ref_file_list = []
        label_file_list = []

        for dir in input_list:
            for chr in input_list[dir]:
                for cls in input_list[dir][chr]:
                    input_file_name = "{}/{}_{}_grid{}.ct".format(dir, chr, cls, num_grid)
                    ref_file_name = "{}/ref_{}_{}_grid{}.ref".format(dir, chr, cls, num_grid)
                    label_file_name = "{}/label_{}_{}_grid{}.lb".format(dir, chr, cls, num_grid)
                    
                    input_file_list.append(input_file_name)
                    ref_file_list.append(ref_file_name)
                    label_file_list.append(label_file_name)

                    reads = (pd.read_csv(input_file_name))['readCount'].values.reshape(num_grid)
                    label = (pd.read_csv(label_file_name))['peak'].values.transpose()
                    label = buildModel.expandingPrediction(label)
                    
                    data_list.append(reads)
                    label_list.append(label)

        return {'data':(data_list, label_list), 'file_name':(input_file_list, ref_file_list, label_file_list)}