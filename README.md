
<p align="center">
    <img src="https://github.com/odb9402/ConvPeaks/blob/master/CNNpeaks.png" alt="CNN-Peaks logo">
</p>

----------

CNN-peaks is a **Convolution Neural Network(CNN)** based ChIP-Seq peak calling software. 

Install
-------------

CNN-Peaks needs Python higher than 3.x. CNN-Peaks requires several Python packages including Tensorflow. 

> **Required Python packages:**
>
> -  numpy
> -  scipy
> -  sklearn
> -  tensorflow
> -  pandas
> -  progressbar2
> -  pysam
>
> **Install with pip**
>
> ```
> pip install numpy scipy sklearn tensorflow pandas progressbar2 pysam
> ```



If you do not have samtools which is higher than 1.7 version you should install it.

> **Install samtools 1.8 :**
>
> ```
> cd dependencies
> ./install_samtools.sh
> ```
>
> 

Note that if you want to GPU accelerated CNN-Peaks, your tensorflow should be configured to use GPU.

Install with Docker
-------------

You can build Docker image if you want to run CNN-peaks on your Docker container. However, you might use Nvidia-docker as long as you use our Dockerfile.

---------

Quick start
-------------
Examples of CNN-peaks command:


> **preprocessing:**
>
> python CNNpeaks -m preprocess -i testdir/
>
> **CNN model building:**
>
> python CNNpeaks -m buildModel -i testdir/ -kf 4 -ncv
>
> **peak calling with trained model:**
>
> python CNNpeaks -m peakCall -i myOwnChIPSeq.bam -g hg38

And CNN-peaks also gives some additional processes for cleaning labeled data, calculating performance ( sensitivity , specificity ) between labeled data and actual peak calling results.



> **calculate performance from labeled data:**
> python CNNpeaks -m errorCall 





Labeled data
--------------- 
CNN-peaks uses labeled data which has its own format. All these approaches that use labeled data for marking are from [1]. Examples of labeled data is as the following below. (It based on ASCII)


> chr1:1,000,000-1,100,000 peaks K562

> chr1:1,100,000-1,200,000 peakStart K562

> chr1:1,250,000-1,300,000 peakEnd K562

> chr2:10,000,000-10,002,000 peaks


In line 1, **peaks**, this means that cell K562 at least has one peak in a region (chr1:1,000,000-1,100,000). In line 2, 3 ,**peakStart, peakEnd**,  it means that cell K562 just only one peak in the regions. In line 4, there is no peak in those regions about K562 or other cells because there is no matched cell line name at this raw. If you want to use this label data on other cells,  all these line 1-4 mean **noPeak** because there is no cell name in  the lines.  If you want to know specific rules or methods of this labeling work, please look [here.](https://academic.oup.com/bioinformatics/article/33/4/491/2608653/Optimizing-ChIP-seq-peak-detectors-using-visual)


Usage of CNN-peaks
---------------
### **1. Training data to build a CNN-peaks model.**

Before you try to call peaks with your ChIP-Seq Data, CNN-peaks model should be trained by internal module in CNN-peaks. CNN-peaks gives **preprocessing** module for generating actual training data samples with labeled data and bam alignment files of them. You can use our labeled data sample or use your own labeled data to generate training data. When you use its preprocessing module, the input of the module is a single directory which includes labeled data and bam files. For example, a directory which has name "./TestPreProcess" includes these files:

> **preprocessing:**
>
> python CNNpeaks -m preprocess -i TestPreProcess

> **./TestPreProcess:**

> - H3K36me3_None.txt 
> - H3K36me3_K562.bam
> - H3K36me3_A549.bam

Each file must follow a format of filename that is **Target_CellName.bam**. The case of labeled data, it must be like this :**Target_AnyString.txt**. The labeled data of this example H3K36me3_None.txt has data which includes both of K562 and A549.
As a result, directories that include labeled data , reference gene data and read depth data are generated.

> **./TestPreProcess:**

> - H3K36me3_None.txt 
> - H3K36me3_K562.bam
> - H3K36me3_A549.bam
> - H3K36me3_K562 (directory)
> - H3K36me3_A549 (directory)



### **2. Build CNN-peaks model with preprocessed data**

After you created your own training data with our **preprocess** module, you can build CNN-peaks model by using our **buildModel** module. The module run as 10-cross validation so it will throw 10 models. But you can adjust this by using -k parameter so that you can build your model with k-fold cross validation. Results of running the module include visualization of peak predictions about test data, train and test sensitivity and specificity during the training process and trained models. You can check those visualization results and saved tensorflow variables at a "models" directory in a path of CNNpeaks.

> **CNN model building:**
>
> python CNNpeaks -m buildModel -i testdir/ -kf 4 -ncv



### **3. Peak calling with the a trained model**

If you finish build your CNNpeaks model, **buildModel** module generated 'k' numbers of saved tensorflow variables. All you have to do is just pick your saved model number and bam alignment file as an input of peak calling.

> **peak calling with trained model**
>
> python CNNpeaks -m peakCall -i myOwnChIPSeq.bam -n 0 -g hg38



## Output format of CNN-Peaks

Basically,the output format of CNN-Peaks follows regular .bed format which is tab delimited. 1 to 5`th columns are absolutely same with a regular bed file format that can be displayed in an annotation track for visualization tools such as UCSC genome browser and IGV.

Field of CNN-Peaks output format is:

1. ​       **Chromosome** – The name of chromosome
2. ​       **Chromosome Start** – The starting position of features in the chromosome.
3. ​       **Chromosome End** – The ending position of the feature in the chromosome.
4. ​       **Name** – The randomly generated string of the peak.
5. ​        **Score** – The score of peak signals determined by column 6 and 9.
6. ​        **Score2** – The score of peak signals determined by column 6 and 8.
7. ​        **Sigmoid activation value** – The sigmoid activation value of the peak from CNN-Peaks model output.
8. ​       **P-value (Avg)** – Average p-value in the peak region on Poisson distribution with a window.
9. ​       **P-value (Min)** – Minimum p-value in the peak region on Poisson distribution with a window.



## Filtering peak calling result

After calling peaks by CNN-Peaks model, you can filter out some peaks with scoring metrics in CNN-Peaks result ( column 5 to 9 ) using script in CNN-Peaks.

> ```
> [filteringPeaks.sh -o<options> -t <threshold> -i <input> > <output>]
> ```

| Options | Type              | Description                                                  |
| ------- | ----------------- | ------------------------------------------------------------ |
| [-o]    | interval          | Filtering peaks by those interval (3'rd col - 2'ed col)      |
|         | score             | Filtering peaks by the score of peak regions ( 4'th col )    |
|         | scoreBroad        | Filtering peaks by the score of peak regions ( 5'th col )    |
|         | sigVal            | Filtering peaks by the sigmoid activation value of peak regions ( 6'th col ) |
|         | pValueBroad       | Filtering peaks by the P-value of peak regions ( 7'th col )  |
|         | pValue            | Filtering peaks by the P-value of peak regions ( 8'th col )  |
| [-t]    | *Threshold Value* | Threshold value for -o                                       |
| [-i]    | *File Name*       | Input file name                                              |
|         |                   |                                                              |


<a href="http://164.125.34.223/CNNPeaks_Alpha/alphaModel.tar" download>Test</a>

------
------
>**CITATION**

1. HOCKING, Toby Dylan, et al. Optimizing ChIP-seq peak detectors using visual labels and supervised machine learning. Bioinformatics, 2016, 33.4: 491-499.
