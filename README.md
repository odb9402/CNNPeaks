CNN-peaks
===================


CNN-peaks is a **Convolution Neural Network(CNN)** based ChIP-Seq peak calling software. 

----------

Install
-------------

CNN-peaks has no extra Installation. You can use it just by making clone of this repository. However, if you do not have samtools which is higher than 1.7 version you should install it.

> **Install samtools 1.8 :**
> cd dependencies
> chmod 775 install_samtools.sh
> ./install_samtools.sh

Install with Docker
-------------

You can build Docker image if you want to run CNN-peaks on your Docker container. However, you might use Nvidia-docker as long as you use our Dockerfile.

---------

Quick start
-------------
Examples of CNN-peaks command:
> **preprocessing:**
> python CNNpeaks -m preprocess -i testdir/
>
> **CNN model building:**
> python CNNpeaks -m buildModel -i testdir/ -k 4
> 
> **peak calling with trained model**
> python CNNpeaks -m peakCall -i myOwnChIPSeq.bam -n 0

And CNN-peaks also gives some additional processes for cleaning labeled data, calculating performance ( sensitivity , specificity ) between labeled data and actual peak calling results.

> **calculate performance from labeled data:**
> python CNNpeaks -m errorCall 

Labeled data
--------------- 
CNN-peaks uses labeled data which has its own format. All these approach that use labeled data for marking is from [1]. Examples of labeled data is as the following below. (It based on ASCII)

> 
	chr1:1,000,000-1,100,000 peaks K562
	chr1:1,100,000-1,200,000 peakStart K562
	chr1:1,250,000-1,300,000 peakEnd K562
	chr2:10,000,000-10,002,000 peaks


In line 1, **peaks**, this means that cell K562 at least has one peak in a region (chr1:1,000,000-1,100,000). In line 2, 3 ,**peakStart, peakEnd**,  it means that cell K562 just only one peak in the regions. In line 4, there is no peak in that regions about K562 or other cell because there is no matched cell line name at this raw. If you want to use this label data on other cells,  all these line 1-4 mean **noPeak** because there is no cell name in  the lines.  If you want to know specific rules or methods of this labeling work, please look [here.](https://academic.oup.com/bioinformatics/article/33/4/491/2608653/Optimizing-ChIP-seq-peak-detectors-using-visual)


Usage of CNN-peaks
---------------
**1. Training data to build a CNN-peaks model.**

Before you try to call peaks with your ChIP-Seq Data, CNN-peaks model should be trained by internal module in CNN-peaks. CNN-peaks gives **preprocessing** module for generating actual training data samples with labeled data and bam alignment files of them. You can use our labeled data sample or use your own labeled data to generate training data. When you use its preprocessing module, the input of the module is a single directory which includes labeled data and bam files. For example, a directory which has name "./TestPreProcess" includes these files:

> **preprocessing:**
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


**2. Build CNN-peaks model with preprocessed data**

After you created your own training data with our **preprocess** module, you can build CNN-peaks model by using our **buildModel** module. The module run as 10-cross validation so it will throw 10 models. But you can adjust this by using -k parameter so that you can build your model with k-fold cross validation. Results of running the module include visualization of peak predictions about test data, train and test sensitivity and specificity during training process and trained models. You can check those visualization results and saved tensorflow variables at a "models" directory in a path of CNNpeaks.

> **CNN model building:**
> python CNNpeaks -m buildModel -i testdir/ -kf 4

**3. Peak calling with a trained model**

If you finish build your CNNpeaks model, **buildModel** module generated 'k' numbers of saved tensorflow variables. All you have to do is just pick your saved model number and bam alignment file as an input of peak calling.

> **peak calling with trained model**
> python CNNpeaks -m peakCall -i myOwnChIPSeq.bam -n 0

------
Optional parameters
------
CNNpeaks gives some additional parameters. ( Do not have to adjust )

| Parameter            |     Phase     | Default | Description                                             |
|----------------------|:-------------:|--------:|---------------------------------------------------------|
| -g , –gridSize       |   All steps   |   12000 | Define size of input and output tensors of model.       |
| -s , –searchingDist  | preprocessing |   60000 | DBSCAN clustering parameter during preprocessing steps. |
| -eps , –basePointEPS | preprocessing |   30000 | DBSCAN clustering parameter during preprocessing steps. |
| -w , –windowSize     | peakCalling   | 100000  | Window size for peak calling.                           |

------
>**CITATION**
1. HOCKING, Toby Dylan, et al. Optimizing ChIP-seq peak detectors using visual labels and supervised machine learning. Bioinformatics, 2016, 33.4: 491-499.
