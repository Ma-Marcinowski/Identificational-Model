## Identificational-Model

### 0. Introduction

* #### 0.1. Objective of the author's repository was to introduce multiple - varying in terms of interpretability - methods for identification of offline handwritten documents' authors by artificial neural networks. The purpose of these approaches was to create an empirical background for analyses of interpretable machine learning tools developed in the field of computational forensics.

* #### 0.2. Versioning and identifiers (e.g. vX.Y.Z):

    * X indicates the model version (otherwise 0);
    * Y indicates the method of preprocessing (otherwise 0);
    * Z indicates any extra variation of the given X.Y base combination (otherwise 0 or omitted).

* #### 0.4. Keywords:

    * Computational, statistical, probabilistic; 
    * Forensics, criminalistics, analysis, examination;
    * Handwriting, signatures, documents;
    * Neural, networks, deep, machine, learning, artificial, intelligence, ANN, AI;
    * Interpretability, explainability. 

* #### 0.5. Data (403 documents by 145 writers):

  * 0.5.1. Raw data:
       
    * Dataset of 1604 documents (full page scans) from CVL database (310 writers), by: F. Kleber, S. Fiel, M. Diem, R. Sablatnig, *CVL-Database: An Off-line Database for Writer Retrieval, Writer Identification and Word Spotting*, "In Proc. of the 12th Int. Conference on Document Analysis and Recognition (ICDAR) 2013" 2013, p. 560 - 564;
                
    * Dataset of 1539 documents (full page scans) from IAM (offline handwritten documents) database (657 writers), by: U. Marti, H. Bunke, *The IAM-database: An English Sentence Database for Off-line Handwriting Recognition*, "Int'l Journal on Document Analysis and Recognition" 2002, No. 5, p. 39 - 46.
  
  * 0.5.2. Categorized data:

    * Test subsets of the IAM and CVL datasets were categorized in terms of handwritting features, that are indicative of gender and handedness, for the purpose of verificational models' evaluation (https://github.com/Ma-Marcinowski/Verificational-Model).

    * Overall there were 403 documents by 145 writers; IAM: 118 documents by 214 writers; CLV: 27 documents by 189 writers.
    
    * Dataframes of handwriting features, categories and labels are available at the /Dataframes/ folder.
    
  * 0.5.3. Preprocessed data:

    * 0.5.3.1. Preprocessing v0.1 (grayscaled, undenoised, 256x256px):

      * Images are converted to grayscale, colour inverted, then rigid extraction of writing space is applied, reduction of extract dimensions to 1024 px on 1024 px, division of extracts into 256 px on 256 px patches, conversion from the tif to png format.
      
      * Patches which do not contain or contain only small amounts of text are omitted (vide /Preprocessing/Filters/ folder ).

      * Overall there were 3384 patches (2018 IAM and 1366 CLV).

  * 0.5.4. Dataframed data:

    * 0.5.4.1. Dataframing v0.1:

      * There is a column containing image paths and there are one hot encoded: 145 columns referring to the authors' IDs; 2 columns referring to the authors' gender and handedness; 18 columns referring to the indicative features.
        
      * Around 20% of each author's document patches were randomly sampled for models' testing and validation, while the remainder was utilized for models' training.

  * 0.5.5. Basic evaluation metrics:

    * Categorical Crossentropy - Loss;
    * Accuracy - Acc;
    * True Positive Rate / Sensitivity - `(TP/P)` - TPR;
    * True Negative Rate / Specificity - `(TN/N)` - TNR;
    * False Positive Rate - `(FP/N)` - FPR;
    * False Negative Rate - `(FN/P)` - FNR;
    * Positive Predictive Value - `(TP/(TP+FP))` - PPV;
    * Negative Predictive Value - `(TN/(TN+FN))` - NPV;
    * Area under the ROC curve - AUC.

### 1. Model v1.1.0

* #### 1.1. Architecture (available at the /Models/Model_Training_v1.1.0.py):

  * 1.1.1. Architecture based on the VGG 16, by: K. Simonyan, A. Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*, arXiv:1409.1556v6 [cs.CV] 2015, pp. 1-14.

  * 1.1.2. However:
  
    * Batch normalization layers are added among the convolutional and dese layers;

    * Dilation of 2 is utilized at convolutional layers;
    
    * Global Average Pooling layer is utilized instead of the last max-pooling layer;

    * Dropout layers are supercedeing all dense layers;

    * The last dense layer has 145 neurons.  

* #### 1.2. Hyperparameteres (all training updates available at the /Logs/Model_v1.1.0_Training_Log.csv):

    * Loss: Categorical Crossentropy
    
    * Optimizer - Adam (Adaptive Moment Estimation):
      * Initial learning rate (alpha) - 0.001 (1e-3);
      * Beta_1 , beta_2, epsilon - as recommended by: D. Kingma, J. Ba, *Adam: A Method for Stochastic Optimization*, arXiv:1412.6980v9 [cs.LG] 2017, p. 2.
      
    * Learning Rate Reductions (automatic, unless manual):
      * Observing - minimal validation loss.
      * If no improvement for - 5 epochs;
      * Then reductions by factor - 0.1.

    * Initial batchsize - 32
    
    * Initial dropout rate - 0.25

* #### 1.3. Training (available at the /Models/Model_Training_v1.1.0.py):

    * Lasted - 60 epochs.
    
    * Training log available at the /Logs/Model_v1.1.0_Training_Log.csv.

    * Results:
    
    | Model | Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
    | --- | --- | --- | --- | --- |  --- |
    | v1.1.0 | 54 | 0.4011 | 0.8797 | 1.2081 | 0.8056 | 

* #### 1.3. Testing (available at the /Models/Model_Testing_v1.1.0.py):

    | Model | Epoch | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | v1.1.0 | ??? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? |
