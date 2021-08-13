## Identificational-Model

### 0. Introduction

* #### 0.1. Objective of the author's repository was to introduce multiple - varying in terms of interpretability - methods for identification of offline handwritten documents' authors by artificial neural networks. The purpose of these approaches was to create an empirical background for analysis of machine learning tools developed in the field of computational forensics in terms of their interpretability.

* #### 0.2. Versioning and identifiers (e.g. vX.Y):

    * X indicates the overall type of the model;
    * Y identifies the given variant of the model.

* #### 0.3. Models (VGG16):

     * Models unrestricted in terms of features extracted, tasked with author identification (type v1):
     
         * Vide 1. Model v1.0 (two FC layers);
     
         * Vide 2. Model v1.1 (two FC layers, but analogous in size to the v2.1);

         * Vide 3. Model v1.2 (three FC layers, but analogous in size to the v2.2);

         * Vide 4. Model v1.3 (three FC layers, but analogous in size to the v2.3);

         * Vide 5. Model v1.4 (three FC layers, but analogous in size to the v2.4);

         * Vide 6. Model v1.5 (two FC layers; tasked with author identification and features extraction);
         

     * Models restricted by supervised features extraction, tasked with author identification (type v2):

        * Vide 7. Model v2.0 (two FC layers, the second one only partially restricted);

        * Vide 8. Model v2.1 (two FC layers, the second one restricted);
     
        * Vide 9. Model v2.2 (three FC layers, the third one restricted);

        * Vide 10. Model v2.3 (three FC layers, the first and third one restricted);

        * Vide 11. Model v2.4 (three FC layers, all restricted);

        * Vide 12. Model v2.5 (analogous to the v2.???, but tasks and restrictions are weighted).

     
* #### 0.4. Keywords:

    * Computational, statistical, probabilistic; 
    * Forensics, criminalistics, analysis, examination;
    * Handwriting, signatures, documents;
    * Neural, networks, deep, machine, learning, artificial, intelligence, ANN, AI;
    * Interpretability, explainability. 

* #### 0.5. Data (403 documents by 145 writers):

  * 0.5.1. Raw data:
       
    * Dataset of 1604 documents (full page scans) from CVL database (310 writers), by: F. Kleber, S. Fiel, M. Diem, R. Sablatnig, *CVL-Database: An Off-line Database for Writer Retrieval, Writer Identification and Word Spotting*, "In Proc. of the 12th Int. Conference on Document Analysis and Recognition (ICDAR) 2013" 2013, p. 560 - 564.
  
  * 0.5.2. Categorized data:

    * Test subset of the CVL dataset was categorized in terms of handwritting features, that were also indicative of gender and handedness, for the purpose of verificational models' evaluation (https://github.com/Ma-Marcinowski/Verificational-Model).

    * Overall there were 189 documents by 27 writers.
    
    * Dataframes of handwriting features, categories and labels are available at the /Dataframes/ folder.
    
  * 0.5.3. Preprocessing (grayscaled, undenoised, 256x256px):

      * Images are converted to grayscale, colour inverted, then rigid extraction of writing space is applied, reduction of extract dimensions to 1024 px on 1024 px, division of extracts into 256 px on 256 px patches, conversion from the tif to png format.
      
      * Patches which do not contain or contain only small amounts of text are omitted (vide /Preprocessing/Filters/ folder ).

      * Overall there were 1366 patches.

  * 0.5.4. Dataframing:

      * There is a column containing image paths and there are one hot encoded: 27 columns referring to the authors' IDs; 4 columns referring to the authors' gender and handedness; 80 columns referring to the indicative features.
        
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





### 1. Model v1.1.0 (plain model, author identification)

* #### 1.1. Architecture (available at the /Models/Model_Training_v1.1.0.py):

  * 1.1.1. Architecture based on the VGG 16, by: K. Simonyan, A. Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*, arXiv:1409.1556v6 [cs.CV] 2015, pp. 1-14.

  * 1.1.2. However:
  
    * Batch normalization layers are added among the convolutional and dese layers;

    * Dilation of 2 is utilized at convolutional layers;
    
    * Global Average Pooling layer is applied instead of the last max-pooling layer;

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
    
### 2. Model v1.1.1 (comparative model, author identification)

* #### 2.1. Architecture (available at the /Models/Model_Training_v1.1.1.py):

  * Vide 1.1.1. Architecture.

  * However:
  
    * There is an additional dense layer of 20 neurons in-between the 2nd and the output dense layer.

* #### 2.2. Hyperparameteres (all training updates available at the /Logs/Model_v1.1.1_Training_Log.csv):

    * Vide 1.2. Hyperparameteres.

    * However: 
    
       * Initial dropout rate - 0.0.

* #### 2.3. Training (available at the /Models/Model_Training_v1.1.1.py):

    * Lasted - 60 epochs.
    
    * Training log available at the /Logs/Model_v1.1.1_Training_Log.csv.

    * Results:
    
    | Model | Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
    | --- | --- | --- | --- | --- |  --- |
    | v1.1.1 | 51 | 0.8940 | 0.7881 | 1.3174| 0.6966  | 

* #### 2.3. Testing (available at the /Models/Model_Testing_v1.1.1.py):

    | Model | Epoch | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | v1.1.1 | ??? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? |
    
### 3. Model v2.1.0 (plain model, author identification and categorization)
### 4. Model v2.1.1 (comparative model, author identification and categorization)
### 5. Model v3.1.0 (plain model, features extraction and author identification and categorization)
### 6. Model v3.1.1 (comparative model, features extraction and author identification and categorization)
### 7. Model v4.1.0 (partial model, features extraction and author categorization)
### 8. Model v4.1.1 (partial model, features extraction to author categorization)
### 9. Model v5.1.0 (main model, features extraction and author categorization to author identification)
### 10. Model v5.1.1 (main model, features extraction to author identification and categorization)
