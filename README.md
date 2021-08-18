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
         
     * Models restricted by supervised features extraction, tasked with author identification (type v2):

        * Vide 6. Model v2.0 (two FC layers, the second one only partially restricted);

        * Vide 7. Model v2.1 (two FC layers, the second one restricted);
     
        * Vide 8. Model v2.2 (three FC layers, the third one restricted);

        * Vide 9. Model v2.3 (three FC layers, the first and third one restricted);

        * Vide 10. Model v2.4 (three FC layers, all restricted);

        * Vide 11. Model v2.5 (analogous to the v2.???, but tasks and restrictions are weighted)

     * Models unrestricted in terms of features extracted, but tasked with author and features identification (type v3):
     
        * Vide 12. Model v3.0 (two FC layers; tasked with author and features identification).
        
        * Vide 13. Model v3.1 (analogous to the v3.1, but tasks are loss weighted in favour of the author identification).

     * Models unrestricted in terms of features extracted, but tasked features identification (type v4):
     
        * Vide 14. Model v4.0 (two FC layers; tasked with features identification).

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

  * 0.5.4. Training and testing subsets:

      * There were 7 documents per author, numbered 1, 2, 3, 4, 6, 7, 8.

      * All documents no. 7 and 8 were sampled for models' testing and validation.

      * All remaining documents (no. 1-6) were sampled for models' training.
    
  * 0.5.4. Preprocessing (grayscaled, undenoised, 256x256px):

      * Images are converted to grayscale, colour inverted, then rigid extraction of writing space is applied, reduction of extract dimensions to 1024 px on 1024 px, division of extracts into 256 px on 256 px patches, conversion from the tif to png format.
      
      * Patches which do not contain or contain only small amounts of text are omitted (vide /Preprocessing/Filters/ folder ).

      * Overall there were 1366 patches (412 test patches and 954 training patches).

  * 0.5.5. Dataframing:

      * There is a column containing image paths and there are one hot encoded: 27 columns referring to the authors' IDs; 4 columns referring to the authors' gender and handedness; 80 columns referring to the indicative features.

  * 0.5.6. Basic evaluation metrics:

    * Categorical Crossentropy - Loss;
    * Accuracy - Acc;
    * True Positive Rate / Sensitivity - `(TP/P)` - TPR;
    * True Negative Rate / Specificity - `(TN/N)` - TNR;
    * False Positive Rate - `(FP/N)` - FPR;
    * False Negative Rate - `(FN/P)` - FNR;
    * Positive Predictive Value - `(TP/(TP+FP))` - PPV;
    * Negative Predictive Value - `(TN/(TN+FN))` - NPV;
    * Area under the ROC curve - AUC.

### 1. Model v1.0

* #### 1.1. Architecture (available at the /Models/Model_Training_v1.0.py):

  * 1.1.1. Architecture based on the VGG 16, by: K. Simonyan, A. Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*, arXiv:1409.1556v6 [cs.CV] 2015, pp. 1-14.

  * 1.1.2. However:
  
    * Batch normalization layers are added among the convolutional and dese layers;

    * Dilation of 2 is utilized at convolutional layers;
    
    * Global Average Pooling layer is applied instead of the last max-pooling layer;

    * Dropout layers are supercedeing all dense layers;

    * The last dense layer has 27 neurons.  

* #### 1.2. Hyperparameteres (all training updates available at the /Logs/Model_v1.0_Training_Log.csv):

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

* #### 1.3. Training (available at the /Models/Model_Training_v1.0.py):

    * Lasted - 90 epochs.
    
    * Training log available at the /Logs/Model_v1.0_Training_Log.csv.

    * Results:
    
    | Model | Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
    | --- | --- | --- | --- | --- |  --- |
    | v1.0 | 62 | 0.0170 | 1.0000 | 0.2989 | 0.9102 | 

* #### 1.3. Testing (available at the /Models/Model_Testing_v1.0.py):

    | Model | Epoch | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | v1.0 | ??? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? |

### 2. Model v1.1

* #### 2.1. Architecture (available at the /Models/Model_Training_v1.1.py):

  * Vide 1.1.1. Architecture.

  * However:
  
    * There is an additional dense layer of 84 neurons in-between the 2nd and the output dense layer (analogous to the v2.1 architecture).

* #### 2.2. Hyperparameteres (all training updates available at the /Logs/Model_v1.1_Training_Log.csv):

    * Vide 1.2. Hyperparameteres.

    * However: 
    
       * Initial dropout rate - 0.1.

* #### 2.3. Training (available at the /Models/Model_Training_v1.1.py):

    * Lasted - ??? epochs.
    
    * Training log available at the /Logs/Model_v1.1_Training_Log.csv.

    * Results:
    
    | Model | Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
    | --- | --- | --- | --- | --- |  --- |
    | v1.1 | ??? | 0.??? | 0.??? | ?.???| 0.???  | 

* #### 2.3. Testing (available at the /Models/Model_Testing_v1.1.py):

    | Model | Epoch | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | v1.1 | ??? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? |

### 3. Model v1.2
### 4. Model v1.3
### 5. Model v1.4

### 6. Model v2.0

* #### 6.1. Architecture (available at the /Models/Model_Training_v2.0.py):

  * Vide 1.1.1. Architecture.

  * However:
  
    * There second layer is split - among 4012 unrestricted 84 restricted neurons (in terms of features extracted) - later concatenated and fed to the output layer.
    
* #### 6.2. Hyperparameteres (all training updates available at the /Logs/Model_v2.0_Training_Log.csv):

    * Vide 1.2. Hyperparameteres.

    * However:

       * Losses: 
         
         * Categorical crossentropy at the author-identification layer (softmax activation);
         
         * Binary crossentropy at the features-extraction layer (sigmoid activation).
    
       * Initial dropout rate - 0.0.

* #### 6.3. Training (available at the /Models/Model_Training_v2.0.py):

    * Lasted - ??? epochs.
    
    * Training log available at the /Logs/Model_v2.0_Training_Log.csv.

    * Author Identification Results:
    
    | Model | Epoch | Task | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
    | --- | --- | --- | --- | --- | --- | --- |
    | v2.0 | ??? | Author Identification | 0.??? | 0.??? | ?.???| 0.??? |
    | v2.0 | ??? | Features Extraction | 0.??? | 0.??? | ?.???| 0.??? |
    
* #### 6.3. Testing (available at the /Models/Model_Testing_v2.0.py):

    | Model | Epoch | Task | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | v2.0 | ??? | Author Identification | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? |
    | v2.0 | ??? | Features Extraction | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? |

### 7. Model v2.1

* #### 7.1. Architecture (available at the /Models/Model_Training_v2.1.py):

  * Vide 1.1.1. Architecture.

  * However:
  
    * The second FC layer consists of 84 neurons (instead of 4096), that are tasked with supervised features extraction (fed directly to the output layer, which identifies the authors).
    
* #### 7.2. Hyperparameteres (all training updates available at the /Logs/Model_v2.1_Training_Log.csv):

    * Vide 1.2. Hyperparameteres.

    * However:

       * Losses: 
         
         * Categorical crossentropy at the author-identification layer (softmax activation);
         
         * Binary crossentropy at the features-extraction layer (sigmoid activation).
             
       * Initial dropout rate - 0.0.

* #### 7.3. Training (available at the /Models/Model_Training_v2.1.py):

    * Lasted - ??? epochs.
    
    * Training log available at the /Logs/Model_v2.1_Training_Log.csv.

    * Results:
    
    | Model | Epoch | Task | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
    | --- | --- | --- | --- | --- | --- | --- |
    | v2.1 | ??? | Author Identification | 0.??? | 0.??? | ?.???| 0.??? |
    | v2.1 | ??? | Features Extraction | 0.??? | 0.??? | ?.???| 0.??? |
    
* #### 7.3. Testing (available at the /Models/Model_Testing_v2.0.py):

    | Model | Epoch | Task | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | v2.1 | ??? | Author Identification | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? |
    | v2.1 | ??? | Features Extraction | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? |

### 8. Model v2.2


* #### 8.1. Architecture (available at the /Models/Model_Training_v2.2.py):

  * Vide 1.1.1. Architecture.

  * However:
  
    * There is and additional third FC layer of 84 neurons, tasked with supervised features extraction (fed directly to the output layer, which identifies the authors).
    
* #### 8.2. Hyperparameteres (all training updates available at the /Logs/Model_v2.2_Training_Log.csv):

    * Vide 1.2. Hyperparameteres.

    * However:

       * Losses: 
         
         * Categorical crossentropy at the author-identification layer (softmax activation);
         
         * Binary crossentropy at the features-extraction layer (sigmoid activation).
              
       * Initial dropout rate - 0.0.

* #### 8.3. Training (available at the /Models/Model_Training_v2.2.py):

    * Lasted - ??? epochs.
    
    * Training log available at the /Logs/Model_v2.2_Training_Log.csv.

    * Results:
    
    | Model | Epoch | Task | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
    | --- | --- | --- | --- | --- | --- | --- |
    | v2.2 | ??? | Author Identification | 0.??? | 0.??? | ?.???| 0.??? |
    | v2.2 | ??? | Features Extraction | 0.??? | 0.??? | ?.???| 0.??? |
    
* #### 6.3. Testing (available at the /Models/Model_Testing_v2.0.py):

    | Model | Epoch | Task | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | v2.2 | ??? | Author Identification | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? |
    | v2.2 | ??? | Features Extraction | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? |

### 9. Model v2.3
### 10. Model v2.4
### 11. Model v2.5

### 12. Model v3.0

* #### 12.1. Architecture (available at the /Models/Model_Training_v3.0.py):

  * Vide 1.1.1. Architecture.

  * However:
  
    * There are two output layers, one tasked with author identification (27 neurons, softmax activation), and the other tasked with features identification (84 neurons, sigmoid activation).

* #### 12.2. Hyperparameteres (all training updates available at the /Logs/Model_v3.0_Training_Log.csv):

    * Vide 1.2. Hyperparameteres.

    * However:

       * Losses: 
         
         * Categorical crossentropy at the author-identification layer (softmax activation);
         
         * Binary crossentropy at the features-extraction layer (sigmoid activation).
              
       * Initial dropout rate - 0.1.

* #### 12.3. Training (available at the /Models/Model_Training_v3.0.py):

    * Lasted - ??? epochs.
    
    * Training log available at the /Logs/Model_v3.0_Training_Log.csv.

    * Results:
    
    | Model | Epoch | Task | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
    | --- | --- | --- | --- | --- | --- | --- |
    | v3.0 | ??? | Author Identification | 0.??? | 0.??? | ?.???| 0.??? |
    | v3.0 | ??? | Features Extraction | 0.??? | 0.??? | ?.???| 0.??? |
    
* #### 12.3. Testing (available at the /Models/Model_Testing_v2.0.py):

    | Model | Epoch | Task | Loss | Acc | TPR | TNR | FPR | FNR | PPV | NPV | AUC |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | v3.0 | ??? | Author Identification | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? |
    | v3.0 | ??? | Features Extraction | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? | 0.???? |
    
### 13. Model v3.1
### 14. Model v4.0    
