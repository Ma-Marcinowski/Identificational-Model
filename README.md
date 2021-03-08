## Identificational-Model

### 0. Introduction

* #### 0.1. Objective of the author's repository was to introduce multiple - varying in terms of interpretability - methods for identification of offline handwritten documents authors by artificial neural networks. The purpose of these approaches was to create an empirical background for analyses of interpretable machine learning tools developed in the field of computational forensics.

* #### 0.2. Versioning and identifiers (e.g. vX.Y.Z):

    * X indicates the model version (otherwise 0);
    * Y indicates the method of preprocessing (otherwise 0);
    * Z indicates any extra variation of the given X.Y base combination (otherwise omitted).

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
    
    * Databases spcific dataframes of features and categories are available at the /Categories/ folder (within other dataframes these features are labelled).
    
  * 0.5.3. Preprocessed data:

    * 0.5.3.1. Preprocessing v0.1 (grayscaled, undenoised, 256x256px):

      * Images are converted to grayscale, colour inverted, then rigid extraction of writing space is applied, reduction of extract dimensions to 1024 px on 1024 px, division of extracts into 256 px on 256 px patches, conversion from the tif to png format.
      
      * Patches which do not contain or contain only small amounts of text are omitted (vide /Preprocessing/Filters/ folder ).

      * Overall there were 3384 patches (2018 IAM and 1366 CLV).

  * 0.5.4. Dataframed data:

    * 0.5.4.1. Dataframing v0.1:

      * There is a column containing image paths and 145 one hot encoded authors' columns.
    
      * Around 80% of each author's document extracts or patches were utilized for models' training. 
    
      * Around 20% of each author's document extracts or patches were utilized for models' testing and validation.
