# Brain-Language Mapping with NLP Models
This project reproduces the results of the research paper [Interpreting and improving natural-language processing (in machines) with natural language-processing (in the brain)](https://arxiv.org/pdf/1905.11833) and extends it by testing the methodology on a new brain-language dataset.

### 1. Introduction
This project explores the connection between brain activity (fMRI data) and natural language representations learned by NLP models such as RoBERTa. The pipeline involves:

          -Extracting NLP model features.
          -Predicting brain responses using these features.
          -Evaluating the predictions against real brain data.
          -The significant contribution made to the original methodology is testing the pipeline on a new brain-language dataset to evaluate its performance in a different context.
### 2. Dependencies
The following tools and libraries are required:

          -Python 3.8 or above
          -PyTorch
          -NumPy
          -SciPy
          -scikit-learn
          -nibabel
          -Transformers (HuggingFace)
   Install the dependencies using:
   
           pip install -r requirements.txt

### 3. Dataset
Original Dataset
The [original dataset](https://drive.google.com/drive/folders/1Q6zVCAJtKuLOh-zWpkS3lH8LBvHcEOE8) is available. It contains fMRI scans and sentence annotations.

[New Dataset](https://openneuro.org/datasets/ds000204/versions/00002)

Description: The new dataset, titled "ds000204", is an fMRI dataset designed to explore brain activity in response to natural language processing tasks. Participants listened to short passages of text while their brain responses were recorded using functional MRI. The dataset includes NIfTI files representing fMRI scans, as well as annotations for the text stimuli. This dataset is useful for investigating the relationship between language representations and brain activity.

### 4. Steps to Reproduce Results
###### 4.1 Data Conversion
Convert the brain imaging data from NIfTI (.nii) to NumPy (.npy) format.

           -python new_dataset/converting_nii_npy.py
   
###### 4.2 Feature Extraction
Extract NLP model features using RoBERTa for sentence representations.

          -python extract_nlp_features.py --nlp_model roberta --sequence_length 4 --output_dir nlp_features
          
-nlp_model: NLP model to use (e.g., roberta, bert).
-sequence_length: The length of the input sequence.
-output_dir: Directory to save the extracted features.

###### 4.3 Prediction
Predict brain responses using the extracted NLP features.

          -python predictbrainfromnlp.py --subject 01 --nlp_feat_type roberta --nlp_feat_dir nlp_features --layer 6 --sequence_length 4 --output_dir OUTPUT_DIR

###### 4.4 Evaluation
Evaluate the predicted brain responses against the ground truth data.

         -python evaluate_brain_predictions.py --input_path OUTPUT_DIR/predict_01_with_roberta_layer_6_len_4.npy --output_path OUTPUT_DIR/evaluation_results --subject 01


### 5. Contribution: New Dataset
To extend the methodology, the entire pipeline was tested on a new brain-language dataset. This contribution evaluates the generalizability of the model in a different context.

Steps for using the new dataset:

          -Place the dataset in the new_dataset folder.
          -Run the same pipeline:
          -Convert NIfTI files to NumPy.
          -Extract NLP features.
          -Predict brain responses.
          -Evaluate predictions.

### 6. How to Run
   
Clone the Repository

         git clone <your_github_repository_link>
         cd brain_language_nlp
         
Prepare the Environment
Install the dependencies:

               pip install -r requirements.txt
Run the Pipeline

               python new_dataset/converting_nii_npy.py
Extract NLP features:

               python extract_nlp_features.py --nlp_model roberta --sequence_length 4 --output_dir nlp_features
Predict brain responses:

               python predictbrainfromnlp.py --subject 01 --nlp_feat_type roberta --nlp_feat_dir nlp_features --layer 6 --sequence_length 4 --output_dir OUTPUT_DIR
Evaluate predictions:

               python evaluate_brain_predictions.py --input_path OUTPUT_DIR/predict_01_with_roberta_layer_6_len_4.npy --output_path OUTPUT_DIR/evaluation_results --subject 01


#### Project Members
             * Sai Manoj Mekapati (200565197)
             * Tavleen Kaur (200573180)

