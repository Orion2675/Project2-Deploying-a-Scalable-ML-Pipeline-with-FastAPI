# Model Card - Salary Predictions

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* Uses RandomForestClassifier from scikit-learn v1.5.1.
* Default Hyperparameters were used.
* min_samples_split was changed from default value of 2 to 10. 

## Intended Use
The intended use of this model to is predict whether or not a persons income exceeds 50K a year based on US Census data.    
Examples of the various demographic features within the dataset that were used for classification are: age, working class, education, marital status, occupation, sex, etc.   
This is model is not intended for production use as it is created for educational purposes only, for the Udacity Deploying a Scalable ML Pipeline with FastAPI project.   
The purpose of the project is to illustrate the ability to deploy a ML Model using FastAPI.

## Training Data
The model was trained on data obtains from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/20/census+income). The data is located in data/census.csv.   
The entire dataset was used for training and evaluation.

## Evaluation Data
The model was trained by splitting the data >census.csv into separate train and test subsets.

## Metrics
The following metrics were used for evaluation: Precision, Recall, and f1-score.    
The trained model performance metrics were computed as follows:   

Precision: 0.7721 | Recall: 0.6335 | F1: 0.6960

The evaluation based of the various model slices is recorded in the following file: slice_output.txt

## Ethical Considerations
The model most likely contains bias based on the skewing found within race and sex, with white males being largely represented more than other races and sexes.   
As this is for educational purposes only, this model should not be used for any sort of representation based on population and attributes within.

## Caveats and Recommendations
None
