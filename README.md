AI In Drug Discovery:Multi-task binary classification model  
 
BY 
NIHITH REDDY 
KONDURU SAI CHETHAN
VISHNU RUDRARAJU
ALLU ARVIND
LALITH SRIKHAR BARLA


ABSTRACT 
 
The accurate classification of chemical compounds and drugs is crucial for advancing drug discovery and ensuring patient safety. This study investigates the application of machine learning (ML) and deep learning (DL) models on datasets such as hiv.csv, clintox.csv, tox21.csv, and sider.csv. Initially, Random Forest (RF) and Long Short-Term Memory (LSTM) models were employed for binary classification tasks, such as identifying HIV inhibitory activity, yielding satisfactory and comparable results. The study then explored multi-task classification using datasets containing chemical properties and FDA approval status. Various models, including Support Vector Machines (SVM), Recurrent Neural Networks (RNN), LSTMs, XGBoost, and AdaBoost, were evaluated. While LSTM excelled in handling sequential data, models like XGBoost and AdaBoost demonstrated superior performance on non-sequential datasets due to their flexibility and robustness. Among the ML models, SVM and XGBoost consistently outperformed others in classification accuracy. These findings highlight the potential of combining ML and DL approaches for efficient analysis of chemical datasets, offering valuable insights for bioinformatics and computational drug discovery. 

CONTENTS

Title page……………………………………………………………....…. 1

Acknowledgements………………………………………........................  2

Certificate……………………………………………………………........ 3

Abstract………………………………………………………………....... 4

Introduction………………………………………………....………......... 6

Problem statement……………………………………….…....….............  8

Chapters

3)Classification Using Random Forest on HIV Data ……………............... 8

4)Classification Using Random Forest on SIDER Data…….……...............13

5)LSTM-Based Classification on HIV Data..................................................16

6)Label-Specific Classification on Clintox Data ………..............................17

⁠7)Multitask Classification Using LSTM on Clintox  Data……………......21

8)Introduction to Adaboost..........................................................................27

9)Support Vector Machines (SVM) on the dataset clintox.csv.....................28
 
10)Introduction to xgboost..............................................................................31

11)Conclusion ……………………………………….....................................34
 
12)Future Work................................................................................................35

13)References…………………………………………………………..........36	

 
 
 
                                       1. Introduction 
Motivation for the Work
This work is driven by the urgent need to enhance the efficiency and accuracy of computational drug discovery. Traditional experimental methods are resource-intensive and time-consuming, limiting the pace of innovation in developing safer and more effective therapeutics. Automating the classification and analysis of molecular data enables faster early-stage drug development, minimizes the risk of adverse drug reactions, and accelerates the discovery of new compounds.
Our findings emphasize how different computational techniques perform optimally under varying conditions, offering a robust framework for researchers and practitioners. This research also paves the way for integrating advanced models, such as Graph Neural Networks (GNNs), to push the boundaries of molecular informatics.
Datasets like hiv.csv and clintox.csv exemplify real-world challenges in drug discovery. The hiv.csv dataset focuses on classifying compounds based on HIV inhibitory activity, while the clintox.csv dataset provides insights into FDA-approved drugs and clinical toxicity. These datasets are characterized by challenges such as imbalanced classes and intricate relationships among features. By employing a mix of machine learning methods, including SVM, XGBoost, and AdaBoost, alongside deep learning models like LSTMs, this study addresses these complexities and achieves improved classification accuracy, contributing to advancements in computational drug discovery.



The objectives of this study include:
1.	Exploring Traditional Approaches: Employing a Random Forest classifier on the hiv.csv and sider.csv dataset to establish baseline performance metrics.
2.	Enhancing with Deep Learning: Applying Long Short-Term Memory (LSTM) networks with 2-fold cross-validation to capture sequential patterns in the hiv.csv data for improved classification.
3.	Multitask Analysis: Implementing a comprehensive LSTM-based multitask classification on the clintox.csv dataset to analyse its overall predictive potential. We have also conducted multiclass classification on the provided datasets using RNN (Recurrent Neural Networks), Support Vector Machines (SVM), and Random Forest algorithms. The models were trained and evaluated to predict outcomes, and the results demonstrated varying levels of accuracy, highlighting the strengths and limitations of each approach for the given data.
These methodologies demonstrate the potential of machine learning and deep learning models in bioinformatics, with applications in drug discovery, toxicity assessment, and beyond:
1.	Drug Development: Accurate classification can streamline the identification of promising drug candidates.
2.	Safety Evaluation: Improved toxicity predictions reduce risks associated with drug approvals.
3.	Scalability: The techniques presented can be extended to other datasets and classification tasks in bioinformatics.
4.	Efficiency: Combining traditional and deep learning approaches offers a balance between computational cost and predictive performance.
The methodology employed in this study offers a robust framework for bioinformatics classification tasks, with significant implications for research and real-world applications in healthcare and drug discovery.


 
                                        
 
 


 2. Problem statement
The primary objective of this project is to address the challenges of classifying chemical compounds and drugs using bioinformatics datasets. The specific goals are as follows:
1.	Address Class Imbalance: We addressed class imbalance in datasets like HIV, SIDER, and Tox21 by integrating algorithms such as XGBoost, AdaBoost, and Support Vector Machines (SVM) with Synthetic Minority Oversampling Technique (SMOTE). SMOTE generated synthetic samples for the minority class, enhancing the dataset's balance. This approach improved the classifiers' ability to identify patterns in imbalanced datasets, leading to more accurate and robust predictive performance.
2.	Utilize the hiv.csv, clintox.csv, sider.csv,tox21.csv datasets, which feature imbalanced class distributions. Implement techniques to enhance the accuracy and robustness of classification models in handling these challenges.
3.	Evaluate Non-Sequential and Sequential Data Approaches:
a.	Leverage Support Vector Machines (SVM) and XGBoost for their effectiveness in handling non-sequential data, ensuring accurate classification in high-dimensional spaces.
b.	Apply Long Short-Term Memory (LSTM) networks to capture sequential patterns and improve classification accuracy, particularly for datasets with inherent temporal or structural dependencies.
c.	Incorporate AdaBoost to address non-linear relationships and enhance model performance on multi-class datasets.
4.	Model Training and Validation:
Train models using robust machine learning techniques (SVM, XGBoost, and AdaBoost) and deep learning methods (LSTM). Employ 2-fold cross-validation to ensure reliable and unbiased evaluation of model performance.

 
3. Classfication using random forest on HIV data set

The dataset contains 38,040 entries and 2 columns. Here is a brief description:
Columns:
1.	smiles:
o	A string being the molecular structure of compounds using the SMILES (Simplified Molecular Input Line Entry System) notation.
o	This is commonly used in cheminformatics to encode chemical structures.
2.	label:
o	An integer column with binary values (0 or 1), typically representing a classification label. For example, it could indicate whether the compound has a specific property (e.g., activity against HIV).

Sample Data:
Here are the first few rows:


smiles	label
CC1S(=O)(=O)OCCOS1(=O)=O	0
Nc1nc(Cl)c(N=Nc2ccc(Cl)cc2)c(NCC2(CO)CCC2)n1	0
CCOC(=O)C(=CNC(=S)Nc1c(CC)cccc1CC)C(=O)OCC	0
CC(C)(C)OC(=O)NCC(CNC(=O)OC(C)(C)C)C(N)=O	0
C1=CPH(c2ccccc2)[Pt-4]23([PH]1(c1c...)	0




2. Methodology for Applying Random Forest for Classification

To apply Random Forest for classification on the hiv.csv dataset, follow these steps:

Step 1: Data Preprocessing
•	Loading the Data: Import the dataset using libraries like pandas.
•	Handle Missing Data: Use imputation or drop rows/columns with missing values.
•	Categorical Data: Convert categorical columns (e.g., gender, ART use) into numerical formats using techniques like one-hot encoding or label encoding.
•	Feature Scaling: Some models benefit from scaling, but Random Forest is less sensitive to feature scaling.
•	Splitting the Data: Divide the dataset into training and testing sets (e.g., 80% training, 20% testing).
 
   Step 2: Train the Random Forest Classifier
•	Import the Random Forest Classifier from sklearn.ensemble. Train the classifier on the training data.  

Step 3: Make Predictions and Evaluate Performance
•	Use the trained model to predict the target variable on the test set.
•	Evaluate the model using metrics like accuracy, precision, recall, F1-score, and confusion matrix
 
 
3.	Results
The Random Forest model shows high overall accuracy (96.77%) and excellent performance on the majority class (0), with a precision of 0.97 and recall of 1.00. However, it struggles significantly with the minority class (1), achieving low precision (0.42) and recall (0.10). This indicates a class imbalance issue, where the model fails to detect many true positives for class 1. Improvements could involve addressing imbalance (e.g., oversampling, class weights) or tuning the model for better minority class performance. The below snippet briefly describes about the results we have got
 


                  4. Random forest classification on sider 21.csv

About the dataset sider.csv
The dataset sider.csv consists of 28 columns and contains information related to chemical compounds and their associated adverse drug reactions (ADRs). Here is a brief description of its structure:
Key Columns:
1.	Smiles: Represents the SMILES (Simplified Molecular Input Line Entry System) notation of chemical compounds, which is a standard way to describe a molecule's structure in text format.
2.	Adverse Reaction Categories: The remaining columns correspond to distinct categories of adverse drug reactions (ADRs) or medical conditions, such as:
o	Hepatobiliary disorders
o	Metabolism and nutrition disorders
o	Eye disorders
o	Gastrointestinal disorders
o	Psychiatric disorders
o	Nervous system disorders
o	And many more...
These columns are binary (0 or 1), where:
o	1 indicates the compound is associated with the corresponding adverse reaction.
o	0 indicates no association.

RESULTS :
 
1.	Top-Performing Classes:
o	The Hepatobiliary disorders, Metabolism and nutrition disorders, and Product issues have the highest ROC AUC scores, indicating better discrimination between positive and negative samples for these classes.
o	This suggests the model is relatively successful in predicting these classes, likely due to distinctive features in the molecular fingerprints.
2.	Low-Performing Classes:
o	Classes such as Injury, poisoning and procedural complications, Nervous system disorders, and Cardiac disorders have lower ROC AUC scores, indicating deficient performance.
o	This could be due to a lack of clear patterns in the molecular features or an imbalance in the dataset (e.g., fewer positive samples for these classes).
3.	Moderate Performance:
o	Most of the classes lie in the mid-range (AUC ~0.4–0.6). These scores indicate moderate discrimination capability but suggest room for improvement in feature extraction or model optimization.
4.	Imbalance in Class Performance:
o	The wide range of AUC scores (from extremely low to high) highlights variability in the model’s performance across different adverse drug reaction classes.





We have also obtained a confusion matrix also:

 
1.	True Positives (TP):
o	109 samples were correctly classified as positive (actual class = 1, predicted class = 1).
o	This indicates that the model is effective at identifying cases where this disorder is present.
2.	True Negatives (TN):
o	70 samples were correctly classified as negative (actual class = 0, predicted class = 0).
o	These are cases where the model successfully recognized the absence of the disorder.
3.	False Positives (FP):
o	43 samples were incorrectly classified as positive (actual class = 0, predicted class = 1).
o	This suggests the model tends to over-predict this class, which could lead to false alarms.
4.	False Negatives (FN):
o	27 samples were incorrectly classified as negative (actual class = 1, predicted class = 0).
o	These are missed cases where the model failed to detect the disorder, which can be a critical issue in medical applications.

Potential Uses:
1.	Drug Discovery:
o	Helps researchers understand potential side effects of chemical compounds.
o	Assists in filtering out compounds with high ADR risks early in the drug development process.
2.	Data Analysis and Machine Learning:
o	Can be used to train models to predict ADRs based on molecular structure.
o	Facilitates clustering and classification of compounds based on their side effect profiles.
3.	Pharmacovigilance:
o	Supports the study of post-market drug safety by correlating molecular structures with known side effects.
 




     









     




    5. LSTM-Based Classification on HIV Data 

1. Introduction to Long Short-Term Memory (LSTM) Networks
Long Short-Term Memory (LSTM) networks are a type of Recurrent Neural Network (RNN) designed to model sequential data. Unlike traditional RNNs, LSTMs can capture long-term dependencies due to their ability to retain information over time using memory cells and gating mechanisms (input, forget, and output gates). This makes LSTMs particularly effective for time-series data, text processing, and other sequences.
Although LSTMs are more commonly used in tasks involving sequences (e.g., time series prediction, natural language processing), they can also be used for classification tasks if the data is formatted appropriately, especially if there is an implicit sequence in the features or relationships over time
   2. Implementation of LSTM with 2-Fold Cross-Validation on hiv.csv
For classification, we can use an LSTM network with 2-fold cross-validation on the hiv.csv dataset. Here is how you would typically implement this:
Step 1: Data Preprocessing
•	Handle missing data, encode categorical variables, and normalize features for neural networks.
•	Reshape data to fit LSTM input requirements (i.e., 3D array samples, timesteps, features).
Step 2: LSTM Model Setup
•	Use Kera's or TensorFlow to build the LSTM model.
•	Implement 2-fold cross-validation to evaluate the model.
•	After training, we calculate the average accuracy over the two folds.
•	This process ensures the model's performance is generalized across different subsets of the data.
Results after lstm classification on hiv.csv dataset:
 



                  6. Label specific classification on clintox dataset

Label specific classification on the ClinTox dataset involves independently predicting two target variables: FDA approval status (fda_approved) and toxicity (ct_tox) of chemical compounds. Each classification task is performed using machine learning models, such as Random Forest, to evaluate the compounds' likelihood of being FDA-approved or toxic based on various molecular and experimental features.
1. Overview of the clintox.csv Dataset
The clintox.csv dataset is typically used in the context of drug discovery and toxicity prediction. It contains data related to the toxicity and FDA approval status of various chemical compounds, with features that may include chemical properties, experimental data, and molecular descriptors. The key columns in this dataset are often:
•	FDA_approved: This binary column indicates whether a compound has been approved by the FDA for use (1 for approved, 0 for not approved).
•	CT_tox: This binary column indicates whether a compound has been classified as toxic based on clinical trials or other toxicity tests (1 for toxic, 0 for non-toxic).
•	Other Features: Other columns may include molecular descriptors, experimental results, or chemical properties of the compounds.
2. Independent Classification of fda_approved and ct_tox Columns
In this task, we perform two separate classification tasks using machine learning models to predict:
•	Whether a compound is FDA approved (fda_approved column).

3. Methodologies for Classification  
Step 1 Data Processing
For both classification tasks, the preprocessing steps will be similar:
•	Handling Missing Values: Remove or impute missing data as needed.
•	Feature Engineering: Derive new features, if necessary, based on domain knowledge or exploration.
•	Encoding Categorical Variables: Encode categorical variables (if any) using techniques like One-Hot Encoding or Label Encoding.
•	Normalization/Scaling: Scale numerical features if needed (though Random Forest is less sensitive to this).
•	Whether a compound is classified as toxic (ct_tox column).
Step 2: Splitting the Data 
For both tasks, split the dataset into training and testing sets:
 
Step 3: Model Training with Random Forest
We can use Random Forest for both classification tasks. Since Random Forest works well for both binary classification and multiclass classification, it is a viable choice.
•	FDA Approval Classification:

 
Toxicity Classification: 
Step 4: Predictions 
     .FDA Approval Prediction:
 
. Toxicity Prediction:
 
Step 5: Model Evaluation
For both classifications, evaluate performance using common metrics:
Accuracy:
 
 Confusion matrix: 




Results for Each Column


The graph is attached is as follows:
 
Key Observations:
1.	CT_TOX:
o	Training Accuracy (blue): Stabilizes around 93.5%.
o	Validation Accuracy (orange): Slightly lower at approximately 93.0%, showing minimal overfitting.
o	
2.	FDA_APPROVED:
o	Training Accuracy (green): Stabilizes around 94.5%.
o	Validation Accuracy (red): Matches training accuracy closely, indicating excellent generalization.
Summary:
•	Both columns show high and stable accuracies with minimal gaps between training and validation, indicating good model performance and generalization to unseen data. This suggests the model is reliable for both tasks.












7. Multiclass Classification Using LSTM on ClinTox Data

The ClinTox dataset can be used to implement a multitask classification task if the target variable has more than two classes (e.g., classifying compounds into multiple categories such as safe, toxic, or partially safe). Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network (RNN), are effective for sequential or time-dependent data but can also work with datasets containing structured features if reformatted into sequences.
2. Implementation of Multitask LSTM Classification on clintox.csv

Step 1: Understanding the Dataset
The dataset likely contains:
•	Features: Molecular descriptors, chemical properties, or experimental results.
•	Target: A multiclass variable representing compound classification (e.g., safe, toxic, partially safe).
Step 2: Data Preprocessing
To prepare the dataset for LSTM:
1.	Load the Dataset
  
Encoding the Target Variable: Convert the multitask target into integer labels using Label Encoding or One-Hot Encoding
 
Sequence Preparation:
•	LSTMs require input in sequence format. If the features are not inherently sequential, reshape the data to simulate a sequence:
 
Train-Test Split:
 

One-Hot Encode the Target: Multitask classification requires one-hot encoded target labels:
 
 
Step 3: Build the LSTM Model
Define an LSTM architecture for the multitask classification task:
 
Step 4: Train the Model
Train the model on the training data:
 
Step 5: Evaluate the Model
Evaluate the model on the test data
 

The results for the lstm multitask classification are as follows: 
The confusion matrix for multi class classification for  Clintox dataset is as follows:
 This confusion matrix visualizes the performance of a multitask classification model on the ClinTox dataset. Here is a breakdown of what the matrix represents:
1.	Axes:
a.	Actual (rows): Indicates the true classes.
b.	Predicted (columns): Indicates the predicted classes by the model.
2.	Structure

a.	Each cell (i, j) represents the number of instances of class i that were predicted as class j.
3.	Interpreting Specific Values:

a.	Diagonal values (e.g., 268 in the centre): These are the correctly classified instances for their respective classes. High values on the diagonal indicate good classification performance.
b.	Off-diagonal values (e.g., 7, 8, 4, 2, etc.): These represent misclassifications. For instance, the 8 in the first row and second column means that 8 samples from class 0_1 were misclassified as class 1_0.
4.	Observations:
a.	The model predicts class 1_0 well (268 correct classifications), suggesting high accuracy for this class.
b.	Misclassifications are mostly seen in smaller numbers across other cells (e.g., 7, 8, 4, 2), indicating that the performance on other classes is less consistent.
5.	General Conclusion:
a.	The model performs well for the dominant class (1_0), likely due to a class imbalance if one class has significantly more samples in the dataset.
b.	Performance might need improvement for the other classes (0_1, 1_1).

The loss and the accuracy curve are as follows:
 
     The image shows two plots: a Loss Curve and an Accuracy Curve, which are commonly             used to evaluate the performance of a machine learning model during training and validation.    These curves represent the model's behaviour over multiple epochs when working on a      multiclass classification task (in this case, the Clintox dataset). Here is an explanation:
 
Loss Curve
Y-axis (Loss): The loss function measures the error between the predicted and true labels. Lower values indicate better performance.
X-axis (Epochs): Represents the number of training iterations.
Train Loss (Blue Line): This curve shows how the loss decreases for the training dataset as the model learns.
Validation Loss (Orange Line): This curve shows the loss for the validation dataset, which helps monitor the model's generalization to unseen data.
Interpretation:
 
Both training and validation accuracies improve over epochs, showing the model is effectively learning.
The validation accuracy closely follows the training accuracy, which indicates good generalization and minimal overfitting.
Toward the end, there are slight fluctuations in validation accuracy, which could imply slight instability or sensitivity in the model.
Key Takeaways
The curves suggest that the model performs well on both the training and validation sets.
There is minimal overfitting, but you might want to watch for slight overfitting signs as the validation loss increases slightly towards the end.
Fluctuations in validation accuracy might benefit from additional regularization (e.g., dropout or early stopping) or fine-tuning of hyperparameters.

3. Challenges and Approaches for Handling Multiclass Problems
Challenges:
1.	Class Imbalance: One class might dominate, leading to biased predictions.
o	Solution: Use techniques like oversampling (SMOTE) or class weighting during training.
2.	Feature Engineering for Sequences: LSTMs expect sequential data, but structured datasets may need to be reshaped.
o	Solution: Reshape the dataset into sequences or explore simpler models like Random Forest or Logistic Regression if LSTM is not ideal.
3.	Overfitting: LSTMs can be overfit with limited data.
o	Solution: Apply regularization techniques like Dropout and reduce model complexity.
4.	Model Complexity: LSTMs require more computational resources compared to simpler models.
o	Solution: Use optimized architectures and reduce the number of LSTM layers or
4. Performance Metrics and Interpretation of Results
To evaluate the performance of the multiclass LSTM model, use the following metrics:
1.	Accuracy:
o	Measures the overall correctness of predictions.
 
Precision, Recall, and F1-Score:
•	Assess the model's performance per class.

 

 Confusion Matrix:
•	Visualize how well the model distinguishes between different classes.
 
Cross-Entropy Loss:
•	The model minimizes this during training. Lower loss indicates better predictions



                                 8.  Introduction to AdaBoost
AdaBoost (Adaptive Boosting) is an ensemble learning algorithm that combines multiple weak learners (typically decision stumps) to form a strong classifier. It works by sequentially training weak learners, where each learner focuses on the misclassified instances of the previous one. AdaBoost assigns higher weights to misclassified samples, ensuring subsequent learners prioritize them.

Key Concepts in AdaBoost
1.	Boosting: Combines multiple weak learners to improve overall accuracy.
2.	Weighted Samples: Instances that are misclassified get higher weights, making the next learner focus on them.
3.	Decision Stumps: Simple decision trees (depth=1) are commonly used as weak learners.
4.	Adaptive Nature: Adapts to the difficulty of samples during training.

Generating Confusion Matrix and Classification Report:  
9.Support Vector Machines (SVM) Explained on the dataset clintox.csv
Introduction to SVM
Support Vector Machines (SVM) is a powerful supervised learning algorithm widely used for classification tasks. SVM works by finding an optimal hyperplane that separates data points belonging to different classes. It is especially effective in high-dimensional spaces and non-linearly separable datasets, thanks to kernel functions.

Key Concepts in SVM
- Hyperplane: A decision boundary separating classes.
 - Margin: The distance between the hyperplane and the nearest data points of either class.
 - Support Vectors: The data points closest to the hyperplane that influence its position.
 - Kernels: Functions to handle non-linear relationships by mapping data into higher dimensions.
One of the most popular kernel functions is the Radial Basis Function (RBF), which works well for non-linearly separable data. RBF kernel projects data into a higher-dimensional space, making it possible to classify using a linear hyperplane.


Code Example: Initializing and Training an SVM Model
from sklearn.svm import SVC

# Initialize SVM with RBF kernel and specific parameters
model = SVC(kernel='rbf', C=0.5, gamma=0.1, probability=True, random state=42)

# Train the model on training data
model.fit(Train, y_train)

Evaluating the Model:
Once the SVM model is trained, it is essential to evaluate its performance using metrics such as accuracy, precision, recall, and F1-score. The confusion matrix provides a detailed breakdown of true positives, true negatives, false positives, and false negatives.
 Generating Confusion Matrix Classification Report:
              precision    recall  f1-score   support

           0       0.83      1.00      0.91      1242
           1       0.50      0.06      0.11        85
           2       0.40      0.04      0.08       191

    accuracy                           0.82      1518
   macro avg       0.58      0.37      0.36      1518
weighted avg       0.76      0.82      0.76      1518 
 


                                 
                            10. Introduction to XGBoost
XGBoost (Extreme Gradient Boosting) is a powerful supervised learning algorithm commonly used for both classification and regression tasks. It is based on the gradient boosting framework, which builds models sequentially, with each model correcting the errors of its predecessor. XGBoost is highly efficient, flexible, and known for its speed and performance.

Key Concepts in XGBoost
1.	Gradient Boosting: Combines weak learners (e.g., decision trees) sequentially to minimize errors.
2.	Regularization: Includes L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting.
3.	Handling Missing Values: Efficiently handles missing data during training.
4.	Tree Pruning: Uses maximum depth to control complexity and avoid overfitting.
XGBoost also supports parallel processing and provides parameters like learning rate, maximum depth, and the number of estimators to tune performance.

Evaluating the Model
After training the model, evaluate its performance using metrics like accuracy, precision, recall, F1-score, and confusion matrix.
 
Generating Confusion Matrix and Classification Report
 
•	Accuracy: 91% overall correctness.
•	Precision, Recall, F1-Score: Provide insights into per-class performance.
•	Macro Average: Averages metrics across all classes equally.
•	Weighted Average: Averages metrics considering the class distribution.

Why XGBoost?
•	Outperforms many traditional algorithms on structured/tabular data.
•	Highly customizable for specific datasets and tasks.
•	Handles imbalanced data with techniques like scale_pos_weight.




   			             11.CONCLUSION

1.Support Vector Machine (SVM):
Among the machine learning models, SVM demonstrated superior performance in terms of classification accuracy and robustness across the datasets. Its capability to effectively handle high-dimensional feature spaces and small sample sizes contributed to its strong performance. SVM excelled particularly in scenarios where the data was well-separated and not overly complex in terms of sequential dependencies.
2.XGBoost:
XGBoost emerged as another top-performing machine learning model, showcasing its ability to handle non-sequential datasets with complex feature interactions. Its flexibility, robustness, and efficient handling of imbalanced classes made it a valuable tool for multi-class classification tasks, consistently outperforming other models in terms of speed and accuracy.

                                   12. FUTURE WORK
 This project successfully demonstrated the potential of AI in drug discovery by addressing challenges like class imbalance and limited labeled data. However, there are exciting opportunities for further research and innovation:
1. Incorporating Graph Neural Networks (GNNs):
Molecules can be represented as graphs, where GNNs like Graph Convolutional Networks (GCNs) can capture the structural relationships between atoms and bonds, providing a more accurate understanding of molecular properties.
2. Multi-Task Learning for Drug Discovery:
By training models to predict multiple outcomes (e.g., toxicity, FDA approval), multi-task learning can improve efficiency and leverage shared patterns across datasets.
3. Explainable AI (XAI):
Incorporating XAI can make model predictions more interpretable, helping researchers understand which molecular features drive classifications, a critical step for regulatory approval in the pharmaceutical industry.
4. Using Pre-trained Models:
Leveraging models like ChemBERTa or Molbert can accelerate progress by transferring knowledge from large molecular datasets, reducing training time and improving performance.
5. Real-Time Predictive Systems:  Developing lightweight, real-time systems optimized for speed and accuracy can revolutionize the initial stages of drug discovery, making AI tools more accessible to researchers.
 These directions offer a roadmap for scaling AI’s impact in drug discovery, paving the way for faster, safer, and more effective therapeutics.

                                     13.REFERENCES

1.	https://mahindraecolecentrale-my.sharepoint.com/:b:/r/personal/shampa_raghunathan_mahindrauniversity_edu_in/Documents/btech/proj/ai_drug/refs/2022_IJQC_122.e26870_Molecular.representations.for.machine.learning.applications.in.chemistry.pdf?csf=1&web=1&e=mNtlJQ
2.	https://medium.com/low-code-for-advanced-data-science/support-vector-machines-svm-an-intuitive-explanation-b084d6238106
3.	https://www.kaggle.com/code/navjindervirdee/lstm-neural-network-from-scratch
4.	https://www.youtube.com/watch?v=ok2s1vV9XW0


