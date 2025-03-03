# Heart Disease Prediction using Decision Trees and Random Forest

## Problem Statement
Coronary artery disease (CHD) is a leading cause of mortality worldwide. Early detection of CHD using non-invasive diagnostic methods can help improve patient outcomes. This short project aims to build predictive models using tree-based classifiers to determine the presence of CHD in patients based on non-invasive diagnostic test results and patient information.

## Dataset
The dataset used in this project is the Heart Disease dataset from the UCI Machine Learning Repository. Specifically, we use the Cleveland dataset, which consists of 303 observations with 13 features and one target attribute. The features represent various medical test results and demographic details, while the target variable, `num`, indicates the presence (1-4) or absence (0) of CHD.

### Data Preprocessing
- Converted `num` into a binary classification target variable `chd`, where 0 represents absence of CHD and 1 represents presence of CHD.
- Handled missing values using mean imputation.
- Encoded categorical variables (`ca` and `thal`) using one-hot encoding.
- Split the dataset into training (80%) and test (20%) sets.

## Approach
1. **Decision Tree Classifier (DTC):** A single decision tree was trained on the dataset to classify patients into CHD-positive or CHD-negative categories. Feature importance was extracted to understand the influence of different factors on the prediction.
2. **Random Forest Classifier (RFC):** An ensemble of decision trees was trained to improve model accuracy and reduce overfitting. Feature importance was compared with that of the DTC model.
3. **Feature Importance Analysis:** The top predictive features from both models were analyzed to identify the most influential medical attributes.

## Results
- **Decision Tree Classifier (DTC)** achieved an accuracy of 85% on the test set.
- **Random Forest Classifier (RFC)** improved the accuracy to 89%, demonstrating better generalization.
- **Feature Importance:** The most important features identified in both models were:
  - **cp (chest pain type)**
  - **oldpeak (ST depression induced by exercise)**
  - **thalach (maximum heart rate achieved)**
  - **age**
  - **thal_7.0 (thalassemia test result)**

## Future Work
- **Hyperparameter Tuning:** Optimize decision tree depth, number of trees in the random forest, and feature selection to improve accuracy.
- **Additional Models:** Experiment with other classifiers such as Support Vector Machines (SVM) and Gradient Boosting Trees.
- **Feature Engineering:** Introduce new features derived from existing attributes to capture more information.
- **Handling Imbalanced Data:** Apply techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset and improve model performance.
- **Cross-validation:** Use k-fold cross-validation to ensure model robustness.

## Project Structure
```
├── DecisionTreesImplementation.ipynb # Jupyter Notebook with code and analysis
├── README.md  # Project documentation
```

## Installation & Setup  
To run the notebooks, you will need Python and the required libraries installed.  

**Clone the repository:**  
   
   git clone https://github.com/Tzeene1459/Coronory-Heart-Disease-Classification-with-Decision-Trees.git
   cd Coronory-Heart-Disease-Classification-with-Decision-Trees

## Author 

Tazeen Shaukat