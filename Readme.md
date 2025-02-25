# Heart Disease Prediction using Support Vector Machines (SVM)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0-red)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

A machine learning project to predict heart disease using SVM classification on clinical data.

## Project Structure
```
Heart_SVM_Clearfinal.ipynb
heart.csv
README.md
```

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

Install requirements:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Dataset
The `heart.csv` dataset contains 13 clinical features and 1 target variable:

| Feature    | Description                          |
|------------|--------------------------------------|
| age        | Age in years                         |
| sex        | Gender (1=male, 0=female)            |
| cp         | Chest pain type                      |
| trestbps   | Resting blood pressure (mmHg)        |
| chol       | Serum cholesterol (mg/dl)            |
| fbs        | Fasting blood sugar > 120 mg/dl      |
| restecg    | Resting electrocardiographic results |
| thalach    | Maximum heart rate achieved          |
| exang      | Exercise induced angina              |
| oldpeak    | ST depression induced by exercise    |
| slope      | Slope of peak exercise ST segment    |
| ca         | Major vessels colored by fluoroscopy |
| thal       | Thalassemia type                     |
| target     | Presence of heart disease (0/1)      |

## Usage
1. Clone repository
2. Start Jupyter Notebook:
```bash
jupyter notebook
```
3. Open `Heart_SVM_Clearfinal.ipynb`
4. Run all cells sequentially

Key steps:
- Data loading and inspection
- Train/test split (80/20)
- Feature scaling with StandardScaler
- SVM model training (RBF kernel, C=100, gamma=1)
- Model evaluation:
  - Accuracy scores
  - Confusion matrix
  - Classification report
  - 5-fold cross-validation
  - Learning curves

## Results
- Training Accuracy: 100%
- Testing Accuracy: 100%
- Cross-validated Accuracy: 98.54% ± 2.34%
- Learning curves show optimal performance with full dataset

## Model Details
**Algorithm:** Support Vector Machine (RBF Kernel)  
**Hyperparameters:**
- Regularization (C): 100
- Kernel Coefficient (gamma): 1

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Cross-validation
- Learning Curves

## Future Work
- Hyperparameter tuning with GridSearchCV
- Experiment with other classifiers (Random Forest, Logistic Regression)
- Feature importance analysis
- Deploy as web application
- Handle class imbalance (if present)

## License
[MIT License](LICENSE)

## Contribution
Contributions welcome! Please open an issue or PR for suggestions.