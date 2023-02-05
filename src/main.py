import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils.columns_utils as cu

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#Reading data source
print(f"\n1. Reading Diabetes training data:")
df_diabetes = pd.read_csv("resource/diabetic_data.csv", index_col='encounter_id')
print(f"\nDiabetes datafrane info: (rows: {df_diabetes.shape[0]}, columns: {df_diabetes.shape[1]})\n")
print(f"Checking data has been read properly: First five rows:\n")
print(df_diabetes.head(5))

#Checking nullability:

print("2. Knowing data:\n")
print("2.1 Checking Nullabilty:\n")
cu.get_info_cleaing_process(df_diabetes)
print(f"Quantity null values (Nan or None): {df_diabetes.isnull().sum().sum()}\n")

#Knowing data:
print("2.2 Knowing columns data type (Text -Quantitative):")
print("2.2.1 Text columns:")
text_columns = cu.get_columns_type(df_diabetes.select_dtypes(include='object'))

print("2.2.1 Quantitative columns:")
quantitative_columns = cu.get_columns_type(df_diabetes.select_dtypes(exclude='object'))


# Categorical Variables Analysis
print("\n2.3 Categoric variables analysis:\n")
# Categorical variables must have a low number of categories.If the number of categories is big, the
# transformation could generate to many columns.

p_var_cat = [ 'race', 'gender', 'age', 'weight', 'payer_code', 'medical_specialty', 'diag_1',
		 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
		 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
         'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
		 'examide', 'citoglipton','insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
		 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted',
		 'patient_nbr', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
		 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency',
		 'number_inpatient', 'number_diagnoses']

df_diabetes[p_var_cat].astype('category')
print(df_diabetes[p_var_cat].nunique())
# Not Categorical Variables:
# age, weight, payer_code, medical_specialty, Diag_1,Diag_2 and Diag_3, patient_nbr, discharge_disposition_id
# admission_source_id, time_in_hospital,num_lab_procedures, num_medications, number_outpatient, number_emergency, number_inpatient, number_diagnoses
# 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
# 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
# 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
# 'examide', 'citoglipton','insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
# 'metformin-rosiglitazone', 'metformin-pioglitazone',
var_cat = [ 'race', 'gender',  'change', 'diabetesMed', 'readmitted', 'num_procedures',
		 'admission_type_id']

cu.distribution_by_category(var_cat,df_diabetes)

#readmitt convert N=0, >30=1, <30=2
df_diabetes ['readmitted_num'] = df_diabetes['readmitted'].str.slice(0, 1)
df_diabetes ['readmitted_num'] =  df_diabetes ['readmitted_num'].replace(['N'],0)
df_diabetes ['readmitted_num'] =  df_diabetes ['readmitted_num'].replace(['>'],1)
df_diabetes ['readmitted_num'] =  df_diabetes ['readmitted_num'].replace(['<'],2)
df_diabetes.drop(['readmitted'],axis=1, inplace=True)

numeric_dtypes_corr = df_diabetes.corr()
sns.heatmap(numeric_dtypes_corr, annot=True)
plt.show()

g = sns.catplot(x="race", y="readmitted_num", hue="gender", col="age", data=df_diabetes, kind="bar",palette='pink')
g.set_xlabels("race")
g.set_ylabels("readmitted")
plt.show()


#Cleaning:





#NOrmaliza?
df_diabetes_target_prediction_variables = pd.get_dummies(df_diabetes[[ 'race', 'gender',  'change', 'diabetesMed', 'readmitted', 'num_procedures',
		 'admission_type_id', 'readmitted_num','num_lab_procedures','time_in_hospital','num_procedures', 'num_medications', 'num_lab_procedures', 'number_diagnoses']])
print(df_diabetes_target_prediction_variables.head())

x = df_diabetes_target_prediction_variables
y = df_diabetes_target_prediction_variables['readmitted_num']

x.drop(['readmitted_num'],axis=1, inplace=True)


# Learning process
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state = 0)
lm_model = KNeighborsClassifier(n_neighbors=3)
lm_model.fit(X_train, y_train)

y_pred = lm_model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
