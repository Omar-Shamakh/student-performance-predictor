#!/usr/bin/env python
# coding: utf-8

# ### Before starting the project , i want really thank `EPSILON` for their efforts 
# I started my journey with them about a 9 month ago , i was very intesrested with my new journy as a step by step data scienest , I started my round with **ENG/Ahmed Noaman** and **ENG/Ayed Ali** but unfortunately I experienced some health conditions leading me to do two surgeries and making a freeze for the diploma , after I got well and come back from the freeze i met agreat instructor **ENG/Salah Tarek** who i want to realy thank him for his effort and of course **ENG/Mohab Allam** , today el hamdullah i am with good health condithions , i have graduated from my college with excellent degree , witnissing the end of the journey with epsilon but it is not an end it is just a start with another journey with them , Now i am doing an end to end data science project :) 

# ## LifeCycle of our project 
# - 1) Unterstanding the Problem Statemet
# - 2) Data Colleection
# - 3) Data Cleaning Phase
# - 4) Explotary Data Analysis (EDA)
# - 5) Feature Engineering
# - 6) EDA & Data Visualization
# - 7) data Preprocessing (DPP)
# - 8) Feature Selection
# - 9) Pick and Tune an Algorithm
# - 10) Validate and Evaluate
# - 11) Best Model Selection
# - 12) Project Deployment

# ### 1) Problem statement
# - This project understands how the student's performance (Maths scores) is affected by other variables such as Gender, Ethnicity, Parental level of education, Lunch and Test preparation course.
# 
# 
# ### 2) Data Collection
# - Dataset Source - https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977
# - The data set consists of 8 column (our main features).

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('student_performance_data.csv')
df.head()


# In[3]:


df.shape


# ## 3) Data Cleaning

# In[4]:


df.isnull().sum()


# In[5]:


df.duplicated().sum()


# In[6]:


df.dtypes


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.describe(include = "O")


# In[10]:


df.nunique()


# ## 4) Explotary Data Analysis (EDA)

# In[11]:


df.columns


# In[12]:


print(f"{df['gender'].nunique()} Unique value in gender ->{df['gender'].unique()}")
print(f"{df['race/ethnicity'].nunique()} unique value in race/ethnicity ->{df['race/ethnicity'].unique()}")
print(f"{df['parental level of education'].nunique()} Unique value in parental level of education ->{df['parental level of education'].unique()}")
print(f"{df['lunch'].nunique()} Unique value in lunch ->{df['lunch'].unique()}")
print(f"{df['test preparation course'].nunique()} Unique value in test preparation course ->{df['test preparation course'].unique()}")


# In[13]:


numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))


# ## 5) Feature Engineering

# In[14]:


df['total score'] = df['math score'] + df['reading score'] + df['writing score']
df['average'] = df['total score']/3
df.head()


# ## 6) EDA & Data Visualization

# In[15]:


fig, axs = plt.subplots(1,2,figsize=(14,6))
plt.subplot(121)
sns.histplot(data=df,x='average',bins=30,kde=True,color='g')
plt.subplot(122)
sns.histplot(data=df,x='average',kde=True,hue='gender')
plt.show()


# In[16]:


fig, ax =plt.subplots(1,2,figsize=(14,5))
plt.subplot(121)
sns.histplot(data=df, x='total score',kde=True, color='b')
plt.subplot(122)
sns.histplot(data=df, x='total score',kde=True,hue='gender')
plt.show()


# #### Insight
# - female studends do perform well then male students

# In[17]:


fig, ax =plt.subplots(1,3,figsize=(24,5))
plt.subplot(131)
sns.histplot(data=df ,x='average',kde=True, hue='lunch')

plt.subplot(132)
sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='lunch')

plt.subplot(133)
sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='lunch')
plt.show()


# ##### Insights
# - Standard lunch helps performe well in exam for both male and female student.

# In[18]:


plt.subplots(1,3,figsize=(25,6))
plt.subplot(131)
ax =sns.histplot(data=df,x='average',kde=True,hue='parental level of education',palette="viridis")
plt.title("All Genders")
plt.subplot(132)
ax =sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='parental level of education',palette="rocket")
plt.title("Male")
plt.subplot(133)
ax =sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='parental level of education',palette="mako")
plt.title("Female")
plt.show()


# #####  Insights
# - In general parent's education don't help student perform well in exam.
# - but we will check it agin in following analysis

# In[19]:


plt.subplots(1,3,figsize=(25,6))
plt.subplot(141)
ax =sns.histplot(data=df,x='average',kde=True,hue='race/ethnicity')
plt.subplot(142)
ax =sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='race/ethnicity')
plt.subplot(143)
ax =sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='race/ethnicity')
plt.show()


# #####  Insights
# - Students of group A and group B tends to perform poorly in exam.
# - Students of group A and group B tends to perform poorly in exam irrespective of whether they are male or female       

# #### Maximumum score of students in all three subjects

# In[20]:


plt.figure(figsize=(18,8))
plt.subplot(1, 4, 1)
plt.title('MATH SCORES')
sns.violinplot(y='math score',data=df,color='red',linewidth=3)
plt.subplot(1, 4, 2)
plt.title('READING SCORES')
sns.violinplot(y='reading score',data=df,color='green',linewidth=3)
plt.subplot(1, 4, 3)
plt.title('WRITING SCORES')
sns.violinplot(y='writing score',data=df,color='blue',linewidth=3)
plt.show()


# ### Multivariate analysis using pieplot

# In[21]:


plt.rcParams['figure.figsize'] = (30, 12)

plt.subplot(1, 5, 1)
size = df['gender'].value_counts()
labels = 'Female', 'Male'
color = ['red','green']

plt.pie(size, colors = color, labels = labels,autopct = '.%2f%%')
plt.title('Gender', fontsize = 20)

plt.subplot(1, 5, 2)
size = df['race/ethnicity'].value_counts()
labels = 'Group C', 'Group D','Group B','Group E','Group A'
color = ['red', 'green', 'blue', 'cyan','orange']

plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
plt.title('Race_Ethnicity', fontsize = 20)
plt.axis('off')



plt.subplot(1, 5, 3)
size = df['lunch'].value_counts()
labels = 'Standard', 'Free'
color = ['red','green']

plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
plt.title('Lunch', fontsize = 20)
plt.axis('off')


plt.subplot(1, 5, 4)
size = df['test preparation course'].value_counts()
labels = 'None', 'Completed'
color = ['red','green']

plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
plt.title('Test Course', fontsize = 20)
plt.axis('off')


plt.subplot(1, 5, 5)
size = df['parental level of education'].value_counts()
labels = 'Some College', "Associate's Degree",'High School','Some High School',"Bachelor's Degree","Master's Degree"
color = ['red', 'green', 'blue', 'cyan','orange','grey']

plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
plt.title('Parental Education', fontsize = 20)
plt.axis('off')


plt.tight_layout()
plt.grid()

plt.show()


# #####  Insights
# - Number of Male and Female students is almost equal
# - Number students are greatest in Group C
# - Number of students who have standard lunch are greater
# - Number of students who have not enrolled in any test preparation course is greater
# - Number of students whose parental education is "Some College" is greater followed closely by "Associate's Degree"

# #### GENDER COLUMN
# - How is distribution of Gender ?
# - Is gender has any impact on student's performance ?

# In[22]:


f,ax=plt.subplots(1,2,figsize=(20,10))
sns.countplot(x=df['gender'],data=df,palette ='bright',ax=ax[0],saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=20)
    
plt.pie(x=df['gender'].value_counts(),labels=['Male','Female'],explode=[0,0.1],autopct='%1.1f%%',shadow=True,colors=['#ff4d4d','#ff8000'])
plt.show()


# #### BIVARIATE ANALYSIS
# * Is gender has any impact on student's performance ? 

# In[23]:


gender_group = df.groupby(df['gender'])


# In[24]:


df


# In[25]:


plt.figure(figsize=(10, 8))

X = ['Total Average', 'Math Average']

# Assuming 'average' and 'math_score' are columns in your original DataFrame
female_scores = [gender_group.get_group('female')['average'].mean(), gender_group.get_group('female')['math score'].mean()]
male_scores = [gender_group.get_group('male')['average'].mean(), gender_group.get_group('male')['math score'].mean()]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, male_scores, 0.4, label='Male')
plt.bar(X_axis + 0.2, female_scores, 0.4, label='Female')

plt.xticks(X_axis, X)
plt.ylabel("Marks")
plt.title("Total average v/s Math average marks of both genders", fontweight='bold')
plt.legend()
plt.show()


# #### Insights 
# - On an average females have a better overall score than men.
# - whereas males have scored higher in Maths.    

# #### RACE/EHNICITY COLUMN
# - How is Group wise distribution ?
# - Is Race/Ehnicity has any impact on student's performance ?

# #### UNIVARIATE ANALYSIS 
# * How is Group wise distribution ?

# In[26]:


f,ax=plt.subplots(1,2,figsize=(20,10))
sns.countplot(x=df['race/ethnicity'],data=df,palette = 'bright',ax=ax[0],saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=20)
    
plt.pie(x = df['race/ethnicity'].value_counts(),labels=df['race/ethnicity'].value_counts().index,explode=[0.1,0,0,0,0],autopct='%1.1f%%',shadow=True)
plt.show()   


# #### Insights 
# - Most of the student belonging from group C /group D.
# - Lowest number of students belong to groupA.

# #### BIVARIATE ANALYSIS
# * Is Race/Ehnicity has any impact on student's performance ? 

# In[27]:


Group_data2=df.groupby('race/ethnicity')
f,ax=plt.subplots(1,3,figsize=(20,8))
sns.barplot(x=Group_data2['math score'].mean().index,y=Group_data2['math score'].mean().values,palette = 'mako',ax=ax[0])
ax[0].set_title('Math score',color='#005ce6',size=20)

for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=15)

sns.barplot(x=Group_data2['reading score'].mean().index,y=Group_data2['reading score'].mean().values,palette = 'flare',ax=ax[1])
ax[1].set_title('Reading score',color='#005ce6',size=20)

for container in ax[1].containers:
    ax[1].bar_label(container,color='black',size=15)

sns.barplot(x=Group_data2['writing score'].mean().index,y=Group_data2['writing score'].mean().values,palette = 'coolwarm',ax=ax[2])
ax[2].set_title('Writing score',color='#005ce6',size=20)

for container in ax[2].containers:
    ax[2].bar_label(container,color='black',size=15)


# #### Insights 
# - Group E students have scored the highest marks. 
# - Group A students have scored the lowest marks. 
# - Students from a lower Socioeconomic status have a lower avg in all course subjects

# #### PARENTAL LEVEL OF EDUCATION COLUMN
# - What is educational background of student's parent ?
# - Is parental education has any impact on student's performance ?

# #### UNIVARIATE ANALYSIS 
# * What is educational background of student's parent ? 

# In[28]:


plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('fivethirtyeight')

education_counts =df['parental level of education'].value_counts()

sns.barplot(x=education_counts.index, y=education_counts.values, palette='Blues')

plt.title('Comparison of Parental Education', fontweight = 30, fontsize = 20)
plt.xlabel('Degree')
plt.ylabel('count')
plt.show()


# #### Insights 
# - Largest number of parents are from some college.

# #### BIVARIATE ANALYSIS 
# Is parental education has any impact on student's performance ? 

# In[29]:


numeric_columns = df.select_dtypes(include=['number']).columns
df.groupby('parental level of education')[numeric_columns].mean().plot(kind='barh', figsize=(10, 10))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# #### Insights 
# - The score of student whose parents possess master and bachelor level education are higher than others.

# ####  LUNCH COLUMN 
# - Which type of lunch is most common amoung students ?
# - What is the effect of lunch type on test results?
# 
# #### UNIVARIATE ANALYSIS 
# * Which type of lunch is most common amoung students ? 

# In[30]:


plt.rcParams['figure.figsize'] = (15, 9)
sns.countplot(data=df ,x='lunch', palette = 'PuBu')
plt.title('Comparison of different types of lunch', fontweight = 30, fontsize = 20)
plt.xlabel('types of lunch')
plt.ylabel('count')
plt.show()


# #### Insights 
# - Students being served Standard lunch was more than free lunch

# #### BIVARIATE ANALYSIS 
# * Is lunch type intake has any impact on student's performance ? 

# In[31]:


plt = sns.countplot(x=df['parental level of education'],data=df,palette = 'bright',hue='lunch')
plt.set_title('Students vs test preparation course ',color='black',size=25)  


# #### Insights 
# - Students who get Standard Lunch tend to perform better than students who got free/reduced lunch

# #### MUTIVARIATE ANALYSIS USING PAIRPLOT

# In[32]:


sns.pairplot(df,hue = 'gender')


# #### Insights
# - From the above plot it is clear that all the scores increase linearly with each other.

# ## 7) data Preprocessing (DPP)

# #### CHECKING OUTLIERS

# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
sns.boxplot(y=df['math score'], color='skyblue')
plt.title('Math Score')

plt.subplot(1, 4, 2)
sns.boxplot(y=df['reading score'], color='hotpink')
plt.title('Reading Score')

plt.subplot(1, 4, 3)
sns.boxplot(y=df['writing score'], color='yellow')
plt.title('Writing Score')

plt.subplot(1, 4, 4)
sns.boxplot(y=df['average'], color='lightgreen')
plt.title('Average Score')

plt.tight_layout()
plt.show()


# **so ther are few outliers , let's deal with them** 

# In[34]:


import numpy as np
import pandas as pd

def detect_outliers(data, n=0, features=None):
    
    outlier_indices = []
    
    if features is None:
        features = data.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in features:
        Q1 = np.percentile(data[col], 25)
        Q3 = np.percentile(data[col], 75)
        IQR = Q3 - Q1
        
        outlier_step = 1.5 * IQR
        
        outlier_list_col = data[(data[col] < Q1 - outlier_step) | (data[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = pd.Series(outlier_indices)
    return outlier_indices[outlier_indices.duplicated(keep=False)].unique()

outliers_indices = detect_outliers(df)
print(f"Found {len(outliers_indices)} outliers")


# so just 6 outliers we can drop them 

# In[35]:


df.drop(outliers_indices, inplace=True)


# In[36]:


df.shape


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
sns.boxplot(y=df['math score'], color='skyblue')
plt.title('Math Score')

plt.subplot(1, 4, 2)
sns.boxplot(y=df['reading score'], color='hotpink')
plt.title('Reading Score')

plt.subplot(1, 4, 3)
sns.boxplot(y=df['writing score'], color='yellow')
plt.title('Writing Score')

plt.subplot(1, 4, 4)
sns.boxplot(y=df['average'], color='lightgreen')
plt.title('Average Score')

plt.tight_layout()
plt.show()


# ### Now we don't have any outliers 

# In[38]:


df.info()


# ## 8) Feature Selection

# In[39]:


df = df[['gender', 'race/ethnicity',  'parental level of education', 'lunch', 'test preparation course' , 'math score', 'reading score', 'writing score']]
df.head()


# **So let's prepare X & Y**

# In[40]:


X =df.drop(columns=['math score'],axis=1)

y =df['math score']


# In[41]:


X.head()


# In[42]:


print(f"{df['gender'].nunique()} categories in the Gender {df['gender'].unique()}")


# In[43]:


cat_column =df.select_dtypes(include='object').columns
for col in cat_column:
    print(f"{df[col].nunique()} categories in the {col} {df[col].unique()}")


# In[44]:


numerical_colums= X.select_dtypes(exclude ='object').columns
numerical_colums


# **let's encode categorical columns**

# In[45]:


from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer

ohe =OneHotEncoder()
stdsclr =StandardScaler()
processor = ColumnTransformer(
    [
        ('OneHotEncoder',ohe,cat_column),
        ('StandardScaler',stdsclr ,numerical_colums),
    ]
)


# ## 9) Pick and Tune an Algorithm

# In[46]:


X =processor.fit_transform(X)


# In[47]:


print(X)


# In[48]:


from sklearn.model_selection import train_test_split


# In[49]:


x_train,x_test,y_train,y_test = train_test_split(X, y, random_state=42,test_size=0.2,)
x_train.shape, x_test.shape


# In[50]:


###############################################################


# In[51]:


from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# In[52]:


models = {
    "LR": LinearRegression(),
    "KNNR" : KNeighborsRegressor(), 
    "Lasso":Lasso(),
    "SVR": SVR(),
    "DT": DecisionTreeRegressor(),
    "RF": RandomForestRegressor(),
    "XGBR": XGBRegressor()
}


# ## 10) Validate and Evaluate

# In[53]:


for name, model in models.items():
    print(f'Using model: {name}')
    model.fit(x_train, y_train)
    print(f'Training Score: {model.score(x_train, y_train)}')
    print(f'Test Score: {model.score(x_test, y_test)}')
    y_pred = model.predict(x_test)
    print(f'MSE Score: {np.sqrt(mean_squared_error(y_test, y_pred))}')  
    print(f'R2 Score: {r2_score(y_test, y_pred)}') 
    print('-'*30)


# In[54]:


# this will help in our deployment
import joblib
from sklearn.pipeline import Pipeline

# Save the ENTIRE pipeline (preprocessor + model)
pipeline = Pipeline([
    ('preprocessor', processor),  
    ('model', model)              
])

joblib.dump(pipeline, 'pipeline.pkl')  


# ## 11) Best Model Selection

# In[55]:


results = []

for name, model in models.items():
    print(f'Using model: {name}')
    model.fit(x_train, y_train)
    
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Train Score': train_score,
        'Test Score': test_score,
        'RMSE': rmse,
        'R2': r2
    })
    
    print(f'Training Score: {train_score}')
    print(f'Test Score: {test_score}')
    print(f'RMSE: {rmse}')
    print(f'R2 Score: {r2}')
    print('-' * 30)

results_df = pd.DataFrame(results)

results_df_sorted = results_df.sort_values('Test Score', ascending=False)

print("\n=== Models Ranked by Test Score (Descending) ===")
print(results_df_sorted)


# # So Our Champion Model is : Linear Regression

# In[56]:


from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model = model.fit(x_train, y_train)


# In[57]:


y_predict = model.predict(x_test)
r2_Score = r2_score(y_test, y_predict)*100
print(f"accuracy is {round(r2_Score,2)}%")


# In[58]:


plt.scatter(y_test,y_predict)
plt.xlabel("True value")
plt.ylabel("Predicted value")


# In[59]:


sns.regplot(x=y_test, y=y_predict,ci=None ,color='blue')


# ## So let's try to optimize our model 

# **Feature Engineering**
# * Scale features
# * Add polynomial features

# In[60]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[61]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x_train)


# **Regularization**

# In[62]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.5).fit(x_train, y_train)


# **GridSearchCV**

# In[63]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

# Hyperparameters to test
param_grid = {
    'fit_intercept': [True, False],  # Test with/without intercept
    'positive': [True, False]        # Force positive coefficients 
}


# In[64]:


# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=LinearRegression(),  
    param_grid=param_grid,         
    cv=5,                          # 5-fold cross-validation
    scoring='neg_mean_squared_error',  # Metric to optimize (RMSE)
    verbose=1                      # Print progress
)

grid_search.fit(x_train, y_train)


# In[65]:


# Best hyperparameters
print("Best parameters:", grid_search.best_params_)

# Best model (automatically refitted on full training data)
best_model = grid_search.best_estimator_

# Performance on test set
y_pred = best_model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.2f}")


# ## Compare Default vs. Tuned Model

# In[66]:


# Default model
default_model = LinearRegression()
default_model.fit(x_train, y_train)
default_rmse = np.sqrt(mean_squared_error(y_test, default_model.predict(x_test)))

# Tuned model
tuned_rmse = np.sqrt(mean_squared_error(y_test, best_model.predict(x_test)))

print(f"Default RMSE: {default_rmse:.2f}")
print(f"Tuned RMSE: {tuned_rmse:.2f}")


# **Model Deployment**

# In[67]:


feature_names = ['gender', 'race/ethnicity',  'parental level of education', 'lunch', 'test preparation course' , 'reading score', 'writing score'] 

joblib.dump(model, 'model.h5')          
joblib.dump(scaler, 'scaler.h5')        
joblib.dump(feature_names, 'features.h5')  


# # 12) Project Deployment

# In[69]:


get_ipython().run_cell_magic('writefile', 'app.py', 'import streamlit as st\nimport pandas as pd\nimport numpy as np\nimport joblib\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\n\n# Set up the app\nst.set_page_config(page_title="Student Performance Predictor", layout="wide")\nst.title("Student Math Score Prediction")\nst.write("This app predicts a student\'s math score based on various factors")\n\n# --- Input Widgets ---\nst.sidebar.header("Student Information")\n\n# Define categorical options (must match training data exactly)\ngender_options = [\'female\', \'male\']\nethnicity_options = [\'group A\', \'group B\', \'group C\', \'group D\', \'group E\']\nparental_education_options = ["some high school", "high school", "some college", \n                            "associate\'s degree", "bachelor\'s degree", "master\'s degree"]\nlunch_options = [\'standard\', \'free/reduced\']\ntest_prep_options = [\'none\', \'completed\']\n\n# Create input widgets\ngender = st.sidebar.selectbox("Gender", gender_options)\nethnicity = st.sidebar.selectbox("Race/Ethnicity", ethnicity_options)\nparental_education = st.sidebar.selectbox("Parental Education Level", parental_education_options)\nlunch = st.sidebar.selectbox("Lunch Type", lunch_options)\ntest_prep = st.sidebar.selectbox("Test Preparation Course", test_prep_options)\nreading_score = st.sidebar.slider("Reading Score", 0, 100, 70)\nwriting_score = st.sidebar.slider("Writing Score", 0, 100, 70)\n\n# --- Prediction Function ---\ndef predict_math_score():\n    try:\n        # Load the full pipeline\n        pipeline = joblib.load(\'pipeline.pkl\')\n        \n        # Create input DataFrame (column names must match training data)\n        input_data = pd.DataFrame({\n            \'gender\': [gender],\n            \'race/ethnicity\': [ethnicity],\n            \'parental level of education\': [parental_education],\n            \'lunch\': [lunch],\n            \'test preparation course\': [test_prep],\n            \'reading score\': [reading_score],\n            \'writing score\': [writing_score]\n        })\n        \n        # Make prediction (pipeline handles all preprocessing)\n        prediction = pipeline.predict(input_data)[0]\n        st.success(f"Predicted Math Score: {prediction:.1f}")\n        \n    except Exception as e:\n        st.error(f"Prediction failed: {str(e)}")\n\n# --- Run Prediction ---\nif st.sidebar.button("Predict Math Score"):\n    predict_math_score()\n\n# --- Explanatory Sections ---\nst.header("How It Works")\nst.markdown("""\nThis model predicts math scores using:\n- **Demographics**: Gender, ethnicity\n- **Background**: Parental education, lunch type\n- **Scores**: Reading and writing marks\n- **Preparation**: Test prep course status\n""")\n\nst.header("Key Insights")\ncol1, col2 = st.columns(2)\n\nwith col1:\n    st.subheader("Performance by Gender")\n    st.markdown("""\n    - 🚺 Females score higher in reading/writing\n    - 🚹 Males score higher in math\n    """)\n\nwith col2:\n    st.subheader("Lunch Impact")\n    st.markdown("""\n    - 🍎 Standard lunch → Higher scores\n    - 🥗 Free/reduced → Slightly lower scores\n    """)\n\n# --- Batch Prediction Section ---\nst.header("Batch Predictions")\nuploaded_file = st.file_uploader("Upload CSV for multiple predictions", type=["csv"])\n\nif uploaded_file is not None:\n    try:\n        batch_data = pd.read_csv(uploaded_file)\n        pipeline = joblib.load(\'pipeline.pkl\')\n        \n        # Check required columns\n        required_cols = [\'gender\', \'race/ethnicity\', \'parental level of education\',\n                       \'lunch\', \'test preparation course\', \'reading score\', \'writing score\']\n        \n        if all(col in batch_data.columns for col in required_cols):\n            predictions = pipeline.predict(batch_data)\n            batch_data[\'predicted_math_score\'] = predictions.round(1)\n            \n            st.dataframe(batch_data)\n            \n            csv = batch_data.to_csv(index=False).encode(\'utf-8\')\n            st.download_button(\n                "Download Predictions",\n                data=csv,\n                file_name=\'math_score_predictions.csv\',\n                mime=\'text/csv\'\n            )\n        else:\n            missing = [col for col in required_cols if col not in batch_data.columns]\n            st.error(f"Missing columns: {\', \'.join(missing)}")\n            \n    except Exception as e:\n        st.error(f"Batch prediction failed: {str(e)}")')


# In[71]:


get_ipython().system('streamlit run app.py')


# # THANKS EPSILON :)
