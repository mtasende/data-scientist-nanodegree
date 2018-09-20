import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv('./survey_results_public.csv')
schema = pd.read_csv('./survey_results_schema.csv')

## A Look at the Data
# Solution to Question 1
num_rows = df.shape[0]
num_cols = df.shape[1]

# Solution to Question 2
no_nulls = set(df.columns[df.isnull().mean()==0])

# Solution to Question 3
most_missing_cols = set(df.columns[df.isnull().mean() > 0.75])

## How To Break Into the Field
# Solution to Question 1
def get_description(schema, column_name):
    '''
    INPUT - schema - pandas dataframe with the schema of the developers survey
            column_name - string - the name of the column you would like to know about
    OUTPUT -
            desc - string - the description of the column
    '''
    desc = list(schema[schema['Column'] == column_name]['Question'])[0]
    return desc

descrips = set(get_description(schema, col) for col in df.columns)

# Solution to Question 4
higher_ed = lambda x: 1 if x in ("Master's degree", "Doctoral", "Professional degree") else 0

df['HigherEd'] = df["FormalEducation"].apply(higher_ed)
higher_ed_perc = df['HigherEd'].mean()


# Solution to Question 6
sol = {'Everyone should get a higher level of formal education': False,
       'Regardless of formal education, online courses are the top suggested form of education': True,
       'There is less than a 1% difference between suggestions of the two groups for all forms of education': False,
       'Those with higher formal education suggest it more than those who do not have it': True}

# What Happened?
# Question 1
desc_sol = {'A column just listing an index for each row': 'Respondent',
       'The maximum Satisfaction on the scales for the survey': 10,
       'The column with the most missing values': 'ExpectedSalary',
       'The variable with the highest spread of values': 'Salary'}

# Question 2
scatter_sol = {'The column with the strongest correlation with Salary': 'CareerSatisfaction',
       'The data suggests more hours worked relates to higher salary': 'No',
       'Data in the ______ column meant missing data in three other columns': 'ExpectedSalary',
       'The strongest negative relationship had what correlation?': -0.15}

# Question 3
a = 'it is a way to assure your model extends well to new data'
b = 'it assures the same train and test split will occur for different users'
c = 'there is no correct match of this question'
d = 'sklearn fit methods cannot except NAN values'
e = 'it is just a convention people do that will likely go away soon'
f = 'python just breaks for no reason sometimes'

lm_fit_sol = {'What is the reason that the fit method broke?': d,
       'What does the random_state parameter do for the train_test_split function?': b,
       'What is the purpose of creating a train test split?': a}

## Job Satisfaction?
# Question 1
job_sol_1 = {'The proportion of missing values in the Job Satisfaction column': 0.2014,
             'According to EmploymentStatus, which group has the highest average job satisfaction?': 'contractors',
             'In general, do smaller companies appear to have employees with higher job satisfaction?': 'yes'}

# Question 2
job_sol_2 = {'Do individuals who program outside of work appear to have higher JobSatisfaction?': 'yes',
             'Does flexibility to work outside of the office appear to have an influence on JobSatisfaction?': 'yes', 
             'A friend says a Doctoral degree increases the chance of having job you like, does this seem true?': 'yes'}


# Removing Values

small_dataset = pd.DataFrame({'col1': [1, 2, np.nan, np.nan, 5, 6],
                             'col2': [7, 8, np.nan, 10, 11, 12],
                             'col3': [np.nan, 14, np.nan, 16, 17, 18]})
# Question 1
all_drop  = small_dataset.dropna()


# Question 2
all_row = small_dataset.dropna(axis=0, how='all')

# Question 3
only3_drop = small_dataset.dropna(subset=['col3'], how='any')

# Question 4
only3or1_drop = small_dataset.dropna(subset=['col1', 'col3'], how='any')


## Your Turn
# Question 1
prop_sals = 1 - df.isnull()['Salary'].mean()
# Question 2
num_vars = df[['Salary', 'CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction']]
sal_rm = num_vars.dropna(subset=['Salary'], axis=0)
# Question 3
question3_solution = 'It broke because we still have missing values in X'
# Question 4
all_rm = num_vars.dropna()
# Question 5
question5_solution = 'It worked, because Python is magic.'
# Question 6
r2_test = 0.019170661803761924
# Question 7
question7_solution = {'The number of reported salaries in the original dataset': 5009,
                       'The number of salaries predicted using our model': 645,
                       'If an individual does not rate stackoverflow, but has a salary': 'We still want to predict their salary',
                       'If an individual does not have a a job satisfaction, but has a salary': 'We still want to predict their salary',
                       'Our model predicts salaries for the two individuals described above.': False}


### Imputation Methods
#Question 1
question1_solution = {'Column A is': 'quantitative',
                      'Column B is': 'quantitative',
                      'Column C is': 'we cannot tell',
                      'Column D is': 'boolean - can treat either way',
                      'Column E is': 'categorical',
                     }

#Question 2
should_we_drop = 'Yes'

#Question 3
impute_q3 = {'Filling column A': "is no problem - it fills the NaN values with the mean as expected.",
             'Filling column D': "fills with the mean, but that doesn't actually make sense in this case.",
             'Filling column E': "gives an error."}

#Question 4
impute_q4 = {'Filling column A': "Did not impute the mode.",
             'Filling column D': "Did not impute the mode.",
             'Filling column E': "Imputes the mode."}

##Imputing Values
#Question 1 Part 1
#Drop the rows with missing salaries
drop_sal_df = num_vars.dropna(subset=['Salary'], axis=0)
#Question 1 Part 2
# Mean function
fill_mean = lambda col: col.fillna(col.mean())
# Fill the mean
fill_df = drop_sal_df.apply(fill_mean, axis=0)

# Question 2
rsquared_score = 0.03257139063404435
length_y_test = 1503



##Categorical Variables
# Question 1
cat_df = df.select_dtypes(include=['object'])
cat_df.shape[1]
#Question 2
cat_df_dict = {'the number of columns with no missing values': 6,
               'the number of columns with more than half of the column missing': 49,
               'the number of columns with more than 75% of the column missing': 13
}
#Question 3
sol_3_dict = {'Which column should you create a dummy variable for?': 'col1',
              'When you use the default settings for creating dummy variables, how many are created?': 2,
              'What happens with the nan values?': 'the NaNs are always encoded as 0'
             }
#Question 4
#create needed dataframe
dummy_var_df = pd.DataFrame({'col1': ['a', 'a', 'b', 'b', 'a', np.nan, 'b', np.nan],
                             'col2': [1, np.nan, 3, np.nan, 5, 6, 7, 8]
})
#dummy cols
dummy_cols_df = pd.get_dummies(dummy_var_df['col1'], dummy_na=True)

#Question 5
cat_cols_lst = cat_df.columns
def create_dummy_df(df, cat_cols, dummy_na):
    for col in  cat_cols:
        try:
            # for each cat add dummy var, drop original column
            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis=1)
        except:
            continue
    return df
df_new = create_dummy_df(df, cat_cols_lst, dummy_na=False)

#Question 6
def clean_fit_linear_mod(df, response_col, cat_cols, dummy_na, test_size=.3, rand_state=42):
    '''
    INPUT:
    df - a dataframe holding all the variables of interest
    response_col - a string holding the name of the column 
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    test_size - a float between [0,1] about what proportion of data should be in the test dataset
    rand_state - an int that is provided as the random state for splitting the data into training and test 
    
    OUTPUT:
    test_score - float - r2 score on the test data
    train_score - float - r2 score on the test data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    
    Your function should:
    1. Drop the rows with missing response values
    2. Drop columns with NaN for all the values
    3. Use create_dummy_df to dummy categorical columns
    4. Fill the mean of the column for any missing values 
    5. Split your data into an X matrix and a response vector y
    6. Create training and test sets of data
    7. Instantiate a LinearRegression model with normalized data
    8. Fit your model to the training data
    9. Predict the response for the training data and the test data
    10. Obtain an rsquared value for both the training and test data
    '''
    #Drop the rows with missing response values
    df  = df.dropna(subset=[response_col], axis=0)

    #Drop columns with all NaN values
    df = df.dropna(how='all', axis=1)

    #Dummy categorical variables
    df = create_dummy_df(df, cat_cols, dummy_na)

    # Mean function
    fill_mean = lambda col: col.fillna(col.mean())
    # Fill the mean
    df = df.apply(fill_mean, axis=0)

    #Split into explanatory and response variables
    X = df.drop(response_col, axis=1)
    y = df[response_col]

    #Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

    lm_model = LinearRegression(normalize=True) # Instantiate
    lm_model.fit(X_train, y_train) #Fit

    #Predict using your model
    y_test_preds = lm_model.predict(X_test)
    y_train_preds = lm_model.predict(X_train)

    #Score using your model
    test_score = r2_score(y_test, y_test_preds)
    train_score = r2_score(y_train, y_train_preds)

    return test_score, train_score, lm_model, X_train, X_test, y_train, y_test


#Test your function with the above dataset
test_score, train_score, lm_model, X_train, X_test, y_train, y_test = clean_fit_linear_mod(df_new, 'Salary', cat_cols_lst, dummy_na=False)


## Putting It All Together
#Question 2
q2_piat = {'add interactions, quadratics, cubics, and other higher order terms': 'no',
           'fit the model many times with different rows, then average the responses': 'yes',
           'subset the features used for fitting the model each time': 'yes',
           'this model is hopeless, we should start over': 'no'}
#Question 4
q4_piat = {'The optimal number of features based on the results is': 1088,
               'The model we should implement in practice has a train rsquared of': 0.80,
               'The model we should implement in practice has a test rsquared of': 0.73,
               'If we were to allow the number of features to continue to increase':
'we would likely have a better rsquared for the training data.'}

#Question 5
q5_piat = {'Country appears to be one of the top indicators for salary': True,
               'Gender appears to be one of the indicators for salary': False,
               'How long an individual has been programming appears to be one of the top indicators for salary': True,
               'The longer an individual has been programming the more they are likely to earn': False}


