import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def preprocess():
    #############################Import Dataset#####################################
    # Add headers to dataset
    headers = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
               'relationship',
               'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
    # 1)read data from csv
    # 2)split attributes with delimiter ","
    # 3)engine: python
    # 4)skip the first row
    income_train = pd.read_csv('data/adult.data', names=headers, delimiter=', ', engine='python')
    income_test = pd.read_csv('data/adult.test', names=headers, delimiter=', ', engine='python', skiprows=1)
    print(income_train.shape)
    print(income_test.shape)

    # concatenate train and test using column
    df = pd.concat([income_train, income_test], axis=0)
    df.reset_index(drop=True, inplace=True)

    df_ori = df

    #############################Missing Values#####################################
    df.replace('?', np.nan, inplace=True)
    # find the attribute with missing values
    print(df.isnull().sum())
    # fill the missing value with the mode of corresponding attribute
    df.fillna(value={'workclass': df.workclass.mode()[0], 'occupation': df.occupation.mode()[0],
                     'native_country': df.native_country.mode()[0]}, inplace=True)
    print(df.isnull().sum())
    df.head()

    #############################Duplicate and Outliers#####################################
    import seaborn as sns

    print(df.columns)
    educations = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
                  '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
    for i in educations:
        # Shows that education and education_num are duplicated in pairs
        print((df.loc[df['education'] == i, ['education', 'education_num']]).value_counts())

    plt.figure(figsize=(8, 6))
    relation = {'Bachelors': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'Some-college': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                '11th': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'HS-grad': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'Prof-school': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'Assoc-acdm': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'Assoc-voc': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                '9th': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                '7th-8th': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                '12th': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                'Masters': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                '1st-4th': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                '10th': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'Doctorate': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                '5th-6th': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                'Preschool': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], }
    # heatmap
    relation = pd.DataFrame(relation)
    relation.index = [13, 10, 7, 9, 15, 12, 11, 5, 4, 8, 14, 2, 6, 16, 3, 1]
    correlation_edu = sns.heatmap(relation, fmt=".2f", cmap='PuBu', linewidths='0.1')

    # drop attribute education
    df = df.drop(['education'], axis=1)
    headers.remove('education')

    # Use a boxplot to detect any outliers
    # df.boxplot(figsize=(16, 5), fontsize=20)
    df = df.drop(['fnlwgt'], axis=1)
    print(df.info())
    headers.remove('fnlwgt')

    #############################Labeling and Scaling#####################################
    # Label income and sex
    df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)  # income >50=1, <=50=0
    df['sex'] = df['sex'].apply(lambda x: 1 if x == 'Male' else 0)  # Male=1, Female=0

    # Converting string to int
    for c in df.columns:
        print("---- %s ---" % c)
        print(df[c].value_counts())

    df = pd.concat([df, pd.get_dummies(df['workclass'], prefix='workclass', prefix_sep=':')], axis=1)
    df.drop('workclass', axis=1, inplace=True)

    df = pd.concat([df, pd.get_dummies(df['marital_status'], prefix='marital_status', prefix_sep=':')], axis=1)
    df.drop('marital_status', axis=1, inplace=True)

    df = pd.concat([df, pd.get_dummies(df['occupation'], prefix='occupation', prefix_sep=':')], axis=1)
    df.drop('occupation', axis=1, inplace=True)

    df = pd.concat([df, pd.get_dummies(df['relationship'], prefix='relationship', prefix_sep=':')], axis=1)
    df.drop('relationship', axis=1, inplace=True)

    df = pd.concat([df, pd.get_dummies(df['race'], prefix='race', prefix_sep=':')], axis=1)
    df.drop('race', axis=1, inplace=True)

    df = pd.concat([df, pd.get_dummies(df['native_country'], prefix='Native country', prefix_sep=':')], axis=1)
    df.drop('native_country', axis=1, inplace=True)

    X = np.array(df.drop(['income'], 1))
    y = np.array(df['income'])
    # normalize
    X = preprocessing.scale(X)  # normalize

    return X, y
