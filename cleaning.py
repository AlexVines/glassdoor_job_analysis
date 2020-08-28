import pandas as pd
import numpy as np


def title_simplifier(title):
    if 'data scientist' in title.lower():
        return 'data scientist'
    elif 'data engineer' in title.lower():
        return 'data engineer'
    elif 'analyst' in title.lower():
        return 'analyst'
    elif 'machine learning' in title.lower():
        return 'mle'
    elif 'manager' in title.lower():
        return 'manager'
    elif 'director' in title.lower():
        return 'director'
    else:
        return 'na'


def seniority(title):
    if 'sr' in title.lower() or 'senior' in title.lower() or 'sr' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower():
        return 'senior'
    elif 'jr' in title.lower() or 'jr.' in title.lower():
        return 'jr'
    else:
        return 'na'

desired_width = 20000
df = pd.read_csv(r'C:\Users\Alex\PycharmProjects\glassdoor_job_analysis\glassdoor_jobs.csv')
pd.set_option('display.width', desired_width)
pd.set_option("display.max_columns", 10)

#Salary parsing
df = df[df['Salary Estimate'] != '-1']
df['Salary Estimate'] = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_KD = df['Salary Estimate'].apply(lambda x: x.replace('K', '').replace('$', ''))

df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)

min_hour =minus_KD.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:', ''))

df['min_salary'] = min_hour.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hour.apply(lambda x: int(x.split('-')[1]))
df['avg'] = (df['min_salary'] + df['max_salary'])/2

#Company name
df['company_txt'] = df['Company Name'].apply(lambda x: x.split('\n')[0])

#State field

df['State'] = df['Location'].apply(lambda x: x.split(',')[1])

#Company age

df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis=1)
df['age'] = df.Founded.apply(lambda x: x if x <1 else 2020-x)

#Parsing Job Descr
#Searching for most used tools (Python, R, Spark, aws, Excel)

df['Python'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)

df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)

df['Excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)

df['Spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)

df['R'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)

df['job_simp'] = df['Job Title'].apply(title_simplifier)
df['Seniority'] = df['Job Title'].apply(seniority)
df['descr_len'] = df['Job Description'].apply(lambda x: len(x))

df.drop(['Unnamed: 0'], axis=1, inplace=True)

df.to_csv('Cleaned data.csv', index=False)
print(df.columns)
#print(df['aws'].value_counts())

