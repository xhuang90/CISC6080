# Author: Xinhong Huang
# CISC 6080 Capstone


import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter
import time


############### PREPROCESS ###############
def explor_df(df):

    # number of instances and attributes
    num_of_ins = df.shape[0]
    num_of_att = df.shape[1]
    print('There are totally', num_of_att, 'attibutes.')
    print('They are ', end = '')
    for att_name in df:
        if att_name != 'y':
          print(att_name, end = ', ')
    print()
    print('There are totally', num_of_ins, 'instances.')

    # create a id column to identify each customer
    df['id'] = range(1, len(df) + 1)
    df = df.set_index('id')

    # 'unknown' category description
    att_unk = []
    att_unk_per =[]
    att = []

    for att_name in df:
        att.append(att_name)
        num_unk = 0
        for data in df[att_name]:
            if data == 'unknown':
                num_unk += 1
        unk_per = format((num_unk / num_of_ins) * 100, '.2f')
        att_unk.append(num_unk)
        att_unk_per.append(unk_per)


    unk_df = pd.DataFrame(
        {
            'Attribute Name': att,
            'Number of unknown data': att_unk,
            'Missing Perc(%)': att_unk_per
        }
    )

    print(unk_df)

    # explore each variable
    for att in df:
        print()
        print(df[att].describe())
        print()
        print(df[att].value_counts())

    # explore the target variable
    df['y'].describe()


def fill_unknown_data(dataset):

    df = pd.read_csv(dataset)

    # replace target variable y 'yes' as 1 and 'no' as 0
    df['y'] = df['y'].replace('yes', 1)
    df['y'] = df['y'].replace('no', 0)

    # for personal info dataset,
    # job, marital, education, default, housing, loan get value "unknown"
    unk_att_lst = ['job', 'marital', 'education', 'default', 'housing', 'loan']

    unk_job_idx = []
    unk_mar_idx = []
    unk_edu_idx = []
    unk_def_idx = []
    unk_hous_idx = []
    unk_loan_idx = []

    for idx in range(len(df)):

        if df.loc[idx, 'job'] == 'unknown':
            unk_job_idx.append(idx)
        if df.loc[idx, 'marital'] == 'unknown':
            unk_mar_idx.append(idx)
        if df.loc[idx, 'education'] == 'unknown':
            unk_edu_idx.append(idx)
        if df.loc[idx, 'default'] == 'unknown':
            unk_def_idx.append(idx)
        if df.loc[idx, 'housing'] == 'unknown':
            unk_hous_idx.append(idx)
        if df.loc[idx, 'loan'] == 'unknown':
            unk_loan_idx.append(idx)

    # find similar customer and impute unknown values
    # by considering relationship among correlated values among the attributes of the dataset.

    # for unknown job value
    # with experience, we know job is usually related to education and age.
    # so we use education and age
    for unk_idx in unk_job_idx:

        edu = df.loc[unk_idx, 'education']
        age = df.loc[unk_idx, 'age']
        job_lst = []

        for idx in range(len(df)):
            if df.loc[idx, 'job'] != 'unknown' and df.loc[idx, 'education'] == edu \
                    and int(age - 3) <= int(df.loc[idx, 'age']) <= int(age + 3):

                job_lst.append(df.loc[idx, 'job'])

        job_counts = Counter(job_lst)
        job_to_fill = str(job_counts.most_common(1)[0][0])
        df.loc[unk_idx, 'job'] = \
            df.loc[unk_idx, 'job'].replace('unknown', job_to_fill)

    print(df['job'].value_counts())

    # for unknown education value
    # with experience, we know education is usually related to job.
    # so we use job
    for unk_idx in unk_edu_idx:

        job = df.loc[unk_idx, 'job']
        edu_lst = []

        for idx in range(len(df)):
            if df.loc[idx, 'education'] != 'unknown' and df.loc[idx, 'job'] == job:
                edu_lst.append(df.loc[idx, 'education'])

        edu_counts = Counter(edu_lst)
        edu_to_fill = str(edu_counts.most_common(1)[0][0])
        df.loc[unk_idx, 'education'] = \
            df.loc[unk_idx, 'education'].replace('unknown', edu_to_fill)

    print(df['education'].value_counts())

    # for unknown housing and loan value
    # we found the unknown instance is totally the same
    # with experience, we know credit and loan is usually related to job and marital.
    # so we use job and marital to fill these two column together
    for unk_idx in unk_hous_idx:

        job = df.loc[unk_idx, 'job']
        mar = df.loc[unk_idx, 'marital']
        hous_lst = []
        loan_lst = []

        for idx in range(len(df)):
            if df.loc[idx, 'housing'] != 'unknown' and df.loc[idx, 'loan'] != 'unknown' \
                    and df.loc[idx, 'job'] == job and df.loc[idx, 'marital'] == mar:

                hous_lst.append(df.loc[idx, 'housing'])
                loan_lst.append(df.loc[idx, 'loan'])

        hous_counts = Counter(hous_lst)
        hous_to_fill = str(hous_counts.most_common(1)[0][0])
        df.loc[unk_idx, 'housing'] = \
            df.loc[unk_idx, 'housing'].replace('unknown', hous_to_fill)

        loan_counts = Counter(loan_lst)
        loan_to_fill = str(loan_counts.most_common(1)[0][0])
        df.loc[unk_idx, 'loan'] = \
            df.loc[unk_idx, 'loan'].replace('unknown', loan_to_fill)

    print(df['housing'].value_counts())
    print(df['loan'].value_counts())

    #df.to_csv('bank_wo_unknown.csv', index = False)


def split_att(dataset):

    df = pd.read_csv(dataset)

    # retrieve attributes age, job, marital, education, default, housing, loan,
    # contact, month, day_of_week, duration, campaign, pdays, previous, poutcome
    # as a subset
    df_sub1 = df.iloc[:, 1: 16]
    # df_sub1.insert(0, 'customer_id', range(1, len(df) + 1))
    df_sub1.insert(15, 'y', df['y'])
    df_sub1.to_csv('bank_sub1.csv', index = False)

    # retrieve attributes emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed
    # as econ index
    df_sub2 = df.iloc[:, 16: 21]
    df_sub2.insert(0, 'customer_id', range(1, len(df) + 1))
    df_sub2.insert(6, 'y', df['y'])
    df_sub2.to_csv('bank_sub2.csv', index = False)


def one_hot_coding(dataset):

    # deal with nominal features
    df = pd.read_csv(dataset)

    df['housing'] = df['housing'].replace('yes', 1)
    df['housing'] = df['housing'].replace('no', 0)

    df['loan'] = df['loan'].replace('yes', 1)
    df['loan'] = df['loan'].replace('no', 0)

    df['contact'] = df['contact'].replace('cellular', 0)
    df['contact'] = df['contact'].replace('telephone', 1)

    df['education'] = df['education'].replace('basic.4y', 'basic.4or6y')
    df['education'] = df['education'].replace('basic.6y', 'basic.4or6y')

    att_to_other = ['retired', 'entrepreneur', 'self-employed', 'housemaid', 'unemployed', 'student']
    for att in att_to_other:
        df['job'] = df['job'].replace(att, 'other')


    df_ohc = pd.get_dummies(df.loc[:, ['job', 'education', 'marital', 'default', 'day_of_week', 'month', 'poutcome']])
    df_to_concat = df.drop(['job', 'education', 'marital', 'default', 'day_of_week', 'month', 'poutcome'], axis = 1)

    df_new = pd.concat([df_ohc, df_to_concat], axis = 1, join_axes = [df_ohc.index])
    df_new.insert(0, 'customer_id', range(1, len(df) + 1))

    df_new.to_csv('bank_sub1_ohc.csv', index = False)


if __name__ == '__main__':

    # import data

    bank_data = pd.read_csv('bank-additional-full.csv')
    
    print('############### PREPROCESS ###############')
    print('Step1. Count values for each attributes.')
    #explor_df(bank_data)
    print()

    # fill unknown data
    print('Step2. Impute all unknown value.')
    beg_unk = time.time()
    fill_unknown_data('bank-additional-full.csv')
    end_unk = time.time()
    t_unk = end_unk - beg_unk
    print('All the unknown value are already filled.')
    print('Output file is bank_wo_unknown.csv')
    print('Using time: ' + str(t_unk) + 's.')
    '''
    # split the data set according the classification of feactures
    beg_spl = time.time()
    split_att('bank_wo_unknown.csv')
    end_spl = time.time()
    t_spl = end_spl - beg_spl
    print()
    print('Original data set is split into 2 subsets and are saved into the working directory.')
    print('Output files are bank_sub1.csv and bank_sub2.csv.')
    print('Using time: ' + str(t_spl) + 's.')


    # add id for each customer and one-hot-coding for dataset
    beg_ohc = time.time()
    one_hot_coding('bank_sub1.csv')
    end_ohc = time.time()
    t_ohc = end_ohc - beg_ohc
    print()
    print('All one-hot-coding files are saved into the working directory.')
    print('Output file is bank_sub1_ohc.csv.')
    print('Using time: ' + str(t_ohc) + 's.')

    
    df = pd.read_csv('econ_info.csv')
    for att in df:
        print()
        print(df[att].value_counts())
    '''
