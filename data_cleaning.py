import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from datetime import datetime
import pandas as pd
import numpy as np
import math
import json



def read_data():
    """Read in the json files and return all 3 DataFrames"""
    
    portfolio = pd.read_json('data/portfolio.json',
                             orient='records', lines=True)
    profile = pd.read_json('data/profile.json', orient='records', lines=True)
    transcript = pd.read_json('data/transcript.json',
                              orient='records', lines=True)
    return portfolio, profile, transcript


def clean_portfolio(portfolio):
    """Return clean portfolio DataFrame"""
    # one-hot encode channels
    mlb = MultiLabelBinarizer()
    channel_df = pd.DataFrame(mlb.fit_transform(portfolio.channels),
                              columns=mlb.classes_,
                              index=portfolio.index)
    portfolio = portfolio.merge(
        channel_df,
        left_index=True,
        right_index=True).drop(columns='channels')

    # one-hot encode offer_type
    offer_dummy = pd.get_dummies(portfolio.offer_type)
    portfolio = pd.concat([portfolio, offer_dummy], axis=1)
    portfolio.drop(columns='offer_type', inplace=True)

    return portfolio


def datetime_convert(x):
    """Return datetime object"""
    return datetime.strptime(str(x), '%Y%m%d')


def clean_profile(profile, today='2020-10-01'):
    """Return clean profile DataFrame"""
    # get rid of nulls
    profile = profile[~profile.gender.isnull()]

    # one-hot encode gender
    gender_dummy = pd.get_dummies(profile.gender, prefix='gender')
    profile = pd.concat([profile, gender_dummy], axis=1)
    profile.drop(columns='gender', inplace=True)

    # fix date columns
    profile['became_member_on'] = profile.became_member_on.apply(
        datetime_convert)
    profile['year'] = pd.DatetimeIndex(profile.became_member_on).year
    profile['month'] = pd.DatetimeIndex(profile.became_member_on).month

    # create member time & convert to int
    today = pd.to_datetime(today)
    profile['member_days'] = (today - profile.became_member_on)
    profile.member_days = profile.member_days.dt.days.astype('int16')

    return profile


def clean_transcript(transcript, profile):
    """Return clean transcript DataFrame"""
    # remove rows that do not have a profile
    transcript = transcript[transcript.person.isin(profile.id)]

    # create columns for the keys of value column
    transcript['value_key'] = transcript.value.apply(lambda x: list(x.keys()))
    # and a string version for later use
    transcript['value_key_str'] = transcript.value_key.astype(str)

    # link each offer completed id's to  transaction amount
    # first separate out the offer id's
    transcript['offer_id'] = transcript['value'].apply(
        lambda x: x['offer_id'] if 'offer_id' in x else (x['offer id'] if 'offer id' in x else None))
    # then attach an offer id to each transaction
    transcript['purchase_amount'] = transcript['value'].apply(
        lambda x: x['amount'] if 'amount' in x else (x['amount id'] if 'amount' in x else None))
    completed = pd.merge(transcript.loc[transcript['event'] == 'offer completed', 
                                        ['person', 'time', 'offer_id']],
                         transcript.loc[transcript['event'] == 'transaction', 
                                        ['person', 'time', 'purchase_amount']],
                         on=['person', 'time'])

    # make a separate dataframe to describe the process of receiving, 
    # and viewing an offer
    # get recieved and viewed offers and merge into one df
    viewed = pd.merge(transcript.loc[transcript['event'] == 'offer received', 
                                     ['person', 'time', 'offer_id']],
                      transcript.loc[transcript['event'] == 'offer viewed', 
                                     ['person', 'time', 'offer_id']],
                      on=['person', 'offer_id'], 
                      how='left', 
                      suffixes=['_received', '_viewed'])
    # remove those that were not viewed
    viewed = viewed[~viewed.time_viewed.isna()]
    # create time to view columns
    viewed['time_to_view'] = viewed.time_viewed - viewed.time_received

    # Since you can receive and complete an offer several times
    # we need to differentiate between each 'cycle' of receiving 
    # and completing an offer
    # use cumcount to get number of instances of each completion
    transcript.sort_values(by=['person', 'offer_id', 'time'], inplace=True)
    transcript['event_count'] = transcript.groupby(
        by=['person', 'offer_id', 'event']).cumcount()+1

    # create cycle id to identify each cycle of viewing and completing an offer
    transcript['cycle_id'] = transcript.event_count.map(
        str) + transcript.person + transcript.offer_id

    # get info for each offer cycle
    offers = transcript[transcript.event != 'transaction']
    offers = offers.pivot_table(values=['time'],
                                index=['cycle_id', 'person', 'offer_id'],
                                columns='event').reset_index()
    offers.columns = ['cycle_id', 'person', 'offer_id',
                      'offer_completed', 'offer_received', 'offer_viewed']
    offers = offers[['cycle_id', 'person', 'offer_id',
                     'offer_received', 'offer_viewed', 'offer_completed']]

    # join with amount data
    offers = offers.merge(completed,
                          how='left',
                          left_on=['person', 'offer_id', 'offer_completed'],
                          right_on=['person', 'offer_id', 'time'])
    offers.drop(columns='time', inplace=True)

    # check which transactions were influenced by an offer
    # if offer was both viewed and completed, then it was influenced
    offers['infl_trx'] = False
    offers.loc[offers.offer_viewed.notnull() & offers.offer_completed.notnull(), [
        'infl_trx']] = True

    return offers


def clean_data(portfolio, profile, transcript):
    """Further clean datasets and merge into one DataFrame so it is ready
    to be used by model. Returns cleaned DataFrame

    Args:
        portfolio ([DataFrame]): cleaned portfolio dataset
        profile ([DataFrame]): cleaned profil dataset
        transcript ([DataFrame]): cleaned transcript dataset
    """
    # merge offers with offer portfolio
    df = transcript.merge(portfolio,
                          how='left',
                          left_on='offer_id',
                          right_on='id').drop(columns='id')

    # merge offers with customer profiles
    df = df.merge(profile,
                  how='left',
                  left_on='person',
                  right_on='id')
    df.drop(columns='id', inplace=True)

    # create influenced/not influenced
    
    return df

if __name__ == '__main__':
    portfolio, profile, transcript = read_data()
    cl_portfolio = clean_portfolio(portfolio)
    cl_profile = clean_profile(profile)
    cl_transcript = clean_transcript(transcript, cl_profile)
    df = clean_data(cl_portfolio, cl_profile, cl_transcript)