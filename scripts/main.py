import pandas as pd
from scripts.logger_config import *

# Set up the logger
logger = setup_logger()


def split_date(df):
    logger.info("Split date into Date, Year, and Month")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['WeekOfYear'] = df.Date.dt.isocalendar().week

def comp_months(df):
    logger.info("Calculate competition open months")
    df['CompetitionOpen'] = 12 * (df.Year - df.CompetitionOpenSinceYear) + (df.Month - df.CompetitionOpenSinceMonth)
    df['CompetitionOpen'] = df['CompetitionOpen'].map(lambda x: 0 if x < 0 else x).fillna(0)

def is_seasonal_holiday(StateHoliday,SchoolHoliday):
    if StateHoliday == 'a':
        return 'State Holiday'
    elif StateHoliday == 'b':
        return 'Easter Holiday'
    elif StateHoliday == 'c':
        return 'Christmas'
    elif SchoolHoliday == 1:
        return 'School Holiday'
    return 'Normal'

