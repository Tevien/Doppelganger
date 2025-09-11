import pandas as pd
import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def calculateMAGGIC(data:pd.DataFrame, columns:dict,

                    ) -> pd.DataFrame:

    '''
    Caculated the MAGGIC score predicting 3-years and 1-year death in heart failure
    patients. Based on Pocock et al. 2013 EHJ:
    https://academic.oup.com/eurheartj/article/34/19/1404/422939

    Parameters

    ----------

    data: pd.DataFrame,
        a pd.DataFrame with the `13` columns to calculate the MAGGIC score.
    columns: dictionary of strings,
        a dictionary where the key maps to `MAGGIC` variables, and
        the `value` maps to the `data` column. See example.

    Example
    -------
    Example `column` input:

    >>> maggic_columns = {
    #>>> 'sex'           : 'Female sex',  # binary 1 for female
    #>>> 'smoking'       : 'Smoker',      # binary 1 for current smoker
    #>>> 'hist_diabetes' : 'Diabetes',     # binary 1 for any diabetes
    #>>> 'hist_copd'     : 'HistoryCOPD', # binary 1 for COPD
    #>>> 'recent_hf'     : 'HF_first18M', # binary 1 HF diagnosis within the first 18M
    #>>> 'betablocker'   : 'Med_BB',      # binary 1 for BB medication
    #>>> 'ace_arb'       : 'Med_ACEi',    # binary 1 for ACE/ARB medication
    !(FROM FREE TEXT)>>> 'lvef'          : 'LV - EF (%)',
    !(FROM FREE TEXT)>>> 'nyha'          : 'NYHA class',
    #>>> 'creatinine'    : 'Creatinine (μmol/L)',
    #>>> 'bmi'           : 'BMI',
    #>>> 'sbp'           : 'systBP',
    #>>> 'age'           : 'Age (years)',
    }

    Returns
    -------
    maggic : pd.DataFrame,
        a DataFrame with the MAGGIC score caculated, the 3 years risk of death
        + 1 year risk of death and an index equal to the index of `data`.
    '''

    # check input

    assert len(columns) == 13, "`columns` should contain `13` entries."
    missing_columns = list(set(columns.values()).difference(data.columns))
    assert len(missing_columns) == 0, "'The following columns are missing: ' "+\
                                   ','.join(
                                       [list(columns.keys())[
                                           list(columns.values()).index(e)
                                       ] for e in missing_columns]
                                   )

    # ### initiate empty MAGGIC
    maggic = [0] * data.shape[0]

    # ### internal function
    def sl(*args):
        ''' Element wise addition of list '''
        return list(map(sum, zip(*args)))

    # ### Constants
    SEX = 'sex'; SMOK = 'smoking'; DIAB = 'hist_diabetes'
    COPD = 'hist_copd'; RECENT_HF = 'recent_hf'; BETAB = 'betablocker'
    ACE = 'ace_arb'; LVEF = 'lvef'; NYHA = 'nyha'; CREAT = 'creatinine'
    BMI = 'bmi'; SBP = 'sbp'; AGE='age'
    MAGGIC='maggic'; MAGGIC_RISK3='maggic (3-years risk of death)'; MAGGIC_RISK1='maggic (1-year risk of death)'

    # score to risk
    DEATH_YEAR3 = {0: 0.039, 1: 0.043, 2: 0.048, 3: 0.052, 4: 0.058, 5: 0.063,
                   6: 0.070, 7: 0.077, 8: 0.084, 9: 0.092, 10: 0.102, 11: 0.111,
                   12: 0.122, 13: 0.134, 14: 0.146, 15: 0.160, 16: 0.175,
                   17: 0.191, 18: 0.209, 19: 0.227, 20: 0.247, 21: 0.269,
                   22: 0.292, 23: 0.316, 24: 0.342, 25: 0.369, 26: 0.397,
                   27: 0.427, 28: 0.458, 29: 0.490, 30: 0.523, 31: 0.556,
                   32: 0.590, 33: 0.625, 34: 0.658, 35: 0.692, 36: 0.725,
                   37: 0.756, 38: 0.787, 39: 0.815, 40: 0.842, 41: 0.866,
                   42: 0.889, 43: 0.908, 44: 0.926, 45: 0.941, 46: 0.953,
                   47: 0.964, 48: 0.973, 49: 0.980, 50: 0.985
                   }
    DEATH_YEAR1 = {0: 0.015, 1: 0.016, 2: 0.018, 3: 0.020, 4: 0.022, 5: 0.024,
                   6: 0.027, 7: 0.029, 8: 0.032, 9: 0.036, 10: 0.039, 11: 0.043,
                   12: 0.048, 13: 0.052, 14: 0.058, 15: 0.063, 16: 0.070,
                   17: 0.077, 18: 0.084, 19: 0.093, 20: 0.102, 21: 0.111,
                   22: 0.122, 23: 0.134, 24: 0.147, 25: 0.160, 26: 0.175,
                   27: 0.186, 28: 0.202, 29: 0.219, 30: 0.237, 31: 0.256,
                   32: 0.292, 33: 0.316, 34: 0.342, 35: 0.369, 36: 0.398,
                   37: 0.427, 38: 0.458, 39: 0.490, 40: 0.523, 41: 0.557,
                   42: 0.591, 43: 0.625, 44: 0.659, 45: 0.692, 46: 0.725,
                   47: 0.757, 48: 0.787, 49: 0.816, 50: 0.842}

    # ### main effects
    # Male sex
    maggic = sl(maggic,[1 if s == 0 else 0 for s in data[columns[SEX]]])
    # Smoking
    maggic = sl(maggic,[1 if s == 1 else 0 for s in data[columns[SMOK]]])
    # diabetes
    maggic = sl(maggic,[3 if s == 1 else 0 for s in data[columns[DIAB]]])
    # copd
    maggic = sl(maggic,[2 if s == 1 else 0 for s in data[columns[COPD]]])
    # recent HF
    maggic = sl(maggic,[2 if s == 1 else 0 for s in data[columns[RECENT_HF]]])
    # BetaBlokc
    maggic = sl(maggic,[3 if s == 1 else 0 for s in data[columns[BETAB]]])
    # ACE/ARB
    maggic = sl(maggic,[1 if s == 1 else 0 for s in data[columns[ACE]]])
    # ### LVEF
    maggic = sl(maggic,[7 if s < 20 else 0 for s in data[columns[LVEF]]])
    maggic = sl(maggic,[6 if (s >= 20) & (s <25) else 0 for s in data[columns[LVEF]]])
    maggic = sl(maggic,[5 if (s >= 25) & (s <30) else 0 for s in data[columns[LVEF]]])
    maggic = sl(maggic,[3 if (s >= 30) & (s <35) else 0 for s in data[columns[LVEF]]])
    maggic = sl(maggic,[2 if (s >= 35) & (s <40) else 0 for s in data[columns[LVEF]]])

    # ### NYHA cla(maggic,s
    maggic = sl(maggic,[2 if s == 2 else 0 for s in data[columns[NYHA]]])
    maggic = sl(maggic,[6 if s == 3 else 0 for s in data[columns[NYHA]]])
    maggic = sl(maggic,[8 if s == 4 else 0 for s in data[columns[NYHA]]])

    # ### creat
    maggic = sl(maggic,[1 if (s >= 90) & (s <110) else 0 for s in data[columns[CREAT]]])
    maggic = sl(maggic,[2 if (s >= 110) & (s <130) else 0 for s in data[columns[CREAT]]])
    maggic = sl(maggic,[3 if (s >= 130) & (s <150) else 0 for s in data[columns[CREAT]]])
    maggic = sl(maggic,[4 if (s >= 150) & (s <170) else 0 for s in data[columns[CREAT]]])
    maggic = sl(maggic,[5 if (s >= 170) & (s <210) else 0 for s in data[columns[CREAT]]])
    maggic = sl(maggic,[6 if (s >= 210) & (s <250) else 0 for s in data[columns[CREAT]]])
    maggic = sl(maggic,[8 if s >= 250 else 0 for s in data[columns[CREAT]]])

    # ### BMI
    maggic = sl(maggic,[6 if s < 15 else 0 for s in data[columns[BMI]]])
    maggic = sl(maggic,[5 if (s >= 15) & (s <20) else 0 for s in data[columns[BMI]]])
    maggic = sl(maggic,[3 if (s >= 20) & (s <25) else 0 for s in data[columns[BMI]]])
    maggic = sl(maggic,[2 if (s >= 25) & (s <30) else 0 for s in data[columns[BMI]]])

    # ### Interact(maggic,on with LVEF and SBP
    # IF LVEF is l(maggic,ss than  30
    maggic = sl(maggic,[5 if (s < 110) & (l < 30) else 0 for s,l in\
                        zip(data[columns[SBP]], data[columns[LVEF]])])
    maggic = sl(maggic,[4 if (s >= 110) & (s <120) & (l < 30) else 0 for s,l in\
                        zip(data[columns[SBP]], data[columns[LVEF]])])
    maggic = sl(maggic,[3 if (s >= 120) & (s <130) & (l < 30) else 0 for s,l in\
                        zip(data[columns[SBP]], data[columns[LVEF]])])
    maggic = sl(maggic,[2 if (s >= 130) & (s <140) & (l < 30) else 0 for s,l in\
                        zip(data[columns[SBP]], data[columns[LVEF]])])
    maggic = sl(maggic,[1 if (s >= 140) & (s <150) & (l < 30) else 0 for s,l in\
                        zip(data[columns[SBP]], data[columns[LVEF]])])

    # IF LVEF is b(maggic,tween 30 and 40
    maggic = sl(maggic,[3 if (s < 110) & (l >= 30) & (l < 40) else 0 for s,l in\
                        zip(data[columns[SBP]], data[columns[LVEF]])])
    maggic = sl(maggic,[2 if (s >= 110) & (s <120) & (l >= 30) & (l < 40) else 0 for s,l in\
                        zip(data[columns[SBP]], data[columns[LVEF]])])
    maggic = sl(maggic,[1 if (s >= 120) & (s <130) & (l >= 30) & (l < 40) else 0 for s,l in\
                        zip(data[columns[SBP]], data[columns[LVEF]])])

    # IF LVEF is 4(maggic, or larger
    maggic = sl(maggic,[2 if (s < 110) & (l >= 40) else 0 for s,l in\
                        zip(data[columns[SBP]], data[columns[LVEF]])])
    maggic = sl(maggic,[1 if (s >= 110) & (s <120) & (l >= 40) else 0 for s,l in\
                        zip(data[columns[SBP]], data[columns[LVEF]])])

    # ### Interact(maggic,on with LVEF and AGE
    # IF LVEF is l(maggic,ss than  30
    maggic = sl(maggic,[1 if (s >= 55) & (s <60) & (l < 30) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])
    maggic = sl(maggic,[2 if (s >= 60) & (s <65) & (l < 30) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])
    maggic = sl(maggic,[4 if (s >= 65) & (s <70) & (l < 30) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])
    maggic = sl(maggic,[6 if (s >= 70) & (s <75) & (l < 30) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])
    maggic = sl(maggic,[8 if (s >= 75) & (s <80) & (l < 30) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])
    maggic = sl(maggic,[10 if (s >= 80) & (l < 30) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])

    # IF LVEF is b(maggic,tween 30 and 40
    maggic = sl(maggic,[2 if (s >= 55) & (s <60) & (l >=30) & (l<40) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])
    maggic = sl(maggic,[4 if (s >= 60) & (s <65) & (l >=30) & (l<40) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])
    maggic = sl(maggic,[6 if (s >= 65) & (s <70) & (l >=30) & (l<40) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])
    maggic = sl(maggic,[8 if (s >= 70) & (s <75) & (l >=30) & (l<40) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])
    maggic = sl(maggic,[10 if (s >= 75) & (s <80) & (l >=30) & (l<40) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])
    maggic = sl(maggic,[13 if (s >= 80) & (l >=30) & (l<40) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])

    # IF LVEF is 4(maggic, or larger
    maggic = sl(maggic,[3 if (s >= 55) & (s <60) & (l >=40) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])
    maggic = sl(maggic,[5 if (s >= 60) & (s <65) & (l >=40) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])
    maggic = sl(maggic,[7 if (s >= 65) & (s <70) & (l >=40) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])
    maggic = sl(maggic,[9 if (s >= 70) & (s <75) & (l >=40) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])
    maggic = sl(maggic,[12 if (s >= 75) & (s <80) & (l >=40) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])
    maggic = sl(maggic,[15 if (s >= 80) & (l >=40) else 0 for s,l in\
                        zip(data[columns[AGE]], data[columns[LVEF]])])

    # #### calculate risk
    death_risk3 = [*map(DEATH_YEAR3.get, maggic)]
    death_risk1 = [*map(DEATH_YEAR1.get, maggic)]

    # turn into pd.DataFrame
    new_data = pd.DataFrame({MAGGIC: maggic, MAGGIC_RISK3:death_risk3,
                                MAGGIC_RISK1:death_risk1},
                        index=data.index)

    # #### add missing values
    new_data.loc[data[list(columns.values())].isnull().any(axis=1),
                 [MAGGIC, MAGGIC_RISK3, MAGGIC_RISK1]] = np.nan

    # ### return
    return new_data