import numpy as np
import pandas as pd

def get_Marino_data(path=None):
    if path is None:
        path_file = './NEWcomp_HIIregions_v1.txt'
    else:
        path_file = path
    t = pd.read_csv(path_file, comment='#')
    t.columns = ('ID', 'REF', 'O2_3727', 'O2_3729', 'O3_4363', 'O3_4959',
                 'O3_5007', 'N2_5755', 'N2_6548', 'N2_6584', 'S3_6312',
                 'S2_6717', 'S2_6731', 'jD', 'jT', 'T3', 'ab')
    t['OH'] = t['ab']
    t['O32'] = np.log10(t['O3_5007'] + t['O2_3727'] + t['O2_3729'])
    t['O3'] = np.log10(t['O3_4363'] / t['O3_5007'])
    t['N2'] = np.log10(t['N2_5755'] / t['N2_6584'])
    return t

def get_Marino_data_werr(path=None):
    if path is None:
        path_file = './catalog_M13+err.csv'
    else:
        path_file = path
    t = pd.read_csv(path_file, comment='#')
    t.columns = ('R2', 'eR2', 'Hb', 'eHb', 'OIII5007', 'eOIII5007',
                 'Ha', 'eHa', 'NII6584', 'eNII6584', 'SII6717', 'eSII6717',
                 'SII6731', 'eSII6731', 'OH_M13_Te')
    ID = ['Marino13-{}'.format(i) for i in np.arange(t.shape[0])]
    t['ID'] = ID
    t = t.set_index('ID')
    t['R2+'] = t['R2']
    t['N2+'] = t['NII6584'] * 1.34
    t['S2'] = t['SII6717'] + t['SII6731']
    t['S2+'] = t['SII6717'] + t['SII6731']
    t['R3+'] = t['OIII5007'] * 1.335
    t['R23'] = t['R2'] + t['R3+']
    t['P'] = t['R3+'] / t['R23']
    t['N2/Ha'] = t['NII6584'] / t['Ha']
    t['O3/Hb'] = t['OIII5007'] / t['Hb']
    t['O3N2'] = t['O3/Hb']/t['N2/Ha']
    t['log O3N2'] = np.log10(t['O3N2'])
    t['log N2/Ha'] = np.log10(t['N2/Ha'])

    return t


def get_Ho_CHAOS3_data(path=None):
    if path is None:
        path_file = './data.txt'
    else:
        path_file = path
    t_temp = np.loadtxt(path_file, comments='#', usecols=[0,1,2,3, 4])
    t = pd.DataFrame(t_temp)
    t.columns = ('R25', 'OIIHb', 'OIIIHb', 'NIIHb', 'SIIHb')
    ID = ['Ho_CHAOS3-{}'.format(i) for i in np.arange(t.shape[0])]
    t['ID'] = ID
    t = t.set_index('ID')
    return t

def get_Ho_data(path=None):
    if path is None:
        path_file = './inputdata.dat'
    else:
        path_file = path
    t_temp = np.loadtxt(path_file, comments='#', usecols=[0, 1, 2, 3, 4])
    t = pd.DataFrame(t_temp)
    t.columns = ('R2', 'OIII5007', 'NII6584', 'S2', 'OH_Ho19_Te')
    ID = ['Ho19-{}'.format(i) for i in np.arange(t.shape[0])]
    t['ID'] = ID
    t = t.set_index('ID')
    t['Ha'] = 2.86
    t['Hb'] = 1
    t['R2+'] = t['R2']
    t['N2+'] = t['NII6584'] * 1.34
    t['S2+'] = t['S2']
    t['R3+'] = t['OIII5007'] * 1.335
    t['R23'] = t['R2'] + t['R3+']
    t['P'] = t['R3+'] / t['R23']
    t['N2/Ha'] = t['NII6584'] / t['Ha']
    t['O3/Hb'] = t['OIII5007']
    t['O3N2'] = t['O3/Hb']/t['N2/Ha']
    t['log O3N2'] = np.log10(t['O3N2'])
    t['log N2/Ha'] = np.log10(t['N2/Ha'])
    return t
