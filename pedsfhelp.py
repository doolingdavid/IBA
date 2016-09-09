
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pytz
from pytz import common_timezones, all_timezones
import matplotlib
matplotlib.style.use('ggplot')
get_ipython().magic('matplotlib inline')
from datetime import datetime
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
matplotlib.style.use('fivethirtyeight')
matplotlib.style.use('seaborn-talk')
import os
import collections


# In[2]:


specsdict = {'patient_id': (0,0+10),
            'casereg': (0,0+2),
            'casenumber': (2,2+8),
            'fivepct': (10,10+1),
            'medicaredateofdeath': (24,24+8),
            'birthm': (32,32+2),
            'birthyr': (36,36+4),
            'm_sex': (40,40+1),
            'race': (41,41+1),
            'state': (60,60+2),
            'county': (62,62+3),
            'zip5': (65,65+5),
            'urbrur': (96,96+1),
            'urban': (97,97+2),
            's_sex': (99,99+1),
            'rac_recb': (100,100+2),
            'rac_recy': (102, 102+1),
            'rac_reca': (103,103+1),
            'icd_code': (104,104+1),
            'cod89v': (105,105+4),
            'cod10v': (109,109+4),
            'codkm': (113,113+5),
            'codpub': (118,118+5),
            'nhiade': (123,123+1),
             'ser_dodm': (124,124+2),
             'ser_dody': (126,126+4),
            'deathflag': (130,130+1),
            'srace': (131,131+2),
            'origin': (133,133+1),
            'origrecb': (134,134+1),
            'stat_rec': (141,141+1),
            'census_pov_ind': (145,145+1),
            'yr_brth': (146,146+4),
            'count': (152,152+2),
            'numprims': (154,154+2),
            'resnrec': (156,156+1),
            'plc_brth_cnt': (171,171+3),
            'plc_brth_state': (174,174+2),
            'reg1': (1877,1877+2),
            'reg2': (1877+300, 1877+300+2),
            'reg3': (1877+2*300, 1877+2*300+2),
             'reg4': (1877+3*300, 1877+3*300+2),
             'reg5': (1877+4*300, 1877+4*300+2),
             'reg6': (1877+5*300, 1877+5*300+2),
             'reg7': (1877+6*300, 1877+6*300+2),
             'reg8': (1877+7*300, 1877+7*300+2),
             'reg9': (1877+8*300, 1877+8*300+2),
             'reg10': (1877+9*300, 1877+9*300+2),
            'marst1': (1879,1879+1),
            'marst2': (1879+300, 1879+300+1),
            'marst3': (1879+2*300, 1879+2*300+1),
             'marst4': (1879+3*300, 1879+3*300+1),
             'marst5': (1879+4*300, 1879+4*300+1),
             'marst6': (1879+5*300, 1879+5*300+1),
             'marst7': (1879+6*300, 1879+6*300+1),
             'marst8': (1879+7*300, 1879+7*300+1),
             'marst9': (1879+8*300, 1879+8*300+1),
             'marst10': (1879+9*300, 1879+9+300+1),
            'agedx1': (1880,1880+3),
             'agedx2': (1880+300, 1880+300+3),
             'agedx3': (1880+2*300, 1880+2*300+3),
             'agedx4': (1880+3*300, 1880+3*300+3),
             'agedx5': (1880+4*300, 1880+4*300+3),
             'agedx6': (1880+5*300, 1880+5*300+3),
             'agedx7': (1880+6*300, 1880+6*300+3),
             'agedx8': (1880+7*300, 1880+7*300+3),
             'agedx9': (1880+8*300, 1880+8*300+3),
             'agedx10': (1880+9*300, 1880+9*300+3),
            'seq1': (1883,1883+2),
             'seq2': (1883+300, 1883+300+2),
             'seq3': (1883+2*300, 1883+2*300+2),
             'seq4': (1883+3*300, 1883+3*300+2),
             'seq5': (1883+4*300, 1883+4*300+2),
             'seq6': (1883+5*300, 1883+5*300+2),
             'seq7': (1883+6*300, 1883+6*300+2),
             'seq8': (1883+7*300, 1883+7*300+2),
             'seq9': (1883+8*300, 1883+8*300+2),
             'seq10': (1883+9*300, 1883+9*300+2),
            'modx1': (1885,1885+2),
             'modx2': (1885+300, 1885+300+2),
             'modx3': (1885+2*300, 1885+2*300+2),
             'modx4': (1885+3*300, 1885+3*300+2),
             'modx5': (1885+4*300, 1885+4*300+2),
             'modx6': (1885+5*300, 1885+5*300+2),
             'modx7': (1885+6*300, 1885+6*300+2),
             'modx8': (1885+7*300, 1885+7*300+2),
             'modx9': (1885+8*300, 1885+8*300+2),
             'modx10': (1885+9*300, 1885+9*300+2),
            'yrdx1': (1887, 1887+4),
             'yrdx2': (1887+300, 1887+300+4),
             'yrdx3': (1887+2*300, 1887+2*300+4),
             'yrdx4': (1887+3*300, 1887+3*300+4),
             'yrdx5': (1887+4*300, 1887+4*300+4),
             'yrdx6': (1887+5*300, 1887+5*300+4),
             'yrdx7': (1887+6*300, 1887+6*300+4),
             'yrdx8': (1887+7*300, 1887+7*300+4),
             'yrdx9': (1887+8*300, 1887+8*300+4),
             'yrdx10': (1887+9*300, 1887+9*300+4),
            'site1': (1891,1891+3),
             'site2': (1891+300, 1891+300+3),
             'site3': (1891+2*300, 1891+2*300+3),
             'site4': (1891+3*300, 1891+3*300+3),
             'site5': (1891+4*300, 1891+4*300+3),
             'site6': (1891+5*300, 1891+5*300+3),
             'site7': (1891+6*300, 1891+6*300+3),
             'site8': (1891+7*300, 1891+7*300+3),
             'site9': (1891+8*300, 1891+8*300+3),
             'site10': (1891+9*300, 1891+9*300+3),
            'lat1': (1894,1894+1),
             'lat2': (1894+300, 1894+300+1),
             'lat3': (1894+2*300, 1894+2*300+1),
             'lat4': (1894+3*300, 1894+3*300+1),
             'lat5': (1894+4*300, 1894+4*300+1),
             'lat6': (1894+5*300, 1894+5*300+1),
             'lat7': (1894+6*300, 1894+6*300+1),
             'lat8': (1894+7*300, 1894+7*300+1),
             'lat9': (1894+8*300, 1894+8*300+1),
             'lat10': (1894+9*300, 1894+9*300+1),
            'hist1': (1900,1900+4),
             'hist2': (1900+300, 1900+300+4),
             'hist3': (1900+2*300, 1900+2*300+4),
             'hist4': (1900+3*300, 1900+3*300+4),
             'hist5': (1900+4*300, 1900+4*300+4),
             'hist6': (1900+5*300, 1900+5*300+4),
             'hist7': (1900+6*300, 1900+6*300+4),
             'hist8': (1900+7*300, 1900+7*300+4),
             'hist9': (1900+8*300, 1900+8*300+4),
             'hist10': (1900+9*300, 1900+9*300+4),
            'beh1': (1904,1904+1),
             'beh2': (1904+300, 1904+300+1),
             'beh3': (1904+2*300, 1904+2*300+1),
             'beh4': (1904+3*300, 1904+3*300+1),
             'beh5': (1904+4*300, 1904+4*300+1),
             'beh6': (1904+5*300, 1904+5*300+1),
             'beh7': (1904+6*300, 1904+6*300+1),
             'beh8': (1904+7*300, 1904+7*300+1),
             'beh9': (1904+8*300, 1904+8*300+1),
             'beh10': (1904+9*300, 1904+9*300+1),
            'grade1': (1905,1905+1),
             'grade2': (1905+300, 1905+300+1),
             'grade3': (1905+2*300, 1905+2*300+1),
             'grade4': (1905+3*300, 1905+3*300+1),
             'grade5': (1905+4*300, 1905+4*300+1),
             'grade6': (1905+5*300, 1905+5*300+1),
             'grade7': (1905+6*300, 1905+6*300+1),
             'grade8': (1905+7*300, 1905+7*300+1),
             'grade9': (1905+8*300, 1905+8*300+1),
             'grade10': (1905+9*300, 1905+9*300+1),
            'dxconf1': (1906,1906+1),
             'dxconf2': (1906+300, 1906+300+1),
             'dxconf3': (1906+2*300, 1906+2*300+1),
             'dxconf4': (1906+3*300, 1906+3*300+1),
             'dxconf5': (1906+4*300, 1906+4*300+1),
             'dxconf6': (1906+5*300, 1906+5*300+1),
             'dxconf7': (1906+6*300, 1906+6*300+1),
             'dxconf8': (1906+7*300, 1906+7*300+1),
             'dxconf9': (1906+8*300, 1906+8*300+1),
             'dxconf10': (1906+9*300, 1906+9*300+1),
             'e10pn1': (1918, 1918 + 2),
             'e10pn2': (1918+300, 1918+300+2),
             'e10pn3': (1918+2*300, 1918+2*300+2),
             'e10pn4': (1918+3*300, 1918+3*300+2),
             'e10pn5': (1918+4*300, 1918+4*300+2),
             'e10pn6': (1918+5*300, 1918+5*300+2),
             'e10pn7': (1918+6*300, 1918+6*300+2),
             'e10pn8': (1918+7*300, 1918+7*300+2),
             'e10pn9': (1918+8*300, 1918+8*300+2),
             'e10pn10': (1918+9*300, 1918+9*300+2),
            'cstum1': (1943,1943+3),
             'cstum2': (1943+300, 1943+300+3),
             'cstum3': (1943+2*300, 1943+2*300+3),
             'cstum4': (1943+3*300, 1943+3*300+3),
             'cstum5': (1943+4*300, 1943+4*300+3),
             'cstum6': (1943+5*300, 1943+5*300+3),
             'cstum7': (1943+6*300, 1943+6*300+3),
             'cstum8': (1943+7*300, 1943+7*300+3),
             'cstum9': (1943+8*300, 1943+8*300+3),
             'cstum10': (1943+9*300, 1943+9*300+3),
            'dajcct1': (1975,1975+2),
             'dajcct2': (1975+300, 1975+300+2),
             'dajcct3': (1975+2*300, 1975+2*300+2),
             'dajcct4': (1975+3*300, 1975+3*300+2),
             'dajcct5': (1975+4*300, 1975+4*300+2),
             'dajcct6': (1975+5*300, 1975+5*300+2),
             'dajcct7': (1975+6*300, 1975+6*300+2),
             'dajcct8': (1975+7*300, 1975+7*300+2),
             'dajcct9': (1975+8*300, 1975+8*300+2),
             'dajcct10': (1975+9*300, 1975+9*300+2),
            'dajccn1': (1977,1977+2),
             'dajccn2': (1977+300, 1977+300+2),
             'dajccn3': (1977+2*300, 1977+2*300+2),
             'dajccn4': (1977+3*300, 1977+3*300+2),
             'dajccn5': (1977+4*300, 1977+4*300+2),
             'dajccn6': (1977+5*300, 1977+5*300+2),
             'dajccn7': (1977+6*300, 1977+6*300+2),
             'dajccn8': (1977+7*300, 1977+7*300+2),
             'dajccn9': (1977+8*300, 1977+8*300+2),
             'dajccn10': (1977+9*300, 1977+9*300+2),
            'dajccm1': (1979,1979+2),
             'dajccm2': (1979+300, 1979+300+2),
             'dajccm3': (1979+2*300, 1979+2*300+2),
             'dajccm4': (1979+3*300, 1979+3*300+2),
             'dajccm5': (1979+4*300, 1979+4*300+2),
             'dajccm6': (1979+5*300, 1979+5*300+2),
             'dajccm7': (1979+6*300, 1979+6*300+2),
             'dajccm8': (1979+7*300, 1979+7*300+2),
             'dajccm9': (1979+8*300, 1979+8*300+2),
             'dajccm10': (1979+9*300, 1979+9*300+2),
            'dajccstg1': (1981,1981+2),
             'dajccstg2': (1981+300, 1981+300+2),
             'dajccstg3': (1981+2*300, 1981+2*300+2),
             'dajccstg4': (1981+3*300, 1981+3*300+2),
             'dajccstg5': (1981+4*300, 1981+4*300+2),
             'dajccstg6': (1981+5*300, 1981+5*300+2),
             'dajccstg7': (1981+6*300, 1981+6*300+2),
             'dajccstg8': (1981+7*300, 1981+7*300+2),
             'dajccstg9': (1981+8*300, 1981+8*300+2),
             'dajccstg10': (1981+9*300, 1981+9*300+2),
            'dss77s1': (1983,1983+1),
             'dss77s2': (1983+300, 1983+300+1),
             'dss77s3': (1983+2*300, 1983+2*300+1),
             'dss77s4': (1983+3*300, 1983+3*300+1),
             'dss77s5': (1983+4*300, 1983+4*300+1),
             'dss77s6': (1983+5*300, 1983+5*300+1),
             'dss77s7': (1983+6*300, 1983+6*300+1),
             'dss77s8': (1983+7*300, 1983+7*300+1),
             'dss77s9': (1983+8*300, 1983+8*300+1),
             'dss77s10': (1983+9*300, 1983+9*300+1),
            'dss00s1': (1984,1984+1),
             'dss00s2': (1984+300, 1984+300+1),
             'dss00s3': (1984+2*300, 1984+2*300+1),
             'dss00s4': (1984+3*300, 1984+3*300+1),
             'dss00s5': (1984+4*300, 1984+4*300+1),
             'dss00s6': (1984+5*300, 1984+5*300+1),
             'dss00s7': (1984+6*300, 1984+6*300+1),
             'dss00s8': (1984+7*300, 1984+7*300+1),
             'dss00s9': (1984+8*300, 1984+8*300+1),
             'dss00s10': (1984+9*300, 1984+9*300+1),
            'sxprif1': (2006,2006+2),
             'sxprif2': (2006+300, 2006+300+2),
              'sxprif3': (2006+2*300, 2006+2*300+2),
              'sxprif4': (2006+3*300, 2006+3*300+2),
              'sxprif5': (2006+4*300, 2006+4*300+2),
              'sxprif6': (2006+5*300, 2006+5*300+2),
              'sxprif7': (2006+6*300, 2006+6*300+2),
              'sxprif8': (2006+7*300, 2006+7*300+2),
              'sxprif9': (2006+8*300, 2006+8*300+2),
              'sxprif10': (2006+9*300, 2006+9*300+2),
            'sxscof1': (2008,2008+1),
             'sxscof2': (2008+300, 2008+300+1),
             'sxscof3': (2008+2*300, 2008+2*300+1),
             'sxscof4': (2008+3*300, 2008+3*300+1),
             'sxscof5': (2008+4*300, 2008+4*300+1),
             'sxscof6': (2008+5*300, 2008+5*300+1),
             'sxscof7': (2008+6*300, 2008+6*300+1),
             'sxscof8': (2008+7*300, 2008+7*300+1),
             'sxscof9': (2008+8*300, 2008+8*300+1),
             'sxscof10': (2008+9*300, 2008+9*300+1),
            'sxsitf1': (2009,2009+1),
             'sxsitf2': (2009+300, 2009+300+1),
             'sxsitf3': (2009+2*300, 2009+2*300+1),
             'sxsitf4': (2009+3*300, 2009+3*300+1),
             'sxsitf5': (2009+4*300, 2009+4*300+1),
             'sxsitf6': (2009+5*300, 2009+5*300+1),
             'sxsitf7': (2009+6*300, 2009+6*300+1),
             'sxsitf8': (2009+7*300, 2009+7*300+1),
             'sxsitf9': (2009+8*300, 2009+8*300+1),
             'sxsitf10': (2009+9*300, 2009+9*300+1),
             'nosrg1': (2013,2013+1),
             'nosrg2': (2013+300, 2013+300+1),
             'nosrg3': (2013+2*300, 2013+2*300+1),
             'nosrg4': (2013+3*300, 2013+3*300+1),
             'nosrg5': (2013+4*300, 2013+4*300+1),
             'nosrg6': (2013+5*300, 2013+5*300+1),
             'nosrg7': (2013+6*300, 2013+6*300+1),
             'nosrg8': (2013+7*300, 2013+7*300+1),
             'nosrg9': (2013+8*300, 2013+8*300+1),
             'nosrg10': (2013+9*300, 2013+9*300+1),
            'rad1': (2014,2014+1),
            'rad2': (2014+300, 2014+300+1),
             'rad3': (2014+2*300, 2014+2*300+1),
             'rad4': (2014+3*300, 2014+3*300+1),
             'rad5': (2014+4*300, 2014+4*300+1),
             'rad6': (2014+5*300, 2014+5*300+1),
             'rad7': (2014+6*300, 2014+6*300+1),
             'rad8': (2014+7*300, 2014+7*300+1),
             'rad9': (2014+8*300, 2014+8*300+1),
             'rad10': (2014+9*300, 2014+9*300+1),
             'radsurg1': (2016,2016+1),
             'radsurg2': (2016+300, 2016+300+1),
             'radsurg3': (2016+2*300, 2016+2*300+1),
             'radsurg4': (2016+3*300, 2016+3*300+1),
             'radsurg5': (2016+4*300, 2016+4*300+1),
             'radsurg6': (2016+5*300, 2016+5*300+1),
             'radsurg7': (2016+6*300, 2016+6*300+1),
             'radsurg8': (2016+7*300, 2016+7*300+1),
             'radsurg9': (2016+8*300, 2016+8*300+1),
             'radsurg10': (2016+9*300, 2016+9*300+1),
             'ager1': (2037,2037+2),
             'ager2': (2037+300, 2037+300+2),
             'ager3': (2037+2*300, 2037+2*300+2),
             'ager4': (2037+3*300, 2037+3*300+2),
             'ager5': (2037+4*300, 2037+4*300+2),
             'ager6': (2037+5*300, 2037+5*300+2),
             'ager7': (2037+6*300, 2037+6*300+2),
             'ager8': (2037+7*300, 2037+7*300+2),
             'ager9': (2037+8*300, 2037+8*300+2),
             'ager10': (2037+9*300, 2037+9*300+2),
             'siterwho1': (2039, 2039+5),
             'siterwho2': (2039+300, 2039+300+5),
             'siterwho3': (2039+2*300, 2039+2*300+5),
             'siterwho4': (2039+3*300, 2039+3*300+5),
             'siterwho5': (2039+4*300, 2039+4*300+5),
             'siterwho6': (2039+5*300, 2039+5*300+5),
             'siterwho7': (2039+6*300, 2039+6*300+5),
             'siterwho8': (2039+7*300, 2039+7*300+5),
             'siterwho9': (2039+8*300, 2039+8*300+5),
             'siterwho10': (2039+9*300, 2039+9*300+5),
             'icdot09_1': (2048, 2048+4),
             'icdot09_2': (2048+300, 2048 + 300 +4),
             'icdot09_3': (2048+2*300, 2048 + 2*300 +4),
             'icdot09_4': (2048+3*300, 2048 + 3*300 +4),
             'icdot09_5': (2048+4*300, 2048 + 4*300 +4),
             'icdot09_6': (2048+5*300, 2048 + 5*300 +4),
             'icdot09_7': (2048+6*300, 2048 + 6*300 +4),
             'icdot09_8': (2048+7*300, 2048 + 7*300 +4),
             'icdot09_9': (2048+8*300, 2048 + 8*300 +4),
             'icdot09_10': (2048+9*300, 2048 + 9*300 +4),
            'iccc3who1': (2056, 2056+3),
             'iccc3who2': (2056+300, 2056+300+3),
             'iccc3who3': (2056+2*300, 2056+2*300+3),
             'iccc3who4': (2056+3*300, 2056+3*300+3),
             'iccc3who5': (2056+4*300, 2056+4*300+3),
             'iccc3who6': (2056+5*300, 2056+5*300+3),
             'iccc3who7': (2056+6*300, 2056+6*300+3),
             'iccc3who8': (2056+7*300, 2056+7*300+3),
             'iccc3who9': (2056+8*300, 2056+8*300+3),
             'iccc3who10': (2056+9*300, 2056+9*300+3),
             'histrec1': (2063, 2063+2),
             'histrec2': (2063+300, 2063+300+2),
             'histrec3': (2063+2*300, 2063+2*300+2),
             'histrec4': (2063+3*300, 2063+3*300+2),
             'histrec5': (2063+4*300, 2063+4*300+2),
             'histrec6': (2063+5*300, 2063+5*300+2),
             'histrec7': (2063+6*300, 2063+6*300+2),
             'histrec8': (2063+7*300, 2063+7*300+2),
             'histrec9': (2063+8*300, 2063+8*300+2),
             'histrec10': (2063+9*300, 2063+9*300+2),
             'hisrcb1': (2065, 2065+2),
             'hisrcb2': (2065+300, 2065+300+2),
             'hisrcb3': (2065+2*300, 2065+2*300+2),
             'hisrcb4': (2065+3*300, 2065+3*300+2),
             'hisrcb5': (2065+4*300, 2065+4*300+2),
             'hisrcb6': (2065+5*300, 2065+5*300+2),
             'hisrcb7': (2065+6*300, 2065+6*300+2),
             'hisrcb8': (2065+7*300, 2065+7*300+2),
             'hisrcb9': (2065+8*300, 2065+8*300+2),
             'hisrcb10': (2065+9*300, 2065+9*300+2),
             'statecd1': (2078, 2078+2),
             'statecd2': (2078+300, 2078+300+2),
             'statecd3': (2078+2*300, 2078+2*300+2),
             'statecd4': (2078+3*300, 2078+3*300+2),
             'statecd5': (2078+4*300, 2078+4*300+2),
             'statecd6': (2078+5*300, 2078+5*300+2),
             'statecd7': (2078+6*300, 2078+6*300+2),
             'statecd8': (2078+7*300, 2078+7*300+2),
             'statecd9': (2078+8*300, 2078+8*300+2),
             'statecd10': (2078+9*300, 2078+9*300+2),
             'cnty1': (2080, 2080+3),
             'cnty2': (2080+300, 2080+300+3),
             'cnty3': (2080+2*300, 2080+2*300+3),
             'cnty4': (2080+3*300, 2080+3*300+3),
             'cnty5': (2080+4*300, 2080+4*300+3),
             'cnty6': (2080+5*300, 2080+5*300+3),
             'cnty7': (2080+6*300, 2080+6*300+3),
             'cnty8': (2080+7*300, 2080+7*300+3),
             'cnty9': (2080+8*300, 2080+8*300+3),
             'cnty10': (2080+9*300, 2080+9*300+3),
             'cssch1': (2097, 2097+2),
             'cssch2': (2097+300, 2097+300+2),
             'cssch3': (2097+2*300, 2097+2*300+2),
             'cssch4': (2097+3*300, 2097+3*300+2),
             'cssch5': (2097+4*300, 2097+4*300+2),
             'cssch6': (2097+5*300, 2097+5*300+2),
             'cssch7': (2097+6*300, 2097+6*300+2),
             'cssch8': (2097+7*300, 2097+7*300+2),
             'cssch9': (2097+8*300, 2097+8*300+2),
             'cssch10': (2097+9*300, 2097+9*300+2),
             'srvm1': (2118, 2118+4),
             'srvm2': (2118+300, 2118+300+4),
             'srvm3': (2118+2*300, 2118+2*300+4),
             'srvm4': (2118+3*300, 2118+3*300+4),
             'srvm5': (2118+4*300, 2118+4*300+4),
             'srvm6': (2118+5*300, 2118+5*300+4),
             'srvm7': (2118+6*300, 2118+6*300+4),
             'srvm8': (2118+7*300, 2118+7*300+4),
             'srvm9': (2118+8*300, 2118+8*300+4),
             'srvm10': (2118+9*300, 2118+9*300+4),
             'insrecpb1': (2128, 2128+1),
             'insrecpb2': (2128+300, 2128+300+1),
             'insrecpb3': (2128+2*300, 2128+2*300+1),
             'insrecpb4': (2128+3*300, 2128+3*300+1),
             'insrecpb5': (2128+4*300, 2128+4*300+1),
             'insrecpb6': (2128+5*300, 2128+5*300+1),
             'insrecpb7': (2128+6*300, 2128+6*300+1),
             'insrecpb8': (2128+7*300, 2128+7*300+1),
             'insrecpb9': (2128+8*300, 2128+8*300+1),
             'insrecpb10': (2128+9*300, 2128+9*300+1),
             'payer_dx1': (2175, 2175 +2),
             'payer_dx2': (2175 + 300, 2175+300 + 2),
             'payer_dx3': (2175 + 2*300, 2175+2*300 + 2),
             'payer_dx4': (2175 + 3*300, 2175+3*300 + 2),
             'payer_dx5': (2175 + 4*300, 2175+4*300 + 2),
             'payer_dx6': (2175 + 5*300, 2175+5*300 + 2),
             'payer_dx7': (2175 + 6*300, 2175+6*300 + 2),
             'payer_dx8': (2175 + 7*300, 2175+7*300 + 2),
             'payer_dx9': (2175 + 8*300, 2175+8*300 + 2),
             'payer_dx10': (2175 + 9*300, 2175+9*300 + 2),
             }

names = sorted(specsdict,key=specsdict.__getitem__)

colspecs = [specsdict[n] for n in names]


# In[3]:

def make_icd9sg_dataframe():
    """Make a dataframe that includes the icd9sg descriptins."""
    
    dficd9sg = pd.read_excel('CMS32_DESC_LONG_SHORT_SG.xlsx',
                        convert_float=False,
                        converters = {'PROCEDURE CODE': str})
    dficd9sg = dficd9sg.rename(columns={'PROCEDURE CODE': 'code',
                        'LONG DESCRIPTION': 'long_description',
                        'SHORT DESCRIPTION': 'short_description'})

    return dficd9sg


def icd9dx_description_trans(df,dficd9dx,columnname):
    """takes the dataframe df and the columnname as a string and adds a 
    column containing the long description to df using the 
    description string in dficd9dx"""
    
    df[columnname] = df[columnname].fillna(-1)
    
    df[columnname] = df[columnname].astype(int)
    
    dfhuh = pd.merge(df[[columnname]].astype('str'), dficd9dx,
                 left_on=columnname, right_on='code',how='left')
    df[columnname + '_description'] = dfhuh['long_description']
    del dfhuh

    
def make_icd9dx_dataframe():
    """Make a dataframe that includes the icd9dx descriptions."""
    
    dficd9dx = pd.read_excel('CMS32_DESC_LONG_SHORT_DX.xlsx',
                        convert_float=False,
                        converters = {'PROCEDURE CODE': str})
    dficd9dx = dficd9dx.rename(columns={'DIAGNOSIS CODE': 'code',
                                    'LONG DESCRIPTION': 'long_description',
                        'SHORT DESCRIPTION': 'short_description'})

    return dficd9dx


# In[4]:

preDouble = "\\\\iobsdc01\\SharedDocs\\SEER_MEDICARE\\SEER_MEDICARE_STAGING"
new = os.chdir(preDouble)
dficd9dx = make_icd9dx_dataframe()


# In[5]:

dfcensus = pd.read_csv('http://www2.census.gov/geo/docs/reference/codes/files/national_county.txt',
                      header=None,names=['State','STATEFIPS','COUNTYFIPS','County','Ignore'],
                      dtype={'STATEFIPS':object,'COUNTYFIPS':object})

dfcensus['CountyState'] = dfcensus['County'] + ', ' + dfcensus['State']

dfcensus['ALLFIPS'] = dfcensus['STATEFIPS'] + dfcensus['COUNTYFIPS']

dfcensus.set_index('ALLFIPS',inplace=True)

dfcensusdict = dfcensus['CountyState'].to_dict()
dfcensusstatedict = dfcensus['State'].to_dict()

dfcensus.set_index('STATEFIPS',inplace=True)

ST_CNTY_ORIGdict = dfcensus['State'].to_dict()


ST_CNTYdict = dfcensusdict


# In[6]:

dfplace = pd.read_pickle('dfnewplace.pickle')

dfplace.set_index('FIPScombo',inplace=True)

FIPScombolatdict = dfplace['lat'].to_dict()
FIPScombolngdict = dfplace['lng'].to_dict()
FIPScomboelevationdict = dfplace['elevation'].to_dict()
FIPScombocountystatedict = dfplace['CountyState'].to_dict()


# In[7]:

casereg_dict = {2: 'Connecticut (1973+)',
               20: 'Detroit (1973+)',
               21: 'Hawaii (1973+)',
               22: 'Iowa (1973+)',
               23: 'New Mexico (1973+)',
               25: 'Seattle (1974+)',
               26: 'Utah (1973+)',
               42: 'Kentucky (2000+)',
               43: 'Louisiana (2000+)',
               44: 'New Jersey (2000+)',
               87: 'Georgia',
               88: 'California'}

##########################################################################

stat_rec_dict = {1: 'Alive',
                 4: 'Dead'}

###########################################################################

fivepct_dict = {'Y': 'Included',
                'N': 'Not included'}

##############################################################

birthm_dict = {1: 'Jan',
              2: 'Feb', 3: 'Mar', 4: 'Apr',
              5: 'May', 6: 'Jun', 7: 'Jul',
              8: 'Aug', 9: 'Sep', 10: 'Oct', 
              11: 'Nov', 12: 'Dec'}

#######################################################

m_sex_dict = {1: 'Male', 2: 'Female'}


#############################################################

race_dict = {1: 'White',
            2: 'Black',
            3: 'Other',
            4: 'Asian',
            5: 'Hispanic',
            6: 'North American Native',
            0: 'Unknown'}


#####################################################################

urbrur_dict = {1: 'Big Metro',
              2: 'Metro', 3: 'Urban', 4: 'Less Urban',
              5: 'Rural', 9: 'Unknown'}


#######################################################################################


urban_dict = {1: 'Counties of metro areas of 1 million population or more',
             2: 'Counties in metro areas of 250,000 - 1,000,000 population',
             3: 'Counties in metro areas of fewer than 250,000 population',
             4: 'Urban population of 20,000 or more, adjacent to a metro area',
             5: 'Urban population of 20,000 or more, not adjacent to a metro area',
             6: 'Urban population of 2,500-19,999, adjacent to a metro area',
             7: 'Urban population of 2,500-19,999, not adjacent to a metro area',
             8: 'Completely rural or less than 2,500 urban population, adjacent to a metro area',
             9: 'Completely rural or less than 2,500 urban population, not adjacent to a metro area',
             99: 'Missing value'}


##############################################################################

rac_recb_dict = {1: 'Caucasian, NOS',
                2: 'Black',
                3: 'American Indian/Alaska Native',
                4: 'Chinese',
                5: 'Japanese',
                6: 'Filipino',
                7: 'Hawaiian',
                8: 'Other Asian or Pacific Islander',
                9: 'Unknown',
                11: 'Caucasian, Spanish origin or surname',
                12: 'Other unspecified (1991+)'}

###################################################################################

rac_recy_dict = {1: 'White',
                2: 'Black',
                3: 'American Indian/Alaska Native',
                4: 'Asian or Pacific Islander',
                7: 'Other unspecified (1991+)',
                9: 'Unknown'}


rac_reca_dict = {1: 'White',
                2: 'Black',
                3: 'Other (American Indian/AK Native, Asian/Pacific Islander)',
                7: 'Other unspecified (1991+)',
                9: 'Unknown'}

#################################################################################

icd_code_dict = {0: 'Patient is alive at last follow-up',
                1: 'Tenth ICD revision',
                8: 'Eighth ICD revision',
                9: 'Ninth ICD revision'}

#######################################################################################

s_sex_dict = {1: 'Male',
              2: 'Female'}

#######################################################################################

nhiade_dict = {0: 'Non-Spanish-Hispanic-Latino',
              1: 'Mexican',
              2: 'Puerto Rican',
              3: 'Cuban',
              4: 'South or Central American excluding Brazil',
              5: 'Other specified Spanish/Hispanic Origin including Europe',
              6: 'Spanish/Hispanic/Latino, NOS',
              7: 'NHIA Surname Match Only',
              8: 'Dominican Republic'}

#######################################################################################


codpub_dict = {0: 'Alive',
        20010: 'Lip',
        20020: 'Tongue',
         20030: 'Salivary Gland',
         20040: 'Floor of Mouth',
         20050: 'Gum and Other Mouth',
         20060: 'Nasopharynx',
         20070: 'Tonsil',
         20080: 'Oropharynx',
         20090: 'Hypopharynx',
         20100: 'Other Oral Cavity and Pharynx',
         21010: 'Esophagus',
         21020: 'Stomach',
         21030: 'Small Intestine',
         21040: 'Colon excluding Rectum',
         21050: 'Rectum and Rectosigmoid Junction',
         21060: 'Anus, Anal Canal and Anorectum',
         21071: 'Liver',
         21072: 'Intrahepatic Bile Duct',
         21080: 'Gallbladder',
         21090: 'Other Biliary',
         21100: 'Pancreas',
         21110: 'Retroperitoneum',
         21120: 'Peritoneum, Omentum and Mesentery',
         21130: 'Other Digestive Organs',
         22010: 'Nose, Nasal Cavity and Middle Ear',
         22020: 'Larynx',
         22030: 'Lung and Bronchus',
         22050: 'Pleura',
         22060: 'Trachea, Mediastinum and Other Respiratory Organs',
         23000: 'Bones and Joints',
         24000: 'Soft Tissue including Heart$',
         25010: 'Melanoma of the Skin',
         25020: 'Non-Melanoma Skin&',
         26000: 'Breast',
         27010: 'Cervix Uteri',
         27020: 'Corpus Uteri',
         27030: 'Uterus, NOS',
         27040: 'Ovary',
         27050: 'Vagina',
         27060: 'Vulva',
         27070: 'Other Female Genital Organs',
         28010: 'Prostate',
         28020: 'Testis',
         28030: 'Penis',
         28040: 'Other Male Genital Organs',
         29010: 'Urinary Bladder',
         29020: 'Kidney and Renal Pelvis',
         29030: 'Ureter',
         29040: 'Other Urinary Organs',
         30000: 'Eye and Orbit',
         31010: 'Brain and Other Nervous System',
         32010: 'Thyroid',
         32020: 'Other Endocrine including Thymus$',
         33010: 'Hodgkin Lymphoma',
         33040: 'Non-Hodgkin Lymphoma',
         34000: 'Myeloma',
         35011: 'Acute Lymphocytic Leukemia',
         35012: 'Chronic Lymphocytic Leukemia',
         35013: 'Other Lymphocytic Leukemia',
         35021: 'Acute myeloid',
         35022: 'Chronic Myeloid Leukemia',
         35023: 'Other Myeloid/Monocytic Leukemia',
         35031: 'Acute Monocytic Leukemia',
         35041: 'Other Acute Leukemia',
         35043: 'Aleukemic, subleukemic and NOS',
         37000: 'Miscellaneous Malignant Cancer',
         38000: 'In situ, benign or unknown behavior neoplasm',
         41000: 'State DC not available or state DC available but no COD',
         50000: 'Tuberculosis',
         50010: 'Syphilis',
         50030: 'Septicemia',
         50040: 'Other Infectious and Parasitic Diseases',
         50050: 'Diabetes Mellitus',
         50051: 'Alzheimers (ICD-9 and 10 only)',
         50060: 'Diseases of Heart',
         50070: 'Hypertension without Heart Disease',
         50080: 'Cerebrovascular Diseases',
         50090: 'Atherosclerosis',
         50100: 'Aortic Aneurysm and Dissection',
         50110: 'Other Diseases of Arteries, Arterioles, Capillaries',
         50120: 'Pneumonia and Influenza',
         50130: 'Chronic Obstructive Pulmonary Disease and Allied Cond',
         50140: 'Stomach and Duodenal Ulcers',
         50150: 'Chronic Liver Disease and Cirrhosis',
         50160: 'Nephritis, Nephrotic Syndrome and Nephrosis',
         50170: 'Complications of Pregnancy, Childbirth, Puerperium',
         50180: 'Congenital Anomalies',
         50190: 'Certain Conditions Originating in Perinatal Period',
         50200: 'Symptoms, Signs and Ill-Defined Conditions',
         50210: 'Accidents and Adverse Effects',
         50220: 'Suicide and Self-Inflicted Injury',
         50230: 'Homicide and Legal Intervention',
         50300: 'Other Cause of Death',
         99999: 'Unknown/missing/invalid COD'}

#################################################################################################################

ser_dodm_dict = {0: 'Alive',
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
                5: 'May', 6: 'Jun', 7: 'Jul', 8:'Aug',
                9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

###############################################################################################################

srace_dict = {1: 'White',
              2: 'Black',
              3: 'American Indian, Aleutian, Alaska Native or Eskimo (includes all indigenous populations of the Western hemisphere)',
              4: 'Chinese',
              5: 'Japanese',
              6: 'Filipino',
              7: 'Hawaiian',
              8: 'Korean',
              10: 'Vietnamese',
              11: 'Laotian',
              12: 'Hmong',
              13: 'Kampuchean (including Khmer and Cambodian)',
              14: 'Thai',
              15: 'Asian Indian or Pakistani, NOS',
              16: 'Asian Indian',
              17: 'Pakistani',
              20: 'Micronesian, NOS',
              21: 'Chamorran',
              22: 'Guamanian, NOS',
              25: 'Polynesian, NOS',
              26: 'Tahitian',
              27: 'Samoan',
              28: 'Tongan',
              30: 'Melanesian, NOS',
              31: 'Fiji Islander',
              32: 'New Guinean',
              96: 'Other Asian, including Asian, NOS and Oriental NOS',
              97: 'Pacific Islander, NOS',
              98: 'Other',
              99: 'Unknown'}

######################################################################################

origin_dict = {0 : 'Non-Spanish/Non-Hispanic',
              1: 'Mexican (includes Chicano)',
              2: 'Puerto Rican',
              3: 'Cuban',
              4: 'South or Central American (except Brazil)',
              5: 'Other specified Spanish/Hispanic origin (includes European; excludes Dominican Republic)',
              6: 'Spanish, NOS; Hispanic, NOS; Latino, NOS',
              7: 'Spanish surname only',
              8: 'Dominican Republic',
              9: 'Unknown whether Spanish/Hispanic or not'}

##############################################################################

origrecb_dict = {0: 'Non-Spanish-Hispanic-Latino',
                1: 'Spanish-Hispanic-Latino'}

######################################################################################

census_pov_ind_dict = {1: '0% - <5% poverty',
                      2: '5% to < 10% poverty',
                      3: '10% to <20% poverty',
                      4: '20% to 100% poverty',
                      9: 'Unknown'}

###########################################################################

resnrec_dict = {0: 'Last Dx; Patient always less than 65.',
               1: 'First Dx at age 65 or later'}


####################################################################################

reg_dict = {1: 'San Fransisco', 2: 'Connecticut', 20: 'Detroit', 21: 'Hawaii',
           22: 'Iowa', 23: 'New Mexico', 25: 'Seattle', 26: 'Utah', 27: 'Atlanta',
           31: 'San Jose', 35: 'Los Angeles', 37:'Rural Georgia', 41: 'Greater California',
           42: 'Kentucky', 43: 'Louisiana', 44: 'New Jersey', 47: 'Greater Georgia'}

#####################################################################################

marst1_dict = {1: 'Single (never married)',
              2: 'Married (including common law)',
              3: 'Separated', 4: 'Divorced', 5: 'Widowed',
              6: 'Unmarried or domesntic partner (same sex or opposite sex or unregistered)',
              9: 'Unknown'}

##################################################################################

modx1_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
             7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}


# In[8]:

PRIMSITEdict = {  'C000':   'External upper lip',
                  'C001':    'External lower lip',
                  'C002':    'External lip, NOS',
                  'C003':   'Mucosa of upper lip',
                  'C004':   'Mucosa of lower lip',
                  'C005':    'Mucosa of lip, NOS',
                  'C006':    'Commissure of lip',
                  'C008':    'Overlapping lesion of lip',
                  'C009':    'Lip, NOS',
                  'C019':    'Base of tongue, NOS',
                  'C020':    'Dorsal surface of tongue, NOS',
                  'C021':     'Border of tongue',
                  'C022':    'Ventral surface of tongue, NOS',
                  'C023':    'Anterior 2/3 of tongue, NOS',
                  'C024':    'Lingual tonsil',
                  'C028':    'Overlapping lesion of tongue',
                  'C029':    'Tongue, NOS',
                  'C030':    'Upper Gum',
                  'C031':    'Lower gum',
                  'C039':    'Gum, NOS',
                  'C040':    'Anterior floor of mouth',
                  'C041':    'Lateral floor of mouth',
                  'C048':    'Overlapping lesion of floor of mouth',
                  'C049':    'Floor of mouth, NOS',
                   'C050':    'Hard palate',
                  'C051':    'Soft palate, NOS',
                  'C052':    'Uvula',
                  'C058':    'Overlapping lesion of palate',
                  'C059':    'Palate, NOS',
                  'C060':    'Cheeck mucosa',
                  'C061':    'Vestibule of mouth',
                  'C062':    'Retromolar area',
                  'C068':    'Overlapping lesion of other and unspecified parts of mouth',
                  'C069':    'Mouth, NOS',
                  'C079':    'Parotid gland',
                  'C080':    'Submandibular gland',
                  'C081':    'Sublingual gland',
                  'C088':    'Overlapping lesion of major salivary glands',
                  'C089':    'Major salivary gland, NOS',
                  'C090':    'Tonsillar fossa',
                  'C091':    'Tonsillar pillar',
                  'C098':    'Overlapping lesion of tonsil',
                  'C099':    'Tonsil, NOS',
                   'C100':    'Vallecula',
                  'C101':    'Anterior surface of epiglottis',
                  'C102':    'Lateral wall of oropharynx',
                  'C103':    'Posterior wall of oropharynx',
                  'C104':    'Branchial cleft',
                  'C108':    'Overlapping lesions of oropharynx',
                  'C109':    'Oropharynx, NOS',
                  'C110':    'Superior wall of nasopharynx',
                  'C111':    'Posterior wall of nasopharynx',
                  'C112':    'Lateral wall of nasopharynx',
                  'C113':    'Anterior wall of nasopharynx',
                  'C118':    'Overlapping lesion of nasopharynx',
                  'C119':    'Nasopharynx, NOS',
                  'C129':    'Pyriform sinus',
                  'C130':    'Postcricoid region',
                  'C131':    'Hypopharyngeal aspect of aryepiglottic fold',
                  'C132':    'Posterior wall of hypopharynx',
                  'C138':    'Overlapping lesion of hypopharynx',
                  'C139':    'Hypopharynx, NOS',
                  'C140':    'Pharynx, NOS',
                  'C142':    'Waldeyer ring',
                  'C148':    'Overlapping lesion of lip, oral cavity and pharynx',
                    'C150':    'Cervical esophagus',
                  'C151':    'Thoracic esophagus',
                  'C152':    'Abdominal esophagus',
                  'C153':    'Upper third of esophagus',
                  'C154':    'Middle third of esophagus',
                  'C155':    'Lower third of esophagus',
                  'C158':    'Overlapping lesion of esophagus',
                  'C159':    'Esophagus, NOS',
                  'C160':    'Cardia, NOS',
                  'C161':    'Fundus of stomach',
                  'C162':    'Body of stomach',
                  'C163':    'Gastric antrum',
                  'C164':    'Pylorus',
                  'C165':    'Lesser curvature of stomach, NOS',
                  'C166':    'Greater curvature of stomach, NOS',
                  'C168':    'Overlapping lesion of stomach',
                  'C169':    'Stomach, NOS',
                  'C170':    'Duodenum',
                  'C171':    'Jejunum',
                  'C172':    'Ileum',
                  'C173':    'Meckel diverticulum',
                  'C178':    'Overlapping lesion of small intestine',
                  'C179':    'Small intestine, NOS',
                  'C180':    'Cecum',
                  'C181':    'Appendix',
                  'C182':    'Ascending colon',
                  'C183':    'Hepatic flexure of colon',
                  'C184':    'Transverse colon',
                  'C185':    'Splenic flexure of colon',
                  'C186':    'Descending colon',
                  'C187':    'Sigmoid colon',
                  'C188':    'Overlapping lesion of colon',
                  'C189':    'Colon, NOS',
                   'C199':    'Rectosigmoid junction',
                   'C209':    'Rectum, NOS',
                    'C210':    'Anus, NOS',
                  'C211':    'Anal canal',
                  'C212':    'Cloacogenic zone',
                  'C218':    'Overlapping lesion of rectum, anus and anal canal',
                  'C220':    'Liver',
                  'C221':    'intrahepatic bile duct',
                     'C239':    'Gallbladder',
                      'C240':    'Extrahepatic bile duct',
                  'C241':   'Ampulla of Vater',
                  'C248':    'Overlapping lesion of billiary tract',
                  'C249':    'Billiary tract, NOS',
                   'C250':    'Head of pancreas',
                  'C251':    'Body of pancreas',
                  'C252':    'Tail of pancreas',
                  'C253':    'Pancreatic duct',
                  'C254':    'Islets of Langerhans',
                  'C257':    'Other specified parts of pancreas',
                  'C258':    'Overlapping lesion of pancreas',
                  'C259':    'Pancreas, NOS',
                  'C260':    'Intestinal tract, NOS',
                  'C268':    'Overlapping lesion of digestive system',
                  'C269':    'Gastrointestinal tract, NOS',
                   'C300':    'Nasal cavity',
                  'C301':    'Middle ear',
                     'C310':    'Maxillary sinus',
                  'C311':    'Ethmoid sinus',
                  'C312':    'Frontal sinus',
                  'C313':    'Sphenoid sinus',
                  'C318':    'Overlapping lesion of accessory sinuses',
                  'C319':    'Accessory sinus, NOS',
                    'C320':    'Glottis',
                  'C321':    'Supraglottis',
                  'C322':    'Subglottis',
                  'C323':    'Laryngeal cartilage',
                  'C328':    'Overlapping lesion of larynx',
                  'C329':    'Larynx, NOS',
                  'C339':    'Trachea',
                   'C340':    'Main bronchus',
                  'C341':    'Upper lobe, lung',
                  'C342':    'Middle lobe, lung',
                  'C343':    'Lower lobe, lung',
                  'C348':    'Overlapping lesion of lung',
                  'C349':    'Lung, NOS',
                  'C379':    'Thymus',
                  'C380':    'Heart',
                  'C381':    'Anterior mediastinum',
                  'C382':    'Posterior mediastinum',
                  'C383':    'Mediastinum, NOS',
                  'C384':    'Pleura, NOS',
                  'C388':    'Overlapping lesion of heart, mediastinum and pleura',
                   'C390':    'Upper respiratory tract, NOS',
                  'C398':    'Overlapping lesion of respiratory system and intrathoracic organs',
                  'C399':    'Ill-defined sites within respiratory system',
                     'C400':    'Long bones of upper limb, scapula and associated joints',
                  'C401':    'Short bones of upper limb and associated joints',
                  'C402':    'Long bones of lower limb and associated joints',
                  'C403':    'Short bones of lower limb and associated joints',
                  'C408':    'Overlapping lesion of bones, joints and articular cartilage of limbs',
                  'C409':    'Bone of limb, NOS',
                    'C410':    'Bones of skull and face and associated joints',
                  'C411':    'Mandible',
                  'C412':    'Vertebral column',
                  'C413':    'Rib, sternum, clavicle and associated joints',
                  'C414':    'Pelvic bones, sacrum, coccyx and associated joints',
                  'C418':    'Overlapping lesion of bones, joints and articular cartilage',
                  'C419':    'Bone, NOS',
                    'C420':    'Blood',
                  'C421':    'Bone marrow',
                  'C422':    'Spleen',
                  'C423':    'Reticuloendothelial system, NOS',
                  'C424':    'Hematopoietic system, NOS',
                   'C440':    'Skin of lip, NOS',
                  'C441':    'Eyelid',
                  'C442':    'External ear',
                  'C443':    'Skin of other and unspecified parts of face',
                  'C444':    'Skin of scalp and neck',
                  'C445':    'Skin of trunk',
                  'C446':    'Skin of upper limb and shoulder',
                  'C447':    'Skin of lower limb and hip',
                  'C448':    'Overlapping lesion of skin',
                  'C449':    'Skin, NOS',
                   'C470':    'Peripheral nerves and autonomic nervous system of head, face, and neck',
                  'C471':    'Peripheral nerves and autonomic nervous system of upper limb and shoulder',
                  'C472':    'Peripheral nerves and autonomic nervous system of lower limb and hip',
                  'C473':    'Peripheral nerves and autonomic nervous system of thorax',
                  'C474':    'Peripheral nerves and autonomic nervous system of abdomen',
                  'C475':    'Peripheral nerves and autonomic nervous system of pelvis',
                  'C476':    'Peripheral nerves and autonomic nervous system of trunk, NOS',
                  'C478':    'Overlapping lesion of peripheral nerves and autonomic nervous system',
                  'C479':    'Autonomic nervous system, NOS',
                  'C480':    'Retroperitoneum',
                  'C481':    'Specified parts of peritoneum',
                  'C482':    'Peritoneum, NOS',
                  'C488':    'Overlapping lesion of retroperitoneum and peritoneum',
                  'C490':    'Connective, Subcutaneous and other soft tissues of head, face, and neck',
                  'C491':    'Connective, Subcutaneous and other soft tissues of upper limb and shoulder',
                  'C492':    'Connective, Subcutaneous and other soft tissues of lower limb and hip',
                  'C493':    'Connective, Subcutaneous and other soft tissues of thorax',
                  'C494':    'Connective, Subcutaneous and other soft tissues of abdomen',
                  'C495':    'Connective, Subcutaneous and other soft tissues of pelvis',
                  'C496':    'Connective, Subcutaneous and other soft tissues of trunk, NOS',
                  'C498':    'Overlapping lesion of connective, subcutaneous and other soft tissues',
                  'C499':    'Connective, Subcutaneous and other soft tissues, NOS',
                  'C500':    'Nipple',
                  'C501':    'Central portion of breast',
                  'C502':    'Upper-inner quadrant of breast',
                  'C503':    'Lower-inner quadrant of breast',
                  'C504':    'Upper-outer quadrant of breast',
                  'C505':    'Lower-outer quadrant of breast',
                  'C506':    'Axillary tail of breast',
                  'C508':    'Overlapping lesion of breast',
                  'C509':    'Breast, NOS',
                    'C510':    'Labium majus',
                  'C511':    'Labium minus',
                  'C512':    'Clitorus',
                  'C518':    'Overlapping lesion of vulva',
                  'C519':    'Vulva, NOS',
                  'C529':    'Vagina, NOS',
                    'C530':    'Endocervix',
                  'C531':    'Exocervix',
                  'C538':    'Overlapping lesion of cervix uteri',
                  'C539':    'Cervix uteri',
                  'C540':    'Isthmus uteri',
                  'C541':    'Endometrium',
                  'C542':    'Myometrium',
                  'C543':    'Fundus uteri',
                  'C548':    'Overlapping lesion of corpus uteri',
                  'C549':    'Corpus uteri',
                  'C559':    'Uterus, NOS',
                  'C569':    'Ovary',
                    'C570':    'Fallopian tube',
                  'C571':    'Broad ligament',
                  'C572':    'Round ligament',
                  'C573':    'Parametrium',
                  'C574':    'Uterine adnexa',
                  'C577':    'Other specified parts of female genital organs',
                  'C578':    'Overlapping lesion of female genital organs',
                  'C579':    'Female genital tract, NOS',
                  'C589':    'Placenta',
                  'C600':    'Prepuce',
                  'C601':    'Glans penis',
                  'C602':    'Body of penis',
                  'C608':    'Overlapping lesion of penis',
                  'C609':    'Penis, NOS',
                  'C619':    'Prostate gland',
                  'C620':    'Undescended testis',
                  'C621':    'Descended testis',
                  'C629':    'Testis, NOS',
                   'C630':    'Epididymis',
                  'C631':    'Spermatic cord',
                  'C632':    'Scrotum, NOS',
                  'C637':    'Other specified parts of male genital organs',
                  'C638':    'Overlapping lesion of male genital organs',
                  'C639':    'Male genital organs, NOS',
                   'C649':    'Kidney, NOS',
                  'C659':    'Renal pelvis',
                  'C669':    'Ureter',
                  'C670':    'Trigone of bladder',
                  'C671':    'Dome of bladder',
                  'C672':    'Lateral wall of bladder',
                  'C673':    'Anterior wall of bladder',
                  'C674':    'Posterior wall of bladder',
                  'C675':    'Bladder neck',
                  'C676':    'Ureteric orifice',
                  'C677':    'Urachus',
                  'C678':    'Overlapping lesion of bladder',
                  'C679':    'Bladder, NOS',
                   'C680':    'Urethra',
                  'C681':    'Paraurethral gland',
                  'C688':    'Overlapping lesion of urinary organs',
                  'C689':    'Urinary system, NOS',
                  'C690':    'Conjunctiva',
                  'C691':    'Cornea, NOS',
                  'C692':    'Retina',
                  'C693':    'Choroid',
                  'C694':    'Ciliary body',
                  'C695':    'Lacrimal gland',
                  'C696':    'Orbit, NOS',
                  'C698':    'Overlapping lesion of eye and adnexa',
                  'C699':    'Eye, NOS',
                  'C700':    'Cerebral meninges',
                  'C701':    'Spinal meninges',
                  'C709':    'Meninges, NOS',
                   'C710':    'Cerebrum',
                  'C711':    'Frontal lobe',
                  'C712':    'Temporal lobe',
                  'C713':    'Parietal lobe',
                  'C714':    'Occipital lobe',
                  'C715':    'Ventricle, NOS',
                  'C716':    'Cerebellum, NOS',
                  'C717':    'Brain stem',
                  'C718':    'Overlapping lesion of brain',
                  'C719':    'Brain, NOS',
                   'C720':    'Spinal cord',
                  'C721':    'Cauda equina',
                  'C722':    'Olfactory nerve',
                  'C723':    'Optic nerve',
                  'C724':    'Acoustic nerve',
                  'C725':    'Cranial nerve, NOS',
                  'C728':    'Overlapping lesion of brain and central nervous system',
                  'C729':    'Nervous system, NOS',
                  'C739':    'Thyroid gland',
                  'C740':    'Cortex of adrenal gland',
                  'C741':    'Medulla of adrenal gland',
                  'C749':    'Adrenal gland, NOS',
                   'C750':    'Parathyroid gland',
                  'C751':    'Pituitary gland',
                  'C752':    'Craniopharyngeal duct',
                  'C753':    'Pineal gland',
                  'C754':    'Carotid body',
                  'C755':    'Aortic body and other paraganglia',
                  'C758':    'Overlapping lesion of endocrine glands and related structures',
                  'C759':    'Endocrine gland, NOS',
                  'C760':    'Head, face or neck, NOS',
                  'C761':    'Thorax, NOS',
                  'C762':    'Abdomen, NOS',
                  'C763':    'Pelvis, NOS',
                  'C764':    'Upper limb, NOS',
                  'C765':    'Lower limb, NOS',
                  'C767':    'Other ill-defined sites',
                  'C768':    'Overlapping lesion of ill-defined sites',
                  'C770':    'Lymph nodes of head, face and neck',
                  'C771':    'Intrathoracic lymph nodes',
                  'C772':    'Intra-abdominal lymph nodes',
                  'C773':    'Lymph nodes of axilla or arm',
                  'C774':    'Lymph nodes of inguinal region or leg',
                  'C775':    'Pelvic lymph nodes',
                  'C778':    'Lymph nodes of multiple regions',
                  'C779':    'Lymph node, NOS',
                  'C809':    'Unknown primary site'}


newkeys = [int(k.replace('C','')) for k in PRIMSITEdict.keys()]
primsite_dict_2 = dict()
for k in range(len(newkeys)):
    primsite_dict_2[newkeys[k]] = PRIMSITEdict[list(PRIMSITEdict.keys())[k]]


# In[9]:

lat1_dict = {0: 'Not a paired site',
            1: 'Right: origin of primary',
            2: 'Left: origin of primary',
            3: 'Only one side involved, right or left origin unspecfied',
            4: 'Bilateral involvement, lateral origin unknown; stated to be single primary',
            5: 'Paired site: midline tumor',
            9: 'Paired site, but no information concerning laterality; midline tumor'}


# In[10]:

hist1_dict = {8000: 'Neoplasm Malignant', 8131:'MicropapillaryTransitnlCellCa', 8281: 'Mix Acidoph Basoph Ca',
               8001: 'Tumor cells Malignant', 8140:'Adeno Carcinoma NOS', 8290: 'Oncocytic Adeno/Ca',
               8002: 'Small Cell Tumor', 8141: 'Scirrhous Adenocarcinoma', 8300: 'Basophil Adeno/Ca',
               8003: 'Giant Cell Tumor', 8142: 'Linitis Plastica',  8310: 'Clear Cell Adeno/Ca',
               8004: 'Spindle Cell Tumor', 8143: 'Superficial Spread Adenoca', 8312: 'Renal Cell Adeno/Ca',
               8005: 'Clear Cell Tumor', 8144: 'Intestinal Adenocarcinoma', 8313: 'ClearcellAdenocarcinofibroma',
               8010: 'Carcinoma NOS', 8145: 'Diffuse Adenocarcinoma', 8314: 'Lipid-Rich Carc',
               8011: 'Epithelioma NOS', 8147: 'Basal Cell Adenocarcinoma', 8315: 'Glycogen-Rich Carc',
               8012: 'Large Cell Carcinoma NOS', 8150: 'Islet Cell Adenocarcinoma', 8316: 'Cyst-assoc renal cell Ca',
               8013: 'Lg Cell Neuroendocrine Carc', 8151: 'Beta-Cell Tumor: Malignant', 8317: 'Chromophobe Renal Cell Ca',
               8014: 'Lg Cell Ca rhabdoidphenotype', 8152: 'Alpha-Cell Tumor: Malignant',
               8318: 'Sarcomatoid Renal Cell Ca',
               8015: 'Glassy Cell Carcinoma', 8153: 'G Cell Tumor', 8319: 'Collecting Duct Carcinoma',
               8020: 'Undifferentiated Carcinoma', 8154: 'MixlsletCellExocrine Adenoca',
               8320: 'Granular Cell Carcinoma',
               8021: 'Anaplastic Carcinom', 8155: 'Vipoma', 8322: 'Water-Clear Cell Adeno Ca',
               8022: 'Pleomorphic Carcinoma', 8156: 'Somatostatinoma: Malignant', 8323: 'Mixed Cell Adenoca',
               8030: 'Giant & Spindle Cell Carc', 8157: 'Enteroglucagonoma:Malignant', 8330: 'Follicular Adeno/Ca',
               8031: 'Giant Cell Carcinoma', 8160: 'Bile Duct Adenocarcinoma', 8331: 'Foll Adeno/Ca, WIDif',
               8032: 'Spindle Cell Carc', 8161: 'Bile Duct Cystadenocarcinoma', 8332: 'Mod Dif Foll Adeno/Ca',
               8033: 'Pseudosarcomatous Carcinoma', 8162: 'Klatskins Tumor', 8333: 'Fetal Adenocarcinoma',
               8034: 'Polygonal Cell Carcinoma', 8170: 'Hepatocarcinoma', 8335: 'Foll Ca Minimally invasive',
               8035: 'Carc w/Osteoclast-like Gnt Cel', 8171: 'Fibrolamellar: Hepato Cell Ca', 8337: 'Insular Carcinoma',
               8041: 'Small Cell Carcinoma NOS', 8172: 'Hepatocellular Carc: Scirrhous', 8340: 'Pap & Foll Adeno/Ca',
               8042: 'Oat Cell Carcinoma', 8173: 'Hepatocellular Carc:SpindleCl', 8341: 'Pap Microcarcinoma',
               8043: 'Fusiform Cell Small Cell Ca', 8174: 'Hepatocellular Carc: Clear', 8342: 'PapCarcinoma Oxyphilic cell',
               8044: 'Intermediate Cell Sm Cell Ca', 8175: 'HepatocellularCa:Pleomorphic',
               8343: 'Pap Carcinoma Encapsulated',
               8045: 'Combined Small Cell Carc', 8180: 'Mixed Hepato Bile Duct Ca', 8344: 'Pap Carcinoma Columnar cell',
               8046: 'Non-Small Cell Carc', 8190: 'Trabecular Adeno/Ca', 8345: 'MedullaryCaw/amyloidStroma',
               8050: 'Papillary Carcinoma', 8191: 'Embryonal Adenoma', 8346: 'Mixed Medullary Follicular Ca',
               8051: 'Verrucous Carcinoma', 8200: 'Edenoid Cystic Carcinoma', 8347: 'Mixed Medullary Papillary Ca',
               8052: 'Papillary Squamous Cell Carc', 8201: 'Cribriform Carcinoma NOS', 8350: 'Nonencap Scleros Tumor',
               8070: 'Squamous Cell Carcinoma', 8210: 'Adenoca in Aden Polyp', 8370: 'Adrenal Cort Adeno/Ca',
               8071: 'Squamous Cell Ca Keratiniz', 8211: 'Tubular Adeno/Ca', 8380: 'Endometrioid Aden/Ca',
               8072: 'Squamous Cell Ca Non-Kerit', 8214: 'Parietal Cell Carcinoma', 8381: 'Endometrioid Adenfib',
               8073: 'Squam Cl Ca SmallCell-Non-k', 8215: 'Adenocarcinoma of anal gland', 8382: 'Endom Aden/Ca Secretory',
               8074: 'Squam Cell Ca Spindle Cell', 8220: 'Adenoca Aden Pol Coli', 8383: 'Endom Aden/CaCiliated Cell',
               8075: 'Pseudoglandulr Squam Cell Ca', 8221: 'Adenoca in Muli Ad Pol', 8384: 'Adenocarcinoma Endocervical',
               8076: 'Squam Cell Ca: Microinvasive', 8230: 'Solid Carcinoma NOS', 8390: 'Skin Appendage Carc',
               8077: 'Intraepith Neoplasia Gradelll', 8231: 'Carcinoma Simplex', 8400: 'Sweat Gland Adeno/Ca',
               8078: 'SquamCellCaw/hornformation', 8240: 'Carcinoid Tumor', 8401: 'Apocrine Adenoca',               
               8080: 'Queyrats Erythroplasial', 8241: 'Argentaff Carc Tumor', 8402: 'Nodular Hidradenoma Malig',
               8081: 'Bowens Disease', 8242: 'Enterochromaffin Cell Tumor', 8403: 'Malig Eccrine Spiradenoma',
               8082: 'Lymphoepithelial Carcinoma', 8243: 'Mucocarcinoid Tumor', 8407: 'Sclerosing Sweat Duct Ca',
               8083: 'Basaloid Squamous Cell Carc', 8244: 'Composite Carcinoid', 8408: 'Eccrine Pap Adenocarcinoma',
               8084: 'Squamous Cell Carc:Clear Cel', 8245: 'Adenocarcinoid Tumor', 8409: 'Eccrine Poroma Malignant',
               8090: 'Basal Cell Carcinoma NOS', 8246: 'Neuroendocrine Carc', 8410: 'Sebaceous Adenocarcinoma',
               8091: 'Multicentric Basal Cell Ca', 8247: 'Merkel Cell Carc', 8420: 'Ceruminous Adeno/Ca',
               8092: 'Basal Cell Ca: Morphea', 8249: 'Atypical Carcinoid Tumor', 8430: 'Mucoepidermoid Carc',
               8093: 'Fibroepithelial Basal Cell Ca', 8250: 'Bron Alveol Adeno/Ca', 8440: 'Cystadenocarcinoma NOS',
               8094: 'Basosquamous Carcinoma', 8251: 'Alveolar Adeno/Ca', 8441: 'Serous Cystadenoca',
               8095: 'Metatypical Carcinoma', 8252: 'Bron-AlveolarCanonmucinous', 8442: 'Serous Tumor Low Mal Pot',
               8097: 'Basal Cell Carc: Nodular', 8253: 'Bron-Alveolar Ca Mucinous', 8450: 'Papillocystic Adenoc',
               8098: 'Adenoid Basal Carcinoma', 8254: 'Bron-Alv Ca Mixed/Non muci', 8451: 'Pap Cystad, Low Mal Pot',
               8102: 'Trichilemmocarcinoma', 8255: 'Adeno CA w/mixed subtypes', 8452: 'Solid Pseudopapillary Ca',
               8110: 'Pilomatrix Carcinoma', 8260: 'Papillary Adeno Ca NOS', 8453: 'Intraductal Pap-Mucinous Ca',
               8120: 'Transitional Cell Carcinoma', 8261: 'Adenoca in Villous Aden', 8460: 'Pap Serous Cystadenoc',
               8121: 'Schneiderian Carcinoma', 8262: 'Villous Adenoca', 8461: 'Serous Surface Papill Ca',
               8122: 'SpindleCellTransitionalCellCa', 8263: 'Adenoca in Tubulovill Ad', 8462: 'Pap Ser Tumor Low Mal Pot',
               8123: 'Basaloid Carcinoma', 8270: 'Chromophobe Adeno/Ca', 8470: 'Mucinous Cystadenoca NOS',
               8124: 'Cloacogenic Carcinoma', 8272: 'Pituitary Carcinoma NOS', 8471: 'Pap Mucininous Cystadenoca',
               8130: 'Papillary Transitional Cell Ca', 8280: 'Acidophil Ca/Adenoc', 8472: 'Mucinous Tumor Low Mal P',
               8473: 'Pap Muc Tumor Low Mal Pot',
               8480: 'Mucinous Ca/Adenoca',
               8481: 'Mucin Prod Ca/Adenoc',
               8482: 'Mucin Adeno/Ca Endocervical',
               8490: 'Signet Ring Cell Adeno/Ca',
               8500: 'Duct Adeno/Ca',
               8501: 'Comedocarcinoma',
               8502: 'Juvenile Ca Breast',
               8503: 'Intraduc Pap Adeno/Ca',
               8504: 'Intracyst Pap Aden/Ca',
               8508: 'Cystic Hypersecretory Ca',
               8510: 'Medullary Adeno/Ca',
               8511: 'Medull Ca wAmyl Stroma',
               8512: 'Medull Ca w/Lym Stroma',
               8513: 'Atypical Medullary Ca',
               8514: 'Duct Ca Desmoplastic Type',
               8520: 'Lobular Ca NOS',
               8521: 'Infiltratiny Ductular Ca',
               8522: 'Mix Duct & Lobular Ca',
               8523: 'Infil Duct mixed w/other Ca',
               8524: 'Infil Lobularmixed w/other Ca',
               8525: 'Polymorph low grade Adenoca',
               8530: 'Inflammatory Aden/Ca',
               8540: 'Pagets Dis of Brst',
               8541: 'Paget Dis & Inf Duct Ca Brst',
               8542: 'Pagets Dis. Extra Mam',
               8543: 'Pagets Dis & Intrad Ca Breast',
               8550: 'Acinic Cell Adeno/Ca',
               8551: 'Acinar Cell Cystadenoca',
               8560: 'Adenosquamous Carcimoma',
               8562: 'Epithelial-Myoepithelial Ca',
               8570: 'Adenoacanthoma',
               8471: 'Adenoca: Cart & Oss Met',
               8572: 'Adenoca: Spind Cel Met',
               8573: 'Aden/Ca: Apocr Metap',
               8574: 'Aden/Ca: Neuroendocrine diff',
               8575: 'Metaplastic Carcinoma NOS',
               8576: 'Hepatoid Adenocarcinoma',
               8580: 'Thymoma Malignant NOS',
               8581: 'Thymoma type A Malignant',
               8582: 'Thymoma type AB Malignant',
               8583: 'Thymoma type B1 Malignant',
               8584: 'Thymoma type B2 Malignant',
               8585: 'Thymoma type B3 Malignant',
               8586: 'Thymic Carcinoma NOS',
               8588: 'Spndl Epithelial tumor thymus',
               8589: 'Ca Showing Thymus-like elmn',
               8600: 'Thecoma Malignant',
               8620: 'Granulosa Cell Tum Malignant',
               8630: 'Androblastoma Malignant',
               8631: 'Sertoli-Leydig Cell Tumor Dif',
               8634: 'Sertoli-Leydig Cell Tumor Dif',
               8640: 'Sertoli Cell Carcinoma',
               8650: 'Interstitial Cell Tum',
               8670: 'Steroid Cell Tumor Malignant',
               8680: 'Paraganglioma Malignant',
               8693: 'Extra-Adrenal Paraganglioma',
               8700: 'Pheochromoblastoma Malig',
               8710: 'Glomangiosarcoma',
               8711: 'Glomus Tumor Malignant',
               8720: 'Malignant Melanoma NOS',
               8721: 'Nodular Melanoma',
               8722: 'Balloon Cell Melanoma',
               8723: 'Regressing Melanoma',
               8728: 'Meningeal Melanomatosis',
               8730: 'Amelanotic Melanoma',
               8740: 'Melan Junction Nevus',
               8741: 'Melan in Precanc Melanoma',
               8742: 'Lentigo Maligna Melanoma',
               8743: 'Superficial Spreading Melan',
               8744: 'Acral Lentiginous Melanoma',
               8745: 'Desmoplastic Melanoma',
               8746: 'Mucosal Lentiginous Melan',
               8761: 'Melan Giant Pig Nevus',
               8770: 'Epithelioid/Spindle Cell Melan',
               8771: 'Epithel Cell Melanoma',
               8772: 'Spindle Cell Melanoma NOS',
               8773: 'Spindle Cell Melanoma type A',
               8774: 'Spindle Cell Melanoma type B',
               8780: 'Blue Nevus, Malignant',
               8800: 'Sarcoma NOS',
               8801: 'Spindle Cell Sarcom',
               8802: 'Giant Cell Sa Non Bone',
               8803: 'Small Cell Sarcoma',
               8804: 'Epitheliod Cell Sa',
               8810: 'Fibrosarcoma NOS',
               8811: 'Fibromyxosarcoma',
               8812: 'Periosteal Sarc NOS',
               8832: 'Dermatofibrosarcoma NOS',
               8833: 'Pigmntd Dermatofibrosarcoma',
               8840: 'Myxosarcoma',
               8850: 'Liposarcoma NOS',
               8851: 'Liposarcoma Differentiated',
               8852: 'Myxoid Liposarcoma',
               8853: 'Round Cell Liposarcoma',
               8854: 'Pleomorphic Liposarcoma',
               8855: 'Mixed Liposarcoma',
               8857: 'Fibroblastic Liposarcoma',
               8858: 'Dedifferentiated Liposarcoma',
               8890: 'Leiomyosarcoma NOS',
               8891: 'Epithelioid Leiomyosarcoma',
               8894: 'Angiomyosarcoma',
               8895: 'Myosarcoma',
               8896: 'Myxoid Leiomyosarcoma',
               8900: 'Rhabdomyosarcoma NOS',
               8901: 'PleomorphicRhabdomyosar',
               8902: 'Mixed type Rhabdomyosarco',
               8910: 'Embro Rhabdomyosarcoma',
               8912: 'Spndl cell Rhabdomyosarcoma',
               8920: 'Alveolar Rhabdomyosarcoma',
               8921: 'Rhabdomyosarcomaw/gang dif',
               8930: 'Endometrial Stromal Sarcoma',
               8931: 'Endom Stromal Sarcomalowgr',
               8933: 'Adenosarcoma',
               8934: 'Carcinofibroma',
               8935: 'Stromal Tumor NOS',
               8936: 'Gastrointest Stromal Sarcoma',
               8940: 'Mixed Tumor Malignant NOS',
               8941: 'Carc in Pleomorphic Adenoma',
               8950: 'Mullerian Mixed Tumor',
               8951: 'Mesodermal Mixed Tumor',
               8959: 'Malignant Cystic Nephroma',
               8960: 'Nephroblastoma NOS',
               8963: 'Malignant Rhabdoid Tumor',
               8964: 'Clear Cell Sarcoma of Kidney',
               8970: 'Hepatoblastoma',
               8971: 'Pancreatoblastoma',
               8972: 'Pulmonary Blastoma',
               8973: 'Pleuropulmonary Blastoma',
               8980: 'Carcinosarcoma NOS',
               8981: 'Carcinosarcoma Embryonal',
               8982: 'Malignant Myoepithelioma',
               8990: 'Mesenchymoma Malignant',
               8991: 'Embryonal Sarcoma',
               9000: 'Brenner Tumor Malignant',
               9014: 'Serous Adenocarcinofibroma',
               9015: 'Mucinus Adenocarcinofibroma',
               9020: 'Phyllodes Tumor Malignant',
               9040: 'Synovial Sarcoma NOS',
               9041: 'Synovial Sarcoma Spndl Cell',
               9042: 'Synovial Sarcoma Epithe Cell',
               9043: 'Synovial Sarcoma Biphasic',
               9044: 'Clear Cell Sarcoma NOS',
               9050: 'Mesothelioma Malignant',
               9051: 'Fibrous Mesothelioma Malig',
               9052: 'Epithelioid Mesothelioma Mal',
               9053: 'Mesothelioma Biphasic Malig',
               9060: 'Dysgerminoma',
               9061: 'Seminoma NOS',
               9062: 'Seminoma Anaplastic',
               9063: 'Spermatocytic Seminoma',
               9064: 'Germinoma',
               9065: 'Germ Cell Tumor Nonsemino',
               9070: 'Embryonal Carcinoma NOS',
               9071: 'Yolk Sac Tumor',
               9072: 'Polyembryoma',
               9080: 'Teratoma Malignant NOS',
               9081: 'Teratocarcinoma',
               9082: 'Malignant Teratoma Undiff',
               9083: 'Malignant Teratoma Intermed',
               9084: 'Teratoma w/Malig Transfor',
               9085: 'Mixed Germ Cell Tumor',
               9090: 'Struma Ovarii Malignant',
               9100: 'Choriocarcinoma NOS',
               9101: 'Choriocarcinoma Combnd grm',
               9102: 'Malignant Teratoma Trophob',
               9105: 'Trophoblastic Tumor Epith',
               9110: 'Mesonephroma Malignant',
               9120: 'Hemangiosarcoma',
               9124: 'Kupffer Cell Sarcoma',
               9130: 'Hemangioendothelioma Malig',
               9133: 'Epith Hemangioendothelioma',
               9140: 'Kaposi Sarcoma',
               9150: 'Hemangiopericytoma Malig',
               9170: 'Lymphangiosarcoma',
               9180: 'Osteosarcoma NOS',
               9182: 'Fibroblastic Osteosarcoma',
               9183: 'Telangiectatic Osteosarcoma',
               9184: 'Osteosarcoma in Paget dis/bon',
               9185: 'Small Cell Osteosarcoma',
               9186: 'Central Osteosarcoma', 9503: 'Neuroepithelioma NOS', 9717: 'Intestinal Tcell Lymphoma',
               9187: 'Intraosseous dif Osteosarcoma', 9504: 'Spongioneuroblastoma', 9718: 'Primary cutaneous CD30+Tcel',
               9192: 'Parosteal Osteosarcoma', 9505: 'Ganglioglioma Anaplastic', 9719: 'NK/Tcell Lymph nasal type',
               9193: 'Periosteal Osteosarcoma', 9508: 'Atyp Teratoid/Rhabdoid Tum', 9727: 'Precursor cell Lymph lympho',
               9194: 'High grade Surface Osteosarco', 9510: 'Retinoblastoma NOS', 9728: 'Precursor Bcell Lymph',
               9195: 'Intracortical Osteosarcoma', 9511: 'Retinoblastoma Diff', 9729: 'Precursor Tcell Lymph lymph',
               9220: 'Chondrosarcoma NOS', 9512: 'Retinoblastoma unDiff', 9731: 'Plasmacytoma NOS',
               9221: 'Juxtacortical Chondrosarcoma', 9513: 'Retinoblastoma Diffuse', 9732: 'Multiple Myeloma',
               9230: 'Chondroblastoma Malignant', 9520: 'Olfactory Neurogenic Tumor', 9733: 'Plasma Cell Leukemia',
               9231: 'Myxoid Chondrosarcoma', 9521: 'Olfactory Neurocytoma', 9734: 'Plasmacytoma Extramedullary',
               9240: 'Mesenchymal Chondrosarcom', 9522: 'Olfactory Neuroblastoma', 9740: 'Mast Cell Sarcoma',
               9242: 'Clear Cell Chondrosarcoma', 9523: 'Olfactory Neuroepithelioma', 9741: 'Malignant Mastocytosis',
               9243: 'Dedifferent Chondrosarcoma', 9530: 'Meningioma Malignant', 9742: 'Mast Cell Leukemia',
               9250: 'Giant Cell Tumor bone Malig', 9538: 'Papillary Meningioma', 9750: 'Malignant Histiocytosis',
               9251: 'Malig Giant Cell Tumorsoftprt', 9539: 'Meningeal Sarcomatosis', 9754: 'Langerhans Cell Histiocyt',
               9252: 'Malig TenosynovialGt celltum', 9540: 'Malig Periph NerveSheath tum', 9755: 'Histiocytic Sarcoma',
               9260: 'Ewing Sarcoma', 9560: 'Neurilemoma Malignant', 9756: 'Langerhans Cell Sarcoma',
               9261: 'Adamantinoma of Long Bones', 9561: 'Malig Periph tumrhabdom diff',
               9757: 'Interdigit Dendritic cl Sarcoma',
               9270: 'Odontogenic Tumor Malignant', 9571: 'Perineurioma Malignant',
               9758: 'Follic Dendritic Cell Sarcoma',
               9290: 'Ameloblastic Odontosarcoma', 9580: 'Granular Cell Tumor Malig', 9760: 'Immunoproliferative Disease',
               9310: 'Ameloblastoma Malignant', 9581: 'Alveolar Soft Part Sarcoma', 9761: 'Waldenstrom Macroglobuline',
               9330: 'Ameloblastic Fibrosarcoma', 9590: 'Malignant Lymphoma NOS', 9762: 'Heavy Chain Disease NOS',
               9342: 'Odontogenic Carcinosarcoma', 9591: 'Malig Lymphoma nonHodgkin',
               9764: 'Immunoprolifr sm intestine dis',
               9362: 'Pineoblastoma', 9596: 'Composite Hodgkin nonHodg', 9800: 'Leukemia NOS',
               9364: 'Perip Neuroectodermal Tumor', 9650: 'Hodgkin Lymphoma NOS',  9801: 'Acute Leukemia NOS',
               9365: 'Askin Tumor', 9651: 'Hodg Lymph Lympho-rich', 9805: 'Acute Biphenotypic Leukemia',
               9370: 'Chordoma NOS', 9652: 'Hodg Lymph mixedCellularity', 9820: 'Lymphoid Leukemia NOS',
               9371: 'Chondroid Chordoma', 9653: 'Hodg Lymph Lymph depletion', 9823: 'Bcell chronic lymph Leukemia',
               9372: 'Dedifferentiated Chordoma', 9654: 'Hodg Lymph diffuse fibrosis', 9826: 'Burkitt Cell Leukemia',
               9380: 'Glioma Malignant', 9655: 'Hodg Lymph depletion reticul', 9827: 'Adult Tcell Leukemia/Lymph',
               9381: 'Gliomatosis Cerebri', 9659: 'Hodg Lymph nod lymph pred', 9832: 'Prolymphocytic Leukemia',
               9382: 'Mixed Glioma', 9661: 'Hodgkin Granuloma', 9833: 'Prolymphocytic Leukem B cell',
               9390: 'Choroid Plexus Carcinoma', 9662: 'Hodgkin Sarcoma', 9834: 'Prolymphocytic Leukem T cell',
               9391: 'Ependymoma NOS', 9663: 'Hodgkin Lymph nod Sclerosis', 9835: 'Precursor cell Lymph Leuke',
               9392: 'Ependymoma Anaplastic', 9664: 'Hodg Lymph nod Sclcellphase', 9836: 'Precursor Bcell Lymph Leuk',
               9393: 'Papillary Ependymoma', 9665: 'Hodg Lymph nod scl grade 1', 9837: 'Precursor Tcell Lymph Leuk',
               9400: 'Astrocytoma NOS', 9667: 'Hodg Lymph nod scl grade 2', 9840: 'Acute myeloid Leukemia M6',
               9401: 'Astrocytoma Anaplastic', 9670: 'Malig Lymph small B lymph', 9860: 'Myeloid Leukemia NOS',
               9410: 'Protoplasmic Astrocytoma', 9671: 'Malig Lymph Lymphoplas', 9861: 'Acute Myeloid Leukemia NOS',
               9411: 'Gemistocytic Astrocytoma', 9673: 'Mantle Cell Lymphoma', 9863: 'Chronic Myeloid Leukemia',
               9420: 'Fibrillary Astrocytoma', 9675: 'Malig Lymph sm/lg cell diffus', 9866: 'Acute Promyelocytic Leuk',
               9423: 'Polar Spongioblastoma', 9678: 'Primary effusion Lymphoma', 9867: 'Acute Myelomonocytic Leuk',
               9424: 'Pleomor Xanthoastrocytoma', 9679: 'Mediastinal Lg B cell Lymph', 9870: 'Acute Basophilic Leukemia',
               9430: 'Astroblastoma', 9680: 'Malig Lymph Lg B cell diffuse', 9871: 'Acute Myeloid Leuk abno mar',
               9440: 'Glioblastoma NOS', 9684: 'Malig LymphlgBcelldifimmun', 9872: 'Acute Myeloid Leuk Min Diff',
               9441: 'Giant Cell Glioblastoma', 9687: 'Burkitt Lymphoma NOS', 9873: 'Acute Myeloid Leuk wo matur',
               9442: 'Gliosarcoma', 9689: 'Splenic Marginal zoneB lymph', 9874: 'Acute Myeloid Leuk w/matur',
               9450: 'Oligodendroglioma NOS', 9690: 'Follicular Lymphoma NOS', 9875: 'Myelod Leuk BCR/ABL pos',
               9451: 'Oligodendroglioma Anaplastic', 9691: 'Follicular Lymphoma Grade 2', 9876: 'Myelod Leuk BCR/ABL neg',
               9460: 'Oligodendroblastoma', 9695: 'Follicular Lymph Grade 1', 9891: 'Acute Monocytic Leukemia',
               9470: 'Medulloblastoma NOS', 9698: 'Follicular Lymph Grade 3', 9895: 'Myeloid Leuk w/multi dysplas',
               9471: 'Desmoplastic Nodular Medull', 9699: 'Marginal zone B cell Lymph',
               9896: 'Acute Myeloid Leuk t(8;21)',
               9472: 'Medullomyoblastoma', 9700: 'Mycosis Fungoides', 9897: 'Myeloid Leuk 11q23 abnorm',
               9473: 'Prim Neuroectodermal Tumor', 9701: 'Sezary Syndrome', 9910: 'Acute Megakaryoblastic Leuk',
               9474: 'Large Cell Medulloblastoma', 9702: 'Mature T-Cell Lymph NOS', 9920: 'Therapy-related myeloid Leuk',
               9480: 'Cerebellar Sarcoma NOS', 9705: 'Angioimmunoblastic Tcell', 9930: 'Myeloid Sarcoma',
               9490: 'Ganglioneuroblastoma', 9708: 'Subcut Panniculitis Tcell lymp', 9931: 'Acute Panmyelosis w/myelofi',
               9500: 'Neuroblastoma NOS', 9709: 'Cutaneous Tcell lymph NOS', 9940: 'Hairy Cell Leukamia',
               9501: 'Medulloepithelioma NOS', 9714: 'Anaplastic lg cell Lymph', 9945: 'Chronic myelomonocytic Leuk',
               9502: 'Teratoid Medulloepithelioma', 9716: 'Hepatosplenic Cell Lymphoma',
               9946: 'Juven myelomonocytic Leuk',
               9948: 'Aggressive NK-Cell Leukemia',
               9950: 'Polycythemia Vera',
               9960: 'Chronic Myeloproliferative dis',
               9961: 'Myeloscl w/myeloid Metaplas',
               9962: 'Essential Thrombocyethemia',
               9963: 'Chronic Neutrophilic Leuk',
               9964: 'Hypereosinophilic Syndrome',
               9980: 'Refractory Anemia',
               9982: 'Refract Anemia w/sideroblast',
               9983: 'Refract Anemia w/exces blast',
               9984: 'Refract Anem exces blast in trs',
               9985: 'Refract cytopenia multi dyspla',
               9986: 'Myelodysplastic Syndrome 5q',
               9987: 'Therapy related Myelod Syndr',
9989: 'Myelody Syndrome NOS'} 


##################################################################################

beh1_dict = {0: 'Benign (Reportable for intracranial and CNS sites only) Uncertain '+            'whether benign or malignant, borderline malignancy, low',
            1: 'Malignant potential, and uncertain malignant potential '+\
            '(Reportable for intracranial and CNS sites only)',
            2: 'Carcinoma in situ; intraepithelial; noninfiltrating; noninvasive',
            3: 'Malignant, primary site (invasive)'}

#######################################################################################

grade_dict = {1: 'Grade I; grade i, grade 1; well differentiated; differentiated, NOS',
             2: 'Grade II; grade ii; grade 2; moderately differentiated; moderately differentiated; intermediate differentiation',
             3: 'Grade III; grade iii, grade 3; poorly differentiated; differentiated',
             4: 'Grade IV; grade iv; grade 4; undifferentaited; anaplastic',
             5: 'T-cell; T-precursor',
             6: 'B-cell; Pre-B;B-Precursor',
             7: 'Null cell; Non T-non B',
             8: 'N K cell (natural killer cell)',
             9: 'cell type not determined, not stated or not applicable'}


# In[11]:

dxconf1_dict = {1: 'Positive histology',
               2: 'Positive cytology',
               3: 'Positive histology AND immunophenotyping AND/OR pos genetic studies', 
               4: 'Positive microscopic confirmation, method not specified',
               5: 'Postitive laboratory test/marker study',
               6: 'Direct visualization without microscopic confirmation',
               7: 'Radiology and other imaging techniques without microscopic confirmation',
               8: 'Clinical diagnosis only (other than 5,6, or 7)',
               9: 'Unknown whether microscopically confirmedl; death certificate only'}

#######################################################################################################

cstum1_dict = {0.0: 'Indicates no mass or no tumor found; for example, when a ' +                 'tumor of a stated primary site is not found, but the tumor has metastasized',
                989.0: '989 millimeters or larger',
                990.0: 'Microscopic focus or foci only; no size of focus is given',
                991.0: 'Described as less than 1 cm',
                992.0: 'Described as less than 2 cm',
                993.0: 'Described as less than 3 cm',
                994.0: 'Described as less than 4 cm',
                995.0: 'Described as less than 5 cm',
                999.0: 'Unknown; size not stated; not stated in patient record',
                888.0: 'Not applicable'}

#####################################################################################################

dajcct1_dict = {99.0: 'TX',
              0.0: 'T0',
              1.0: 'Ta',
              5.0: 'Tis',
              6.0: 'Tispu (Urethra only)',
              7.0: 'Tispd (Urethra only)',
              10.0: 'T1',
              11.0: 'T1mic',
              19.0: 'T! NOS',
              12.0: 'T1a',
              13.0: 'T1a1',
              14.0: 'T1a2',
              15.0: 'T1b',
              16.0: 'T1b1',
              17.0: 'T1b2',
              18.0: 'T1c',
              20.0: 'T2',
              29.0: 'T2 NOS',
              21.0: 'T2a',
              22.0: 'T2b',
              23.0: 'T2c',
              30.0: 'T3',
              39.0: 'T3 NOS',
              31.0: 'T3a',
              32.0: 'T3b',
              33.0: 'T3c',
              40.0: 'T4',
              49.0: 'T4 NOS',
              41.0: 'T4a',
              42.0: 'T4b',
              43.0: 'T4c',
              44.0: 'T4d',
              80.0: 'T1aNOS',
              81.0: 'T1b NOS',
              88.0: 'Not applicable'}

##############################################################################

dajccn1_dict = {99.0: 'NX',
             0.0: 'N0',
             1.0: 'N0(i-)',
             2.0: 'N0(i+)',
             3.0: 'N0(mol-)',
             4.0: 'N0(mol+)',
             10.0: 'N1',
             19.0: 'N1 NOS',
             11.0: 'N1a', 12.0: 'N1b', 13.0: 'N1c',
             18.0: 'N1mi', 20.0: 'N2', 29.0: 'N2 NOS',
             21.0: 'N2a', 22.0: 'N2b', 23.0: 'N2c',
             30.0: 'N3', 39.0: 'N3 NOS', 31.0: 'N3a',
             32.0: 'N3b', 33.0: 'N3c', 88.0: 'Not applicable'}

############################################################################################

dajccm1_dict =  {99.0: 'MX',
              0.0: 'M0',
              10.0: 'M1', 11.0: 'M1a', 12.0: 'M1b',
              13.0: 'M1c', 19.0: 'M1 NOS', 88.0: 'Not applicable'}

##############################################################################################


dajccstg1_dict =  {0.0: 'Stage 0', 1.0: 'Stage 0a',
                2.0: 'Stage 0is', 10.0: 'Stage I',
                11.0: 'Stage I NOS', 12.0: 'Stage IA',
                13.0: 'Stage IA1', 14.0: 'Stage IA2',
                15.0: 'Stage IB', 16.0: 'Stage IB1',
                17.0: 'Stage IB2', 18.0: 'Stage IC',
                19.0: 'Stage IS', 23.0: 'Stage ISA (lymphoma only)',
                24.0: 'Stage ISB (lymphoma only)',
                20.0: 'Stage IEA (lymphoma only)',
                21.0: 'Stage IEB (lymphoma only)',
                22.0: 'Stage IE (lymphoma only)',
                30.0: 'Stage II', 31.0: 'Stage II NOS',
                32.0: 'Stage IIA', 33.0: 'Stage IIB',
                34.0: 'Stage IIC', 35.0: 'Stage IIEA (lymphoma only)',
                36.0: 'Stage IIEB (lymphoma only)',
                37.0: 'Stage IIE (lymphoma only)',
                38.0: 'Stage IISA (lymphoma only)',
                39.0: 'Stage IISB (lymphoma only)',
                40.0: 'Stage IIS (lymphoma only)',
                41.0: 'Stage IIESA (lymphoma only)',
                42.0: 'Stage IIESB (lymphoma only)',
                43.0: 'Stage IIES (lymphoma only)',
                50.0: 'Stage III',
                51.0: 'Stage III NOS',
                52.0: 'Stage IIIA',
                53.0: 'STage IIIB',
                54.0: 'Stage IIIC',
                55.0: 'Stage IIIEA (lymphoma only)',
                56.0: 'Stage IIIEB (lymphoma only)',
                57.0: 'Stage IIIE (lymphoma only)',
                58.0: 'Stage IIISA (lymphoma only)',
                59.0: 'Stage IIISB (lymphoma only)',
                60.0: 'Stage IIIS (lymphoma only)',
                61.0: 'Stage IIIESA (lymphoma only)',
                62.0: 'Stage IIIESB (lymphoma only)',
                63.0: 'Stage IIIES (lymphoma only)',
                70.0: 'Stage IV',
                71.0: 'Stage IV NOS',
                72.0: 'Stage IVA',
                73.0: 'Stage IVB',
                74.0: 'Stage IVC',
                88.0: 'Not applicable',
                90.0: 'Stage Occult',
                99.0: 'Stage Unknown'}

######################################################################################

dss77s1_dict =  {0.0: 'In Situ',
                1.0: 'Localized',
                2.0: 'Regional, direct extension',
                3.0: 'Regional, lymph nodes only',
                4.0: 'Regional, extension and nodes',
                5.0: 'Regional, NOS',
                7.0: 'Distant',
                8.0: 'Not applicable',
                9.0: 'Unknown/Unstaged'}

#############################################################################################

dss00s1_dict =  {0.0: 'In Situ',
                1.0: 'Localized',
                2.0: 'Regional, direct extension',
                3.0: 'Regional, lymph nodes only',
                4.0: 'Regional, extension and nodes',
                5.0: 'Regional, NOS',
                7.0: 'Distant',
                8.0: 'Not applicable',
                9.0: 'Unknown/Unstaged'}

##############################################################################################


sxprif1_dict = {0.0: 'None; no surgical procedure of primary site; '+
                'diagnosed at autopsy only',
                90.0: 'Surgery, NOS. A surgical procedure to the primary '+
                'site was done, but no information on the type of surgical '+
                'procedure is provided',
                98.0: 'Special codes for hematopoietic, reticuloendothelial, '+
                'immunoproliferative, myeloproliferative diseases; ill-defined '+
                'sites; and unknown primaries (See site-specific codes for the sites '+
                'and histologies), except death certificate only',
                99.0: 'Unknown if surgery performed; death certificate only'}

##################################################################################################

sxscof1_dict =  {0.0: 'No regional lymph nodes removed or aspirated; '+
                'diagnosed at autopsy',
                1.0: 'Biopsy or aspiration of regional lymph node; NOS',
                2.0: 'Sentinel lymph node biopsy [only]',
                3.0: 'Number of regional lymph nodes removed unknown, '+
                'not stated; regional lymph nodes removed, NOS',
                4.0: '1 to 3 regional lymph nodes removed',
                5.0: '4 or more regional lymph nodes removed',
                6.0: 'Sentinel node biopsy and code 3, 4, or 5 at same '+
                'time or timing not noted',
                7.0: 'Sentinel node biopsy and code 3, 4, or 5 at '+
                'different times',
                9.0: 'Unknown or not applicable; death certificate only'}

##########################################################################################################


sxsitf1_dict = {0.0: 'None; diagnosed at autopsy',
                1.0: 'Nonprimary surgical procedure performed',
                2.0: 'Nonprimary surgical procedure to other regional sites',
                3.0: 'Nnoprimary surgical procedure to distant lymph node(s)',
                4.0: 'Nonprimary surgical procedure to distant site',
                5.0: 'Combination of codes 2,3, or 4',
                9.0: 'Unknown; death certificate only'}

#########################################################################################################

rad1_dict =  {0: 'None; diagnosed at autopsy',
               1: 'Beam radiation',
               2: 'Radioactive implants',
               3: 'Radioisotopes',
               4: 'Combination of beam radiation with radioactive '+
               'implants or radioisotopes',
               5: 'Radiation, NOS - method or source not provided',
               6: 'Other radiation',
               7: "Patient or patient's guardian refused radiation therapy",
               8: 'Radiation recommended, unknown if administered',
               9: 'Unknown if radiation administered'}

##############################################################################################

e10pn1_dict = {0: 'All nodes examined are negative',
              90: '90 or more number of nodes postiive',
              95: 'Positive aspiration of lymph node(s) was performed',
              97: 'Positive nodes are documented, but number is unspecified',
              98: 'No nodes were examined',
              99: 'Unknown whether nodes are positive; not applicable; not stated in patient record'}

###########################################################################################################

nosrg1_dict = {0 : 'Surgery performed',
              1: 'Surgery not recommended',
              2: 'Contraindicated due to other conditions; Autopsy Only case',
              5: 'Patient died before recommended surgery',
              6: 'Unknown reason for no surgery',
              7: "Patient or patient's guardian refused",
              8: 'Recommended, unknown if done',
              9: 'Unknown if surgery performed; Death Certificate Only case'}

#############################################################################################################

radsurg1_dict = {0: 'No radiation and/or surgery as defined above',
                2: 'Radiation before surgery',
                3: 'Radiation after surgery',
                4: 'Radiation both before and after surgery',
                5: 'Intraoperative radiation therapy',
                6: 'Intraoperative radiation with other radiation before or '+ \
                'after surgery',
                9: 'Sequence unknown, but both surgery and radiation were given'}

##############################################################################################################

ager1_dict = {0: 'Age 00',
             1: 'Ages 01-04',
             2: 'Ages 05-09',
             3: 'Ages 10-14',
             4: 'Ages 15-19',
             5: 'Ages 20-24',
             6: 'Ages 25-29',
             7: 'Ages 30-34',
             8: 'Ages 35-39',
             9: 'Ages 40-44',
             10: 'Ages 45-49',
             11: 'Ages 50-54',
             12: 'Ages 55-59',
             13: 'Ages 60-64',
             14: 'Ages 65-69',
             15: 'Ages 70-74',
             16: 'Ages 75-79',
             17: 'Ages 80-84',
             18: 'Ages 85+',
             99: 'Unknown Age'}


# In[12]:

siterwho1_dict = {20010: 'Lip', 20020: 'Tongue', 20030: 'Salivary Gland',
                20040: 'Floor of Mouth', 20050: 'Gum and Other Mouth',
                20060: 'Nasopharynx', 20070: 'Tonsil', 20080: 'Oropharynx',
                20090: 'Hypopharynx', 20100: 'Other Oral Cavity and Pharynx',
                21010: 'Esophagus', 21020: 'Stomach', 21030: 'Small Intestine',
                21041: 'Cecum', 21042: 'Appendix', 21043: 'Ascending Colon',
                21044: 'Hepatic Flexure', 21045: 'Transverse Colon',
                21046: 'Splenic Flexure', 21047: 'Descending Colon',
                21048: 'Sigmoid Colon', 21049: 'Large Intestine, NOS',
                21051: 'Rectosigmoid Junction', 21052: 'Rectum',
                21060: 'Anus, Anal Canal and Anorectum', 21071: 'Liver',
                21072: 'Intrahepatic Bile Duct', 21080: 'Gallbladder',
                21090: 'Other Bilary', 21100: 'Pancreas',
                21110: 'Retroperitoneum', 21120: 'Peritoneum, Omentum and Mesentery',
                21130: 'Other Digestive Organs', 22010: 'Nose, Nasal Cavity and Middle Ear',
                22020: 'Larynx', 22030: 'Lung and Bronchus', 22050: 'Pleura',
                22060: 'Trachea, Mediastinum and Other Respiratory Organs',
                23000: 'Bones and Joints', 24000: 'Soft Tissue including Heart',
                25010: 'Melanoma of the Skin', 25020: 'Other Non-Epithelial Skin',
                26000: 'Breast', 27010: 'Cervix Uteri', 27020: 'Corpus Uteri',
                27030: 'Uterus, NOS', 27040: 'Ovary', 27050: 'Vagina',
                27060: 'Vulva', 27070: 'Other Female Genital Organs',
                28010: 'Prostate', 28020: 'Testis', 28030: 'Penis',
                28040: 'Other Male Genital Organs', 29010: 'Urinary Bladder',
                29020: 'Kidney and Renal Pelvis', 29030: 'Ureter',
                29040: 'Other Urinary Organs', 30000: 'Eye and Orbit',
                31010: 'Brain', 31040: 'Cranial Nerves Other Nervous System',
                32010: 'Thyroid', 32020: 'Other Endocrine including Thymus',
                33011: 'Hodkin - Nodal', 33012: 'Hodgkin - Extranodal',
                33041: 'NHL - Nodal', 33042: 'NHL - Extranodal',
                34000: 'Myeloma', 35011: 'Acute Lymphocytic Leukemia',
                35012: 'Chronic Lymphocytic Leukemia', 35013: 'Other Lymphocytic Leukemia',
                35021: 'Acute Myeloid Leukemia', 35031: 'Acute Monocytic Leukemia',
                35022: 'Chronic Myeloid Leukemia',
                35023: 'Other Myeloid/Monocytic Leukemia',
                35041: 'Other Acute Leukemia', 35043: 'Aleukemic, subleukemic and NOS',
                36010: 'Mesothelioma', 36020: 'Kaposi Sarcoma',
                37000: 'Miscellaneous', 99999: 'Invalid'}

###########################################################################################################

iccc3who1_dict = {11: 'Lymphoid leukemias',
                12: 'Acute myeloid leukemias',
                13: 'Chronic myeloproliferative diseases',
                14: 'Myelodysplastic syndrome and other myeloproliferative diseases',
                15: 'Unspecified and other specified leukemias',
                21: 'Hodgkin lymphomas',
                22: 'Non-Hodgkin lymphomas (except Burkitt lymphoma)',
                23: 'Burkitt lymphoma',
                24: 'Miscellaneous lymphoreticular neoplasms',
                25: 'Unspecified lymphomas',
                31: 'Ependymomas and choroid plexus tumor',
                32: 'Astrocytomas',
                33: 'Intracranial and intraspinal embryonal tumors',
                34: 'Other gliomas',
                35: 'Other specified intracranial and intraspinal neoplasms',
                36: 'Unspecified intracranial and intraspinal neoplasms',
                41: 'Neuroblastoma and ganglioneuroblastoma',
                42: 'Other peripheral nervous cell tumors',
                50: 'Retinoblastoma',
                61: 'Nephroblastoma and other nonepithelial renal tumors',
                62: 'Renal carcinomas',
                63: 'Unspecified malignant renal tumors',
                71: 'Hepatoblastoma',
                72: 'Hepatic carcinomas',
                73: 'Unspecified malignant hepatic tumors',
                81: 'Osteosarcomas',
                82: 'Chondrosarcomas',
                83: 'Ewing tumor and related sarcomas of bone',
                84: 'Other specified malignant bone tumors',
                85: 'Unspecified malignant bone tumors',
                91: 'Rhabdomyosarcomas',
                92: 'Fibrosarcomas, peripheral nerve sheath tumors, and other fibrous neoplasms',
                93: 'Kaposi sarcoma',
                94: 'Other specified soft tissue sarcomas',
                95: 'Unspecified soft tissue sarcomas',
                101: 'Intracranial and intraspinal germ cell tumors',
                102: 'Malignant extracranial and extragonadal germ cell tumors',
                103: 'Malignant gonadal germ cell tumors',
                104: 'Gonadal carcinomas',
                105: 'Other and unspecified malignant gonadal tumors',
                111: 'Adrenocortical carcinomas',
                112: 'Thyroid carcinomas',
                113: 'Nasopharyngeal carcinomas',
                114: 'Malignant melanomas',
                115: 'Skin carcinomas',
                116: 'Other and unspecified carcinomas',
                121: 'Other specified malignant tumors',
                122: 'Other unspecified malignant tumors'}

################################################################################################

histrec1_dict = {0: '8000-8009: unspecified neoplasms',
               1: '8010-8049: epithelial neplasms, NOS',
               2: '8050-8089: squamous cell neplasms',
               3: '8090-8119: basal cell neplasms',
               4: '8120-8139: transitional cell papilomas and carcinomas',
               5: '8140-8398: adenomas and adenocarcinomas',
               6: '8390-8429: adnexal and skin appendage neoplasms',
               7: '8430-8439: mucoepidermoid neoplasms',
               8: '8440-8499: cystic, mucinous and serous neoplasms',
               9: '8500-8549: ductal and lobular neoplasms',
               10: '8550-8559: acinar cell neoplasms',
               11: '8560-8579: complex epithelial neoplasms',
               12: '8580-8589: thymic epithelial neoplasms',
               13: '8590-8679: specialized gonadal neoplasms',
               14: '8680-8719: paragangliomas and glumus tumors',
               15: '8720-8799: nevi and melanomas',
               16: '8800-8809: soft tissue tumors and sarcomas, NOS',
               17: '8810-8839: fibromatous neoplasms',
               18: '8840-8849: myxomatous neoplasms',
               19: '8850-8889: lipomatous neoplasms',
               20: '8890-8929: myomatous neoplasms',
               21: '8930-8999: complex mixed and stromal neoplasms',
               22: '9000-9039: fibroepithelial neoplasms',
               23: '9040-9049: synovial-like neoplasms',
               24: '9050-9059: mesothelial neoplasms',
               25: '9060-9099: germ cell neoplasms',
               26: '9100-9109: trophoblastic neoplasms',
               27: '9110-9119: mesonephromas',
               28: '9120-9169: blood vessel tumors',
               29: '9170-9179: lymphatic vessel tumors',
               30: '9180-9249: osseous and chondromatous neoplasms',
               31: '9250-9259: giant cell tumors',
               32: '9260-9269: miscellaneous bone tumors (C40._,C41._)',
               33: '9270-9349: odontogenic tumors (C41._)',
               34: '9350-9379: miscellaneous tumors',
               35: '9380-9489: gliomas',
               36: '9490-9529: neuroepitheliomatous neoplasms',
               37: '9530-9539: meningiomas',
               38: '9540-9579: nerve sheath tumors',
               39: '9580-9589: granular cell tumors & alveolar soft part sarcomas',
               40: '9590-9599: malignant lymphomas, NOS or diffuse',
               41: '9650-9669: hodgkin lymphomas',
               42: '9670-9699: nhl - mature b-cell lymphomas',
               43: '9700-9719: nhl - mature t and nk-cell lymphomas',
               44: '9720-9729: nhl - precursor cell lymphoblastic lymphoma',
               45: '9730-9739: plasma cell tumors',
               46: '9740-9749: mast cell tumors',
               47: '9750-9759: neoplasms of histiocytes and accessorty lymphoid cells',
               48: '9760-9769: immunoproliferative diseases',
               49: '9800-9805: leukemias, nos',
               50: '9820-9839: lymphoid leukemias (C42.1)',
               51: '9840-9939: myeloid leukemias (C42.1)',
               52: '9940-9949: other leukemias (C42.1)',
               53: '9950-9969: chronic myeloproliferative disorders (C42.1)',
               54: '9970-9979: other mematologic disorders',
               55: '9980-9989: myelodysplastic syndrome',
               98: 'other'}


##########################################################################################


hisrcb1_dict = {1: 'Diffuse astrocytoma (protoplasma, fibrillary)',
                2: 'Anaplastic astrocytoma',
                3: 'Glioblastoma',
                4: 'Pilocytic astrocytoma',
                5: 'Unique astrocytoma variants',
                6: 'Oligodendroglioma',
                7: 'Anaplastic oligodendroglioma',
                8: 'Ependymoma/anaplastic ependymoma',
                9: 'Ependymoma variants',
                10: 'Mixed glioma',
                11: 'Astrocytoma, NOS',
                12: 'Glioma, NOS',
                13: 'Choroid plexus',
                14: 'Neuroepithelial',
                15: 'Benign & malignant neuronal/glial, neuronal & mixed',
                16: 'Pineal parenchymal',
                17: 'Embryonal/primitive/medulloblastoma',
                18: 'Nerve sheath, benign and malignant',
                19: 'Meningioma, benign and malignant',
                20: 'Other mesenchymal, benign and malignant',
                21: 'Hemangioma and hemagioblastoma',
                22: 'Lymphoma',
                23: 'Germ cell tumors, cysts, and heterotopias',
                24: 'Chordoma/chondrosarcoma',
                25: 'Pituitary',
                26: 'Craniopharyngioma',
                27: 'Neoplasm, unspecified, benign and malignant',
                97: 'Other Brain Histologies',
                98: 'Not Brain'}


# In[13]:

cssch1_dict = {1: 'LipUpper',
                2: 'LipLower',
                3: 'OthLip',
                4: 'BaseTongue',
                5: 'AntTongue',
                6: 'GumUpper',
                7: 'GumLower',
                8: 'OthGum',
                9: 'FOM',
                10: 'HardPalate',
                11: 'SoftPalate',
                12: 'OthMouth',
                13: 'BuccalMucosa',
                14: 'ParotidGland',
                15: 'SubmandibularGland',
                16: 'OthSalivary',
                17: 'Oropharynx',
                18: 'AntEpiglottis',
                19: 'Nasopharynx',
                20: 'Hypopharynx',
                21: 'OthPharynx',
                22: 'Esophagus', 23: 'Stomach', 24: 'SmallIntestine',
                25: 'Colon', 26: 'Rectum', 27: 'Anus', 28: 'Liver',
                29: 'Gallbladder', 30: 'ExtraHepaticDucts',
                31: 'Ampulla', 32: 'OthBiliary',
                33: 'PancreasHead', 34: 'PancreasBodyTail', 35: 'OthPancreas',
                36: 'OthDigestive', 37: 'NasalCavity', 38: 'MiddleEar',
                39: 'MaxillarySinus', 40: 'EthmoidSinus', 41: 'OthSinus',
                42: 'GlotticLarynx', 43: 'SupralLarynx', 44: 'SubLarynx',
                45: 'OthLarynx', 46: 'Trachea', 47: 'Lung',
                48: 'HeartMediastinum', 49: 'Pleura', 50: 'OthRespiratory',
                51: 'Bone', 52: 'Skin', 53: 'SkinEyeLid',
                54: 'Melanoma', 55: 'MF', 56: 'SoftTissue', 57: 'Peritoneum',
                58: 'Breast', 59: 'Vulva', 60: 'Vagina', 61: 'Cervix',
                62: 'Corpus', 63: 'Ovary', 64: 'FallapianTube', 65: 'OthAdnexa',
                66: 'OthFemaleGen', 67: 'Placenta', 68: 'Penis', 69: 'Prostate',
                70: 'Testis', 71: 'OthMaleGen', 72: 'Scrotum', 73: 'Kidney',
                74: 'RenalPelvis', 75: 'Bladder', 76: 'Urethra', 77: 'OthUrinary',
                78: 'Conjunctiva', 79: 'MelanomaConjunctiva',
                80: 'OthEye', 81: 'MelanomaIrisCiliary', 82: 'MelanomaChoroid',
                83: 'MelanomaOthEye', 84: 'LacrimalGland', 85: 'Orbit',
                86: 'Retinoblastoma', 87: 'Brain', 88: 'OthCNS',
                89: 'Thyroid', 90: 'OthEndrocrine', 91: 'KS', 92: 'Lymphoma',
                93: 'HemeRetic', 94: 'OthIllDef'}

################################################################################################################


insrecpb1_dict = {1: 'Uninsured',
                 2: 'Any Medicaid',
                 3: 'Insured',
                 4: 'Insured/No specifics',
                 5: 'Insurance status unknown',
                 9: 'Not available (Los Angeles)'}

######################################################################################################

payer_dx1_dict = {1: 'Not insured',
                 2: 'Not insured, self pay',
                 10: 'Insurance, NOS',
                 20: 'Private Insurance: Managed care, HMO, or PPO',
                 21: 'Private Insurance: Fee-for-Service',
                 31: 'Medicaid',
                 35: 'Medicaid - Administered through a Managed Care plan',
                 60: 'MEdicare/Medicare, NOS',
                 61: 'Medicare with supplement, NOS',
                 62: 'Medicare - Administered through a Managed Care plan',
                 63: 'Medicare with private supplement',
                 64: 'Medicare with Medicaid eligibility',
                 65: 'TRICARE',
                 66: 'Military',
                 67: 'Veterans Affairs',
                 68: 'Indian/Public Health Service',
                 99: 'Insurance status unknown'}

###################################################################################################


# In[14]:

replacedict = dict()
replacedict['casereg'] = casereg_dict
replacedict['stat_rec'] = stat_rec_dict
replacedict['fivepct'] = fivepct_dict
replacedict['birthm'] = birthm_dict
replacedict['m_sex'] = m_sex_dict
replacedict['race'] = race_dict
replacedict['urbrur'] = urbrur_dict
replacedict['urban'] = urban_dict
replacedict['rac_recb'] = rac_recb_dict
replacedict['rac_recy'] = rac_recy_dict
replacedict['rac_reca'] = rac_reca_dict
replacedict['icd_code'] = icd_code_dict
replacedict['s_sex'] = s_sex_dict
replacedict['nhiade'] = nhiade_dict
replacedict['codpub'] = codpub_dict
replacedict['ser_dodm'] = ser_dodm_dict
replacedict['srace'] = srace_dict
replacedict['origin'] = origin_dict
replacedict['origrecb'] = origrecb_dict
replacedict['census_pov_ind'] = census_pov_ind_dict
replacedict['resnrec'] = resnrec_dict
#####################################################################
replacedict['reg1'] = reg_dict
replacedict['reg2'] = reg_dict
replacedict['reg3'] = reg_dict
replacedict['reg4'] = reg_dict
replacedict['reg5'] = reg_dict
replacedict['reg6'] = reg_dict
replacedict['reg7'] = reg_dict
replacedict['reg8'] = reg_dict
replacedict['reg9'] = reg_dict
replacedict['reg10'] = reg_dict
######################################################################
replacedict['marst1'] = marst1_dict
replacedict['marst2'] = marst1_dict
replacedict['marst3'] = marst1_dict
replacedict['marst4'] = marst1_dict
replacedict['marst5'] = marst1_dict
replacedict['marst6'] = marst1_dict
replacedict['marst7'] = marst1_dict
replacedict['marst8'] = marst1_dict
replacedict['marst9'] = marst1_dict
replacedict['marst10'] = marst1_dict
################################################################################
replacedict['modx1'] = modx1_dict
replacedict['modx2'] = modx1_dict
replacedict['modx3'] = modx1_dict
replacedict['modx4'] = modx1_dict
replacedict['modx5'] = modx1_dict
replacedict['modx6'] = modx1_dict
replacedict['modx7'] = modx1_dict
replacedict['modx8'] = modx1_dict
replacedict['modx9'] = modx1_dict
replacedict['modx10'] = modx1_dict
#####################################################################################
replacedict['site1'] = primsite_dict_2
replacedict['site2'] = primsite_dict_2
replacedict['site3'] = primsite_dict_2
replacedict['site4'] = primsite_dict_2
replacedict['site5'] = primsite_dict_2
replacedict['site6'] = primsite_dict_2
replacedict['site7'] = primsite_dict_2
replacedict['site8'] = primsite_dict_2
replacedict['site9'] = primsite_dict_2
replacedict['site10'] = primsite_dict_2
#################################################################################
replacedict['lat1'] = lat1_dict
replacedict['lat2'] = lat1_dict
replacedict['lat3'] = lat1_dict
replacedict['lat4'] = lat1_dict
replacedict['lat5'] = lat1_dict
replacedict['lat6'] = lat1_dict
replacedict['lat7'] = lat1_dict
replacedict['lat8'] = lat1_dict
replacedict['lat9'] = lat1_dict
replacedict['lat10'] = lat1_dict
#####################################################################################
replacedict['hist1'] = hist1_dict
replacedict['hist2'] = hist1_dict
replacedict['hist3'] = hist1_dict
replacedict['hist4'] = hist1_dict
replacedict['hist5'] = hist1_dict
replacedict['hist6'] = hist1_dict
replacedict['hist7'] = hist1_dict
replacedict['hist8'] = hist1_dict
replacedict['hist9'] = hist1_dict
replacedict['hist10'] = hist1_dict
###################################################################################
replacedict['beh1'] = beh1_dict
replacedict['beh2'] = beh1_dict
replacedict['beh3'] = beh1_dict
replacedict['beh4'] = beh1_dict
replacedict['beh5'] = beh1_dict
replacedict['beh6'] = beh1_dict
replacedict['beh7'] = beh1_dict
replacedict['beh8'] = beh1_dict
replacedict['beh9'] = beh1_dict
replacedict['beh10'] = beh1_dict
########################################################################################
replacedict['grade1'] = grade_dict
replacedict['grade2'] = grade_dict
replacedict['grade3'] = grade_dict
replacedict['grade4'] = grade_dict
replacedict['grade5'] = grade_dict
replacedict['grade6'] = grade_dict
replacedict['grade7'] = grade_dict
replacedict['grade8'] = grade_dict
replacedict['grade9'] = grade_dict
replacedict['grade10'] = grade_dict
####################################################################################
replacedict['dxconf1'] = dxconf1_dict
replacedict['dxconf2'] = dxconf1_dict
replacedict['dxconf3'] = dxconf1_dict
replacedict['dxconf4'] = dxconf1_dict
replacedict['dxconf5'] = dxconf1_dict
replacedict['dxconf6'] = dxconf1_dict
replacedict['dxconf7'] = dxconf1_dict
replacedict['dxconf8'] = dxconf1_dict
replacedict['dxconf9'] = dxconf1_dict
replacedict['dxconf10'] = dxconf1_dict
####################################################################################
replacedict['e10pn1'] = e10pn1_dict
replacedict['e10pn2'] = e10pn1_dict
replacedict['e10pn3'] = e10pn1_dict
replacedict['e10pn4'] = e10pn1_dict
replacedict['e10pn5'] = e10pn1_dict
replacedict['e10pn6'] = e10pn1_dict
replacedict['e10pn7'] = e10pn1_dict
replacedict['e10pn8'] = e10pn1_dict
replacedict['e10pn9'] = e10pn1_dict
replacedict['e10pn10'] = e10pn1_dict
#########################################################################################
replacedict['cstum1'] = cstum1_dict
replacedict['cstum2'] = cstum1_dict
replacedict['cstum3'] = cstum1_dict
replacedict['cstum4'] = cstum1_dict
replacedict['cstum5'] = cstum1_dict
replacedict['cstum6'] = cstum1_dict
replacedict['cstum7'] = cstum1_dict
replacedict['cstum8'] = cstum1_dict
replacedict['cstum9'] = cstum1_dict
replacedict['cstum10'] = cstum1_dict
##############################################################################################
replacedict['dajcct1'] = dajcct1_dict
replacedict['dajcct2'] = dajcct1_dict
replacedict['dajcct3'] = dajcct1_dict
replacedict['dajcct4'] = dajcct1_dict
replacedict['dajcct5'] = dajcct1_dict
replacedict['dajcct6'] = dajcct1_dict
replacedict['dajcct7'] = dajcct1_dict
replacedict['dajcct8'] = dajcct1_dict
replacedict['dajcct9'] = dajcct1_dict
replacedict['dajcct10'] = dajcct1_dict
######################################################################################
replacedict['dajccn1'] = dajccn1_dict
replacedict['dajccn2'] = dajccn1_dict
replacedict['dajccn3'] = dajccn1_dict
replacedict['dajccn4'] = dajccn1_dict
replacedict['dajccn5'] = dajccn1_dict
replacedict['dajccn6'] = dajccn1_dict
replacedict['dajccn7'] = dajccn1_dict
replacedict['dajccn8'] = dajccn1_dict
replacedict['dajccn9'] = dajccn1_dict
replacedict['dajccn10'] = dajccn1_dict
######################################################################################
replacedict['dajccm1'] = dajccm1_dict
replacedict['dajccm2'] = dajccm1_dict
replacedict['dajccm3'] = dajccm1_dict
replacedict['dajccm4'] = dajccm1_dict
replacedict['dajccm5'] = dajccm1_dict
replacedict['dajccm6'] = dajccm1_dict
replacedict['dajccm7'] = dajccm1_dict
replacedict['dajccm8'] = dajccm1_dict
replacedict['dajccm9'] = dajccm1_dict
replacedict['dajccm10'] = dajccm1_dict
############################################################################
replacedict['dajccstg1'] = dajccstg1_dict
replacedict['dajccstg2'] = dajccstg1_dict
replacedict['dajccstg3'] = dajccstg1_dict
replacedict['dajccstg4'] = dajccstg1_dict
replacedict['dajccstg5'] = dajccstg1_dict
replacedict['dajccstg6'] = dajccstg1_dict
replacedict['dajccstg7'] = dajccstg1_dict
replacedict['dajccstg8'] = dajccstg1_dict
replacedict['dajccstg9'] = dajccstg1_dict
replacedict['dajccstg10'] = dajccstg1_dict
######################################################################
replacedict['dss77s1'] = dss77s1_dict
replacedict['dss77s2'] = dss77s1_dict
replacedict['dss77s3'] = dss77s1_dict
replacedict['dss77s4'] = dss77s1_dict
replacedict['dss77s5'] = dss77s1_dict
replacedict['dss77s6'] = dss77s1_dict
replacedict['dss77s7'] = dss77s1_dict
replacedict['dss77s8'] = dss77s1_dict
replacedict['dss77s9'] = dss77s1_dict
replacedict['dss77s10'] = dss77s1_dict
###########################################################################
replacedict['dss00s1'] = dss00s1_dict
replacedict['dss00s2'] = dss00s1_dict
replacedict['dss00s3'] = dss00s1_dict
replacedict['dss00s4'] = dss00s1_dict
replacedict['dss00s5'] = dss00s1_dict
replacedict['dss00s6'] = dss00s1_dict
replacedict['dss00s7'] = dss00s1_dict
replacedict['dss00s8'] = dss00s1_dict
replacedict['dss00s9'] = dss00s1_dict
replacedict['dss00s10'] = dss00s1_dict
####################################################################################
replacedict['sxprif1'] = sxprif1_dict
replacedict['sxprif2'] = sxprif1_dict
replacedict['sxprif3'] = sxprif1_dict
replacedict['sxprif4'] = sxprif1_dict
replacedict['sxprif5'] = sxprif1_dict
replacedict['sxprif6'] = sxprif1_dict
replacedict['sxprif7'] = sxprif1_dict
replacedict['sxprif8'] = sxprif1_dict
replacedict['sxprif9'] = sxprif1_dict
replacedict['sxprif10'] = sxprif1_dict
###################################################################################
replacedict['sxscof1'] = sxscof1_dict
replacedict['sxscof2'] = sxscof1_dict
replacedict['sxscof3'] = sxscof1_dict
replacedict['sxscof4'] = sxscof1_dict
replacedict['sxscof5'] = sxscof1_dict
replacedict['sxscof6'] = sxscof1_dict
replacedict['sxscof7'] = sxscof1_dict
replacedict['sxscof8'] = sxscof1_dict
replacedict['sxscof9'] = sxscof1_dict
replacedict['sxscof10'] = sxscof1_dict
###################################################################################
replacedict['sxsitf1'] = sxsitf1_dict
replacedict['sxsitf2'] = sxsitf1_dict
replacedict['sxsitf3'] = sxsitf1_dict
replacedict['sxsitf4'] = sxsitf1_dict
replacedict['sxsitf5'] = sxsitf1_dict
replacedict['sxsitf6'] = sxsitf1_dict
replacedict['sxsitf7'] = sxsitf1_dict
replacedict['sxsitf8'] = sxsitf1_dict
replacedict['sxsitf9'] = sxsitf1_dict
replacedict['sxsitf10'] = sxsitf1_dict
##############################################################################
replacedict['nosrg1'] = nosrg1_dict
replacedict['nosrg2'] = nosrg1_dict
replacedict['nosrg3'] = nosrg1_dict
replacedict['nosrg4'] = nosrg1_dict
replacedict['nosrg5'] = nosrg1_dict
replacedict['nosrg6'] = nosrg1_dict
replacedict['nosrg7'] = nosrg1_dict
replacedict['nosrg8'] = nosrg1_dict
replacedict['nosrg9'] = nosrg1_dict
replacedict['nosrg10'] = nosrg1_dict
#############################################################################
replacedict['rad1'] = rad1_dict
replacedict['rad2'] = rad1_dict
replacedict['rad3'] = rad1_dict
replacedict['rad4'] = rad1_dict
replacedict['rad5'] = rad1_dict
replacedict['rad6'] = rad1_dict
replacedict['rad7'] = rad1_dict
replacedict['rad8'] = rad1_dict
replacedict['rad9'] = rad1_dict
replacedict['rad10'] = rad1_dict
###########################################################################################
replacedict['radsurg1'] = radsurg1_dict
replacedict['radsurg2'] = radsurg1_dict
replacedict['radsurg3'] = radsurg1_dict
replacedict['radsurg4'] = radsurg1_dict
replacedict['radsurg5'] = radsurg1_dict
replacedict['radsurg6'] = radsurg1_dict
replacedict['radsurg7'] = radsurg1_dict
replacedict['radsurg8'] = radsurg1_dict
replacedict['radsurg9'] = radsurg1_dict
replacedict['radsurg10'] = radsurg1_dict
####################################################################################
replacedict['ager1'] = ager1_dict
replacedict['ager2'] = ager1_dict
replacedict['ager3'] = ager1_dict
replacedict['ager4'] = ager1_dict
replacedict['ager5'] = ager1_dict
replacedict['ager6'] = ager1_dict
replacedict['ager7'] = ager1_dict
replacedict['ager8'] = ager1_dict
replacedict['ager9'] = ager1_dict
replacedict['ager10'] = ager1_dict
##########################################################################################
replacedict['siterwho1'] = siterwho1_dict
replacedict['siterwho2'] = siterwho1_dict
replacedict['siterwho3'] = siterwho1_dict
replacedict['siterwho4'] = siterwho1_dict
replacedict['siterwho5'] = siterwho1_dict
replacedict['siterwho6'] = siterwho1_dict
replacedict['siterwho7'] = siterwho1_dict
replacedict['siterwho8'] = siterwho1_dict
replacedict['siterwho9'] = siterwho1_dict
replacedict['siterwho10'] = siterwho1_dict
###############################################################################
replacedict['iccc3who1'] = iccc3who1_dict
replacedict['iccc3who2'] = iccc3who1_dict
replacedict['iccc3who3'] = iccc3who1_dict
replacedict['iccc3who4'] = iccc3who1_dict
replacedict['iccc3who5'] = iccc3who1_dict
replacedict['iccc3who6'] = iccc3who1_dict
replacedict['iccc3who7'] = iccc3who1_dict
replacedict['iccc3who8'] = iccc3who1_dict
replacedict['iccc3who9'] = iccc3who1_dict
replacedict['iccc3who10'] = iccc3who1_dict
#################################################################
replacedict['histrec1'] = histrec1_dict
replacedict['histrec2'] = histrec1_dict
replacedict['histrec3'] = histrec1_dict
replacedict['histrec4'] = histrec1_dict
replacedict['histrec5'] = histrec1_dict
replacedict['histrec6'] = histrec1_dict
replacedict['histrec7'] = histrec1_dict
replacedict['histrec8'] = histrec1_dict
replacedict['histrec9'] = histrec1_dict
replacedict['histrec10'] = histrec1_dict
#####################################################################################
replacedict['hisrcb1'] = hisrcb1_dict
replacedict['hisrcb2'] = hisrcb1_dict
replacedict['hisrcb3'] = hisrcb1_dict
replacedict['hisrcb4'] = hisrcb1_dict
replacedict['hisrcb5'] = hisrcb1_dict
replacedict['hisrcb6'] = hisrcb1_dict
replacedict['hisrcb7'] = hisrcb1_dict
replacedict['hisrcb8'] = hisrcb1_dict
replacedict['hisrcb9'] = hisrcb1_dict
replacedict['hisrcb10'] = hisrcb1_dict
####################################################################################
replacedict['cssch1'] = cssch1_dict
replacedict['cssch2'] = cssch1_dict
replacedict['cssch3'] = cssch1_dict
replacedict['cssch4'] = cssch1_dict
replacedict['cssch5'] = cssch1_dict
replacedict['cssch6'] = cssch1_dict
replacedict['cssch7'] = cssch1_dict
replacedict['cssch8'] = cssch1_dict
replacedict['cssch9'] = cssch1_dict
replacedict['cssch10'] = cssch1_dict
#########################################################################################
replacedict['insrecpb1'] = insrecpb1_dict
replacedict['insrecpb2'] = insrecpb1_dict
replacedict['insrecpb3'] = insrecpb1_dict
replacedict['insrecpb4'] = insrecpb1_dict
replacedict['insrecpb5'] = insrecpb1_dict
replacedict['insrecpb6'] = insrecpb1_dict
replacedict['insrecpb7'] = insrecpb1_dict
replacedict['insrecpb8'] = insrecpb1_dict
replacedict['insrecpb9'] = insrecpb1_dict
replacedict['insrecpb10'] = insrecpb1_dict
##############################################################################
replacedict['payer_dx1'] = payer_dx1_dict
replacedict['payer_dx2'] = payer_dx1_dict
replacedict['payer_dx3'] = payer_dx1_dict
replacedict['payer_dx4'] = payer_dx1_dict
replacedict['payer_dx5'] = payer_dx1_dict
replacedict['payer_dx6'] = payer_dx1_dict
replacedict['payer_dx7'] = payer_dx1_dict
replacedict['payer_dx8'] = payer_dx1_dict
replacedict['payer_dx9'] = payer_dx1_dict
replacedict['payer_dx10'] = payer_dx1_dict


# In[15]:

def make_dirty_dataframe(filename,iterator=False):
    df = pd.read_fwf(filename, colspecs=colspecs, names=names,
                    iterator=iterator)
    return df


# In[16]:

def make_clean_dataframe(filename,iterator=False,nrows=None):
    df = pd.read_fwf(filename,nrows=nrows,colspecs=colspecs,names=names,
                    iterator=iterator)
    df.replace(replacedict,inplace=True)
    df.state = df.state.apply(str)
    df.state = df.state.apply(lambda x: x.rjust(2, '0'))
    df.county = df.county.apply(str)
    df.county = df.county.apply(lambda x: x.rjust(3, '0'))
    df['ST_CNTY'] = df.state + df.county
    df['ST_CNTY'] = df['ST_CNTY'].apply(lambda x: x.rjust(5, '0'))
    df['FIPScombo'] = df['ST_CNTY'].copy()
    df['FIPScombo'] = df['FIPScombo'].astype(str)
    df['ST_CNTY_ORIG'] = df['ST_CNTY'].copy()
    df['ST_CNTY_ORIG'] = df['ST_CNTY_ORIG'].apply(lambda x: x[:2])
    df['STATE'] = df['ST_CNTY_ORIG'].replace(ST_CNTY_ORIGdict)
    df['elevation'] = df['FIPScombo'].replace(FIPScomboelevationdict)
    df['lat'] = df['FIPScombo'].replace(FIPScombolatdict)
    df['lng'] = df['FIPScombo'].replace(FIPScombolngdict)
    df['countystate'] = df['FIPScombo'].replace(FIPScombocountystatedict)
    
    df.patient_id = df.patient_id.astype(str)
    df.patient_id = df.patient_id.apply(lambda x: x.rjust(10, '0'))
    
    df.statecd1 = df.statecd1.fillna(99)
    df.statecd2 = df.statecd2.fillna(99)
    df.statecd3 = df.statecd3.fillna(99)
    df.statecd4 = df.statecd4.fillna(99)
    df.statecd5 = df.statecd5.fillna(99)
    df.statecd6 = df.statecd6.fillna(99)
    df.statecd7 = df.statecd7.fillna(99)
    df.statecd8 = df.statecd8.fillna(99)
    df.statecd9 = df.statecd9.fillna(99)
    df.statecd10 = df.statecd10.fillna(99)

    df.statecd1 = df.statecd1.astype(int)
    df.statecd2 = df.statecd2.astype(int)
    df.statecd3 = df.statecd3.astype(int)
    df.statecd4 = df.statecd4.astype(int)
    df.statecd5 = df.statecd5.astype(int)
    df.statecd6 = df.statecd6.astype(int)
    df.statecd7 = df.statecd7.astype(int)
    df.statecd8 = df.statecd8.astype(int)
    df.statecd9 = df.statecd9.astype(int)
    df.statecd10 = df.statecd10.astype(int)

    df.statecd1 = df.statecd1.apply(str)
    df.statecd2 = df.statecd2.apply(str)
    df.statecd3 = df.statecd3.apply(str)
    df.statecd4 = df.statecd4.apply(str)
    df.statecd5 = df.statecd5.apply(str)
    df.statecd6 = df.statecd6.apply(str)
    df.statecd7 = df.statecd7.apply(str)
    df.statecd8 = df.statecd8.apply(str)
    df.statecd9 = df.statecd9.apply(str)
    df.statecd10 = df.statecd10.apply(str)
    df.statecd1 = df.statecd1.apply(lambda x: x.rjust(2, '0'))
    df.statecd2 = df.statecd2.apply(lambda x: x.rjust(2, '0'))
    df.statecd3 = df.statecd3.apply(lambda x: x.rjust(2, '0'))
    df.statecd4 = df.statecd4.apply(lambda x: x.rjust(2, '0'))
    df.statecd5 = df.statecd5.apply(lambda x: x.rjust(2, '0'))
    df.statecd6 = df.statecd6.apply(lambda x: x.rjust(2, '0'))
    df.statecd7 = df.statecd7.apply(lambda x: x.rjust(2, '0'))
    df.statecd8 = df.statecd8.apply(lambda x: x.rjust(2, '0'))
    df.statecd9 = df.statecd9.apply(lambda x: x.rjust(2, '0'))
    df.statecd10 = df.statecd10.apply(lambda x: x.rjust(2, '0'))


    df.cnty1 = df.cnty1.fillna(999)
    df.cnty2 = df.cnty2.fillna(999)
    df.cnty3 = df.cnty3.fillna(999)
    df.cnty4 = df.cnty4.fillna(999)
    df.cnty5 = df.cnty5.fillna(999)
    df.cnty6 = df.cnty6.fillna(999)
    df.cnty7 = df.cnty7.fillna(999)
    df.cnty8 = df.cnty8.fillna(999)
    df.cnty9 = df.cnty9.fillna(999)
    df.cnty10 = df.cnty10.fillna(999)



    df.cnty1 = df.cnty1.astype(int)
    df.cnty2 = df.cnty2.astype(int)
    df.cnty3 = df.cnty3.astype(int)
    df.cnty4 = df.cnty4.astype(int)
    df.cnty5 = df.cnty5.astype(int)
    df.cnty6 = df.cnty6.astype(int)
    df.cnty7 = df.cnty7.astype(int)
    df.cnty8 = df.cnty8.astype(int)
    df.cnty9 = df.cnty9.astype(int)
    df.cnty10 = df.cnty10.astype(int)

    df.cnty1 = df.cnty1.apply(str)
    df.cnty2 = df.cnty2.apply(str)
    df.cnty3 = df.cnty3.apply(str)
    df.cnty4 = df.cnty4.apply(str)
    df.cnty5 = df.cnty5.apply(str)
    df.cnty6 = df.cnty6.apply(str)
    df.cnty7 = df.cnty7.apply(str)
    df.cnty8 = df.cnty8.apply(str)
    df.cnty9 = df.cnty9.apply(str)
    df.cnty10 = df.cnty10.apply(str)

    df.cnty1 = df.cnty1.apply(lambda x: x.rjust(3, '0'))
    df.cnty2 = df.cnty2.apply(lambda x: x.rjust(3, '0'))
    df.cnty2 = df.cnty2.apply(lambda x: x.rjust(3, '0'))
    df.cnty3 = df.cnty3.apply(lambda x: x.rjust(3, '0'))
    df.cnty4 = df.cnty4.apply(lambda x: x.rjust(3, '0'))
    df.cnty5 = df.cnty5.apply(lambda x: x.rjust(3, '0'))
    df.cnty6 = df.cnty6.apply(lambda x: x.rjust(3, '0'))
    df.cnty7 = df.cnty7.apply(lambda x: x.rjust(3, '0'))
    df.cnty8 = df.cnty8.apply(lambda x: x.rjust(3, '0'))
    df.cnty9 = df.cnty9.apply(lambda x: x.rjust(3, '0'))
    df.cnty10 = df.cnty10.apply(lambda x: x.rjust(3, '0'))


    df['ST_CNTY_1'] = df.statecd1 + df.cnty1
    df['ST_CNTY_2'] = df.statecd2 + df.cnty2
    df['ST_CNTY_3'] = df.statecd3 + df.cnty3
    df['ST_CNTY_4'] = df.statecd4 + df.cnty4
    df['ST_CNTY_5'] = df.statecd5 + df.cnty5
    df['ST_CNTY_6'] = df.statecd6 + df.cnty6
    df['ST_CNTY_7'] = df.statecd7 + df.cnty7
    df['ST_CNTY_8'] = df.statecd8 + df.cnty8
    df['ST_CNTY_9'] = df.statecd9 + df.cnty9
    df['ST_CNTY_10'] = df.statecd10 + df.cnty10

    df['ST_CNTY_1'] = df['ST_CNTY_1'].apply(lambda x: x.rjust(5, '0'))
    df['ST_CNTY_2'] = df['ST_CNTY_2'].apply(lambda x: x.rjust(5, '0'))
    df['ST_CNTY_3'] = df['ST_CNTY_3'].apply(lambda x: x.rjust(5, '0'))
    df['ST_CNTY_4'] = df['ST_CNTY_4'].apply(lambda x: x.rjust(5, '0'))
    df['ST_CNTY_5'] = df['ST_CNTY_5'].apply(lambda x: x.rjust(5, '0'))
    df['ST_CNTY_6'] = df['ST_CNTY_6'].apply(lambda x: x.rjust(5, '0'))
    df['ST_CNTY_7'] = df['ST_CNTY_7'].apply(lambda x: x.rjust(5, '0'))
    df['ST_CNTY_8'] = df['ST_CNTY_8'].apply(lambda x: x.rjust(5, '0'))
    df['ST_CNTY_9'] = df['ST_CNTY_9'].apply(lambda x: x.rjust(5, '0'))
    df['ST_CNTY_10'] = df['ST_CNTY_10'].apply(lambda x: x.rjust(5, '0'))

    df['FIPScombo_1'] = df['ST_CNTY_1'].copy()
    df['FIPScombo_2'] = df['ST_CNTY_2'].copy()
    df['FIPScombo_3'] = df['ST_CNTY_3'].copy()
    df['FIPScombo_4'] = df['ST_CNTY_4'].copy()
    df['FIPScombo_5'] = df['ST_CNTY_5'].copy()
    df['FIPScombo_6'] = df['ST_CNTY_6'].copy()
    df['FIPScombo_7'] = df['ST_CNTY_7'].copy()
    df['FIPScombo_8'] = df['ST_CNTY_8'].copy()
    df['FIPScombo_9'] = df['ST_CNTY_9'].copy()
    df['FIPScombo_10'] = df['ST_CNTY_10'].copy()


    df['FIPScombo_1'] = df['FIPScombo_1'].astype(str)
    df['FIPScombo_2'] = df['FIPScombo_2'].astype(str)
    df['FIPScombo_3'] = df['FIPScombo_3'].astype(str)
    df['FIPScombo_4'] = df['FIPScombo_4'].astype(str)
    df['FIPScombo_5'] = df['FIPScombo_5'].astype(str)
    df['FIPScombo_6'] = df['FIPScombo_6'].astype(str)
    df['FIPScombo_7'] = df['FIPScombo_7'].astype(str)
    df['FIPScombo_8'] = df['FIPScombo_8'].astype(str)
    df['FIPScombo_9'] = df['FIPScombo_9'].astype(str)
    df['FIPScombo_10'] = df['FIPScombo_10'].astype(str)


    df['elevation_1'] = df['FIPScombo_1'].replace(FIPScomboelevationdict)
    df['elevation_2'] = df['FIPScombo_2'].replace(FIPScomboelevationdict)
    df['elevation_3'] = df['FIPScombo_3'].replace(FIPScomboelevationdict)
    df['elevation_4'] = df['FIPScombo_4'].replace(FIPScomboelevationdict)
    df['elevation_5'] = df['FIPScombo_5'].replace(FIPScomboelevationdict)
    df['elevation_6'] = df['FIPScombo_6'].replace(FIPScomboelevationdict)
    df['elevation_7'] = df['FIPScombo_7'].replace(FIPScomboelevationdict)
    df['elevation_8'] = df['FIPScombo_8'].replace(FIPScomboelevationdict)
    df['elevation_9'] = df['FIPScombo_9'].replace(FIPScomboelevationdict)
    df['elevation_10'] = df['FIPScombo_10'].replace(FIPScomboelevationdict)


    df['lat_1'] = df['FIPScombo_1'].replace(FIPScombolatdict)
    df['lat_2'] = df['FIPScombo_2'].replace(FIPScombolatdict)
    df['lat_3'] = df['FIPScombo_3'].replace(FIPScombolatdict)
    df['lat_4'] = df['FIPScombo_4'].replace(FIPScombolatdict)
    df['lat_5'] = df['FIPScombo_5'].replace(FIPScombolatdict)
    df['lat_6'] = df['FIPScombo_6'].replace(FIPScombolatdict)
    df['lat_7'] = df['FIPScombo_7'].replace(FIPScombolatdict)
    df['lat_8'] = df['FIPScombo_8'].replace(FIPScombolatdict)
    df['lat_9'] = df['FIPScombo_9'].replace(FIPScombolatdict)
    df['lat_10'] = df['FIPScombo_10'].replace(FIPScombolatdict)

    df['lng_1'] = df['FIPScombo_1'].replace(FIPScombolngdict)
    df['lng_2'] = df['FIPScombo_2'].replace(FIPScombolngdict)
    df['lng_3'] = df['FIPScombo_3'].replace(FIPScombolngdict)
    df['lng_4'] = df['FIPScombo_4'].replace(FIPScombolngdict)
    df['lng_5'] = df['FIPScombo_5'].replace(FIPScombolngdict)
    df['lng_6'] = df['FIPScombo_6'].replace(FIPScombolngdict)
    df['lng_7'] = df['FIPScombo_7'].replace(FIPScombolngdict)
    df['lng_8'] = df['FIPScombo_8'].replace(FIPScombolngdict)
    df['lng_9'] = df['FIPScombo_9'].replace(FIPScombolngdict)
    df['lng_10'] = df['FIPScombo_10'].replace(FIPScombolngdict)

    df['countystate_1'] = df['FIPScombo_1'].replace(FIPScombocountystatedict)
    df['countystate_2'] = df['FIPScombo_2'].replace(FIPScombocountystatedict)
    df['countystate_3'] = df['FIPScombo_3'].replace(FIPScombocountystatedict)
    df['countystate_4'] = df['FIPScombo_4'].replace(FIPScombocountystatedict)
    df['countystate_5'] = df['FIPScombo_5'].replace(FIPScombocountystatedict)
    df['countystate_6'] = df['FIPScombo_6'].replace(FIPScombocountystatedict)
    df['countystate_7'] = df['FIPScombo_7'].replace(FIPScombocountystatedict)
    df['countystate_8'] = df['FIPScombo_8'].replace(FIPScombocountystatedict)
    df['countystate_9'] = df['FIPScombo_9'].replace(FIPScombocountystatedict)
    df['countystate_10'] = df['FIPScombo_10'].replace(FIPScombocountystatedict)
    
    toprocess = ['icdot09_1','icdot09_2','icdot09_3','icdot09_4',
            'icdot09_5','icdot09_6','icdot09_7','icdot09_8',
            'icdot09_9','icdot09_10']

    for c in toprocess:
        icd9dx_description_trans(df, dficd9dx, c)
        
    return df


# In[ ]:



