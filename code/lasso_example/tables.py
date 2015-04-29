
import pandas as pd, numpy as np

from data_carving import summary

# Gaussian first

summary(np.inf)
gaussian = pd.read_csv('gaussian/summary.csv')

gaussian_df = pd.DataFrame({'Algorithm': [r'$\text{Carve}_{100}$',
                                          r'$\text{Split}_{50}$',
                                          r'$\text{Carve}_{50}$',            
                                          r'$\text{Split}_{75}$',
                                          r'$\text{Carve}_{75}$'],
                            'power_names':['power_carve', 
                                           'power_split', 
                                           'power_carve', 
                                           'power_split', 
                                           'power_carve'],
                            'level_names':['level_carve', 
                                           'level_split', 
                                           'level_carve', 
                                           'level_split', 
                                           'level_carve'],
                            'split':[1., 0.5, 0.5, 0.75, 0.75]})

level = []
power = []
FP = []
TP = []
FDP = []
screen = []

for lw, pw, frac in zip(gaussian_df['level_names'],
                        gaussian_df['power_names'], 
                        gaussian_df['split']):
    level.append(np.mean(gaussian[lw][gaussian['split'] == frac]))
    power.append(np.mean(gaussian[pw][gaussian['split'] == frac]))
    FP.append(np.mean(gaussian['fp'][gaussian['split'] == frac]))
    TP.append(np.mean(gaussian['tp'][gaussian['split'] == frac]))
    FDP.append(np.mean(gaussian['fdp'][gaussian['split'] == frac]))
    screen.append(np.mean(gaussian['p_screen'][gaussian['split'] == frac]))

gaussian_df['Power'] = power
gaussian_df['Level'] = level
gaussian_df[r'$\mathbb{E}[V]$'] = FP
gaussian_df[r'$\mathbb{E}[R-V]$'] = TP
gaussian_df['FDR'] = FDP
gaussian_df[r'$p_{\text{screen}}$'] = screen

del(gaussian_df['power_names'])
del(gaussian_df['level_names'])
del(gaussian_df['split'])

gaussian_df = gaussian_df.reindex_axis([
        'Algorithm',
        r'$p_{\text{screen}}$',
        r'$\mathbb{E}[V]$',
        r'$\mathbb{E}[R-V]$',
        'FDR',
        'Power',
        'Level'], axis=1)

file('tables/gaussian.tex', 'w').write(gaussian_df.to_latex(index=False, float_format=lambda x: "%0.02f" % x).replace("\\_", "_"))


# now to T_5

summary(5)
T5 = pd.read_csv('df_5/summary.csv')
T5_df = pd.DataFrame({'Algorithm': [r'$\text{Carve}_{100}$',
                                          r'$\text{Split}_{50}$',
                                          r'$\text{Carve}_{50}$'],
                            'power_names':['power_carve', 
                                           'power_split', 
                                           'power_carve'],
                            'level_names':['level_carve', 
                                           'level_split', 
                                           'level_carve'],
                            'split':[1., 0.5, 0.5]})

level = []
power = []
FP = []
TP = []
FDP = []
screen = []

for lw, pw, frac in zip(T5_df['level_names'],
                        T5_df['power_names'], 
                        T5_df['split']):
    level.append(np.mean(T5[lw][T5['split'] == frac]))
    power.append(np.mean(T5[pw][T5['split'] == frac]))
    FP.append(np.mean(T5['fp'][T5['split'] == frac]))
    TP.append(np.mean(T5['tp'][T5['split'] == frac]))
    FDP.append(np.mean(T5['fdp'][T5['split'] == frac]))
    screen.append(np.mean(T5['p_screen'][T5['split'] == frac]))

T5_df['Power'] = power
T5_df['Level'] = level
T5_df[r'$\mathbb{E}[V]$'] = FP
T5_df[r'$\mathbb{E}[R-V]$'] = TP
T5_df['FDR'] = FDP
T5_df[r'$p_{\text{screen}}$'] = screen

del(T5_df['power_names'])
del(T5_df['level_names'])
del(T5_df['split'])

T5_df = T5_df.reindex_axis([
        'Algorithm',
        r'$p_{\text{screen}}$',
        r'$\mathbb{E}[V]$',
        r'$\mathbb{E}[R-V]$',
        'FDR',
        'Power',
        'Level'], axis=1)

file('tables/T5.tex', 'w').write(T5_df.to_latex(index=False, float_format=lambda x: "%0.02f" % x).replace("\\_", "_"))
