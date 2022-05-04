import pandas as pd
from regex import P

df = None
for i in range(10):
    tmp = pd.read_csv('%d_imp2.csv'%i)
    tmp.columns = ['%d'%i, 'feature']
    if df is None:
        df = tmp
    else:
        df = df.merge(tmp, on='feature')
df = df[['feature'] + ['%d'%i for i in range(10)]]
df['importance'] = df[['%d'%i for i in range(10)]].sum(axis=1)
df = df[['feature', 'importance']]
df = df.sort_values(by='importance')
df.to_csv('imp1.csv', index=False)
print(df['feature'].values[20:].tolist())
