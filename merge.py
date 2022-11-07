import pandas as pd
from matplotlib.pyplot import plot, show
from seaborn import heatmap

if __name__ == '__main__':

    ans1 = pd.read_csv('ans/lgb_5285.csv')
    ans2 = pd.read_csv('ans/lgb_5272.csv')
    ans3 = pd.read_csv('ans/submit28.csv')
    ans4 = pd.read_csv('ans/sub_5225.csv')

    ans = pd.DataFrame()
    ans['A_0.5285'] = ans1['ret'].rank()
    ans['A_0.5272'] = ans2['ret'].rank()
    ans['A_0.5265'] = ans3['ret'].rank()
    ans['A_0.5224'] = ans4['ret'].rank()

    heatmap(ans.corr())
    show()
    print(ans.corr())

    ans = ans1.copy()

    # 根据模型与模型之间的相关性设定融合权重（相关性越小融合结果越好，因此所占权重也应更大）
    ans['ret'] = ans1['ret'].rank() * 0.3 + ans2['ret'].rank() * 0.3 + ans4['ret'].rank() * 0.3 + ans3[
        'ret'].rank() * 0.4
    ans['ret'] = (ans['ret'] - ans['ret'].min()) / (ans['ret'].max() - ans['ret'].min())

    ans[['id', 'ret']].to_csv('ans/ans.csv', index=False)
