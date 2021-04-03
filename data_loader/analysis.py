#%% 
import pandas as pd
# %%
submission_df = pd.read_csv('/opt/ml/input/data/eval/submission 오후 4.43.13.csv')
# %%
values = list(submission_df.ans.values)
values = [value % 6 for value in values]
# %%
from collections import Counter
Counter(values)

#%%
submission_df = pd.read_csv('/opt/ml/input/data/eval/submission.csv')
# %%
values = list(submission_df.ans.values)
values = [value % 6 for value in values]
# %%
from collections import Counter
Counter(values)

# %%
submission_df = pd.read_csv('/opt/ml/input/data/eval/submission27.csv')
# %%
values = list(submission_df.ans.values)
values = [value % 6 for value in values]
# %%
from collections import Counter
Counter(values)

# %%
submission_df = pd.read_csv('/opt/ml/input/data/eval/submission29.csv')
# %%
values = list(submission_df.ans.values)
values = [value % 6 for value in values]
# %%
from collections import Counter
Counter(values)
# %%
