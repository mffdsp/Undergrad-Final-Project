import pandas as pd
import numpy as np

# # X-From
# # X-cc
# # X-bcc
# # X-Folder
# # X-Origin
# # X-FileName
# # Content
# # file
# # allen-p

data = pd.read_csv('./emails.csv')
data = data["message"]


# text_file = open("raw.txt", "w+")
# text_file.write(np.fromstring(data.values) )
# text_file.close()
np.savetxt(r'raw.txt', data.values, fmt='%s')


# #export DataFrame to text file
# with open(path, 'a') as f:
#     df_string = data.to_string(header=False, index=False)
#     f.write(df_string)