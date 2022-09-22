import pandas as pd

# Data from archived files from 22.9.2018 version of MVČR website https://www.mvcr.cz/clanek/cetnost-jmen-a-prijmeni.aspx
# https://web.archive.org/web/20180922062314/https://www.mvcr.cz/clanek/cetnost-jmen-a-prijmeni.aspx

# Data look like (for name, in surname "JMÉNO" is replaced by "PŘÍJMENÍ"):
#                  JMÉNO  0  1899  1900  1901  1902  1903  1904  1905  1906  1907  ...  2008  2009  2010  2011  2012  2013  2014  2015  2016  2017  3000
# 0                 A-MI  0     0     0     0     0     0     0     0     0     0  ...     0     0     0     0     0     0     0     0     0     0     1
# 1                A-RIA  0     0     0     0     0     0     0     0     0     0  ...     0     0     0     0     0     0     0     0     0     0     1
# 2          AADAM SAMER  0     0     0     0     0     0     0     0     0     0  ...     1     0     0     0     0     0     0     0     0     0     1
# 3                AADIV  0     0     0     0     0     0     0     0     0     0  ...     0     0     0     0     0     0     0     0     1     0     1
# 4                AAGOT  0     0     0     0     0     0     0     0     0     0  ...     0     0     0     0     0     0     0     0     0     0     1
# ...                ... ..   ...   ...   ...   ...   ...   ...   ...   ...   ...  ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...
# 64994         VASILIJE  0     0     0     0     0     0     0     0     0     0  ...     0     0     0     0     0     0     0     0     0     0     3
# 64995         VASILIKA  0     0     0     0     0     0     0     0     0     0  ...     0     0     0     0     0     0     0     0     0     0     6
# 64996         VASILIKI  0     0     0     0     0     0     0     0     0     0  ...     0     0     0     0     0     0     0     0     0     0    39
# 64997  VASILIKI-HELENA  0     0     0     0     0     0     0     0     0     0  ...     0     0     0     0     0     0     0     0     0     0     1
# 64998  VASILIKI-ZUZANA  0     0     0     0     0     0     0     0     0     0  ...     0     0     0     0     0     0     0     0     0     0     1

# Load data from Excel file into Pandas DataFrame
df_dict = pd.read_excel('lexicons/četnost-jména-dnar.xls', sheet_name=None)
df_names = pd.concat(df_dict.values())

# Filter names with 200 or more occurances
df_names.sort_values(by=[3000], inplace=True, ascending=False, ignore_index=True)
common_names = df_names[df_names[3000] > 200]["JMÉNO"].str.lower()
common_names.drop(index=0, inplace=True)

# Save common names to JSON file
common_names.to_json("lexicons/common_names.json", orient="records", force_ascii=False, indent=4)


# Load data from Excel file into Pandas DataFrame
df_dict = pd.read_excel('lexicons/četnost-příjmení-dnar.xls', sheet_name=None)
df_surnames = pd.concat(df_dict.values())

# Filter surnames with 1000 or more occurances
df_surnames.sort_values(by=[3000], inplace=True, ascending=False, ignore_index=True)
common_surnames = df_surnames[df_surnames[3000] > 1000]["PŘÍJMENÍ"].str.lower()
common_surnames.drop(index=0, inplace=True)

# Save common surnames to JSON file
common_surnames.to_json("lexicons/common_surnames.json", orient="records", force_ascii=False, indent=4)