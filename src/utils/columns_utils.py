import pandas as pd

def get_columns_type(df_columns_type):
    column_names_list = list()
    for col in df_columns_type:
        column_names_list.append(col)
        print(f"\t\t {col}")
    return column_names_list

def get_info_cleaing_process(df_train):
    columns_missing_values = df_train.isnull().sum()
    print("Columns missing values")
    print(columns_missing_values[columns_missing_values > 0])

def distribution_by_category(var_cat, df):
    print("\nDistribution by category per feature(variable)\n")
    for i in var_cat:
        print(f"Category:{i}")
        c = df[i].value_counts(dropna=False)
        p = round(df[i].value_counts(dropna=False, normalize=True),1)
        print(pd.concat([c,p], axis=1, keys=['counts', '%']))