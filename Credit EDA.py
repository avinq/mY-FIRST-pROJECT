# importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
import warnings
warnings.filterwarnings('ignore')

# loading application data file-1
df= pd.read_csv('application_data.csv')

# checking the shape of the data
print(df.shape)

# checking the info of datafile
df.info()

# checking the statistical details
df.describe()

## Quality checking and checking missing values

pd.set_option('display.max_columns', 125)
pd.set_option('display.max_rows',200)

round(100*df.isnull().sum()/ len(df),2)

# removing the columns with high missing value % greater than 50%
df=df.loc[:, 100*df.isnull().sum()/len(df)<50]

# checking the shape of the dataframe
df.shape

#Select columns with less or equal to the 13% null vallues
list(df.columns[(df.isnull().mean()<=0.13) & (df.isnull().mean()>0)])

#check these columns for imputation

### TAKING 1 at a time
#### 1.AMT_ANNUITY

 print(df.AMT_ANNUITY.describe())
sns.boxplot(df['AMT_ANNUITY'])
plt.show()


#  it has lot of outliers so considering median measure
df.AMT_ANNUITY.median()
# we can impute 24903(median) as missing values

#### 2.AMT_GOODS_PRICE


print(df.AMT_GOODS_PRICE.describe())
sns.boxplot(df['AMT_GOODS_PRICE'])
plt.show()

# it has lot of outliers so considering median 
df.AMT_GOODS_PRICE.median()
# we can impute 450000.0 as missing values

#### 3.NAME_TYPE_SUITE

print(df.NAME_TYPE_SUITE.describe())
# since it is categorical value, considering mode to impute missing values
print(df.NAME_TYPE_SUITE.mode())
print('Clearly the column NAME_TYPE_SUITE is a categorical column. So this column can be imputed using the mode of the column i.e Unaccompanied') 

#### 4.CNT_FAM_MEMBERS

print(df.CNT_FAM_MEMBERS.describe())
sns.boxplot(df['CNT_FAM_MEMBERS'])
plt.show()

# it has lot of outliers so considering median 
df.CNT_FAM_MEMBERS.median()
# we can impute "2.0" as missing values

#### 5.EXT_SOURCE_2

print(df.EXT_SOURCE_2.describe())
sns.boxplot(df['EXT_SOURCE_2'])
plt.show()

#  it seems, mean and median are almost same but the box is shifted up so go with median
df.EXT_SOURCE_2.median()
# so, we can impute 0.5659614260608526 as missing values

# checking the d-types of all the columns and changing the d-type like negative age and date
print(df.info())

# converting negative DAYS_BIRTH value to positive value
df['DAYS_BIRTH']=df['DAYS_BIRTH'].abs()
# converting negative DAYS_EMPLOYED value to positive value
df['DAYS_EMPLOYED']=df['DAYS_EMPLOYED'].abs()
# converting negative DAYS_REGISTRATION value to positive value
df['DAYS_REGISTRATION']=df['DAYS_REGISTRATION'].abs()
# converting negative DAYS_ID_PUBLISH value to positive value
df['DAYS_ID_PUBLISH']=df['DAYS_ID_PUBLISH'].abs()
# converting negative DAYS_LAST_PHONE_CHANGE value to positive value
df['DAYS_LAST_PHONE_CHANGE']=df['DAYS_LAST_PHONE_CHANGE'].abs()
df.head()

# checking the count of unique values fro every column
print(df.nunique().sort_values())

# converting of columns integer to categorical
for col in df.columns:
    if df[col].nunique() <= 3: #considering columns with 3 unique values as categorical variables
        df[col] = df[col].astype(object)

df.info() 

print('No columns found')

#Making Gender more readable
df['CODE_GENDER'].value_counts()

# Dropping the Gender = XNA from the data  
df = df[df['CODE_GENDER']!='XNA']
df['CODE_GENDER'].replace(['M','F'],['Male','Female'],inplace=True)

### Checking for imbalance in Target

df['TARGET'].value_counts(normalize=True)*100


plt.pie(df['TARGET'].value_counts(normalize=True)*100,labels=['NON-DEFAULTER (TARGET=0)','DEFAULTER (TARGET=1)'],explode=(0,0.08),autopct='%1.f%%')
plt.title('Target Variable - DEFAULTER Vs NONDEFAULTER')
plt.show()

# More than 92% of people where non-defaulters as compared to 8% who failed

#### Numerical columns checks fro outliers and report them for atleast 5 variables

plt.boxplot(df['CNT_CHILDREN'])
plt.show()
# From box plot, we can conclude that there exists values which are above upper whisker(maximum) considered to be as outliers. 
Q1 = df['CNT_CHILDREN'].quantile(0.25)
Q3 = df['CNT_CHILDREN'].quantile(0.75)
IQR = Q3 - Q1
lowerwhisker=(Q1 - 1.5 * IQR)
upperwhisker=(Q3 + 1.5 * IQR)
# According to Statictics the values above the upper whisker and below the lower whisker are considered as outliers
#and as we can see in plot outliers are present only above the upper wisker so considering them as outliers
print("The values greater than {} are considered to be outliers,since count of children cannot be in decimals we can conclude that count greater than 3 can be an outlier".format(upperwhisker))

sns.boxplot(df['AMT_CREDIT'])
plt.title('AMT_CREDIT')
plt.show()
# Calculating whiskers- upper and lower
Q1 = df['AMT_CREDIT'].quantile(0.25)
Q3 = df['AMT_CREDIT'].quantile(0.75)
IQR = Q3 - Q1
lowerwhisker=(Q1 - 1.5 * IQR)
upperwhisker=(Q3 + 1.5 * IQR)
#we can see in plot outliers are present only above the upper wisker so considering them as outliers
print("The amount credited greater than {} can be considered as an outlier".format(upperwhisker))

df['AMT_CREDIT'].describe()
df['AMT_CREDIT'].max()

data=df['AMT_ANNUITY']
filtered = data[~np.isnan(data)]
sns.boxplot(filtered)
plt.show()
# Calculating upper and lower whiskers
Q1 = df['AMT_ANNUITY'].quantile(0.25)
Q3 = df['AMT_ANNUITY'].quantile(0.75)
IQR = Q3 - Q1
lowerwhisker=(Q1 - 1.5 * IQR)
upperwhisker=(Q3 + 1.5 * IQR)
#we can see in plot outliers are present only above the upper wisker so considering them as outliers
print("Population count greater than {} is considered to be an outlier".format(upperwhisker))

sns.boxplot(df['REGION_POPULATION_RELATIVE'])
plt.show()
# Calculating whiskers
Q1 = df['REGION_POPULATION_RELATIVE'].quantile(0.25)
Q3 = df['REGION_POPULATION_RELATIVE'].quantile(0.75)
IQR = Q3 - Q1
lowerwhisker=(Q1 - 1.5 * IQR)
upperwhisker=(Q3 + 1.5 * IQR)
#we can see in plot outliers are present only above the upper wisker so considering them as outliers
print("Population count greater than {} is considered to be an outlier".format(upperwhisker))

data=df['AMT_GOODS_PRICE']
filtered = data[~np.isnan(data)]
plt.boxplot(filtered)
plt.show()
# Calculating Whiskers
Q1 = df['AMT_GOODS_PRICE'].quantile(0.25)
Q3 = df['AMT_GOODS_PRICE'].quantile(0.75)
IQR = Q3 - Q1
lowerwhisker=(Q1 - 1.5 * IQR)
upperwhisker=(Q3 + 1.5 * IQR)
#we can see in plot outliers are present only above the upper wisker so considering them as outliers
print("Population count greater than {} is considered to be an outlier".format(upperwhisker))

#### Binning variables for Analysis

# Binning of continuous variables.Check if you need to bin any variable in different categories
q1=df['AMT_INCOME_TOTAL'].quantile(0.25)
q2=df['AMT_INCOME_TOTAL'].quantile(0.50)
q3=df['AMT_INCOME_TOTAL'].quantile(0.75)
m=df['AMT_INCOME_TOTAL'].max()

# Binning AMT_INCOME_TOTAL into AMT_INCOME_TOTAL_bin so we don't loose data and have binned values
df['AMT_INCOME_TOTAL_bin'] = pd.cut(df['AMT_INCOME_TOTAL'],[q1, q2, q3,m ], labels = ['Low', 'medium', 'High'])
print(df.AMT_INCOME_TOTAL_bin.value_counts())

q1=df['AMT_CREDIT'].quantile(0.25)
q2=df['AMT_CREDIT'].quantile(0.50)
q3=df['AMT_CREDIT'].quantile(0.75)
m=df['AMT_CREDIT'].max()

# Binning AMT_CREDIT into AMT_CREDIT_bin so we don't loose data and have binned values
df['AMT_CREDIT_bin'] = pd.cut(df['AMT_CREDIT'],[q1, q2, q3,m ], labels = ['Low', 'medium', 'High'])
print(df.AMT_CREDIT_bin.value_counts())

## Analysis

df.head()

#Checking the imbalance percentage.
print(100*df.TARGET.value_counts()/ len(df))
(df.TARGET.value_counts()/ len(df)).plot.bar()
plt.xticks(rotation=0)
plt.show()
# In application_data there exists 91.927118% of "not default" and 8.072882% of "default" customers.

# Divide the data into two sets, i.e., Target-1 and Target-0
df_1 = df[df['TARGET']==1]
df_0 = df[df['TARGET']==0]

#### Univariate Analysis

# for target=0
df_0.WEEKDAY_APPR_PROCESS_START.value_counts(normalize=True).plot.bar()
plt.title('for non-default')
plt.show()
# for target=1
df_1.WEEKDAY_APPR_PROCESS_START.value_counts(normalize=True).plot.bar()
plt.title('for default')
plt.show()
# Conclusion-application starting processes are less on saturday and sunday

# for target=0
df_0.NAME_EDUCATION_TYPE.value_counts(normalize=True).plot.pie()
plt.tight_layout()
plt.title('for non-default')
plt.show()
# Conclusion-secondary/special educated people are applying loans in high in number.
# for Target=1
df_1.NAME_EDUCATION_TYPE.value_counts(normalize=True).plot.pie()
plt.tight_layout()
plt.title('for default')
plt.show()
# Conclusion- secondary/special educated people are applying loans high in number.

# for TARGET=0
df_0.NAME_FAMILY_STATUS.value_counts(normalize=True).plot.bar()
plt.title('for non-default')
plt.show()
# for TARGET=1
df_1.NAME_FAMILY_STATUS.value_counts(normalize=True).plot.bar()
plt.title('for default')
plt.show()

# for TARGET=0
df_0.NAME_INCOME_TYPE.value_counts(normalize=True).plot.bar()
plt.title('for non-default')
plt.show()
# for TARGET=1
df_1.NAME_INCOME_TYPE.value_counts(normalize=True).plot.bar()
plt.title('for default')
plt.show()

# for TARGET=0
df_0.NAME_HOUSING_TYPE.value_counts(normalize=True).plot.bar()
plt.title('for non-default')
plt.show()
# for TARGET=1
df_1.NAME_HOUSING_TYPE.value_counts(normalize=True).plot.bar()
plt.title('for default')
plt.show()

#### Comparing the target variables in the categorical variables against Target 0 and 1

# 10 categorical columns
ctg_columns=['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY',
                     'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE',
                    'WEEKDAY_APPR_PROCESS_START','AMT_CREDIT_bin','AMT_INCOME_TOTAL_bin']

plt.figure(figsize=(22,25))
for i in (enumerate(ctg_columns)):
    plt.subplot(len(ctg_columns)//2,2,i[0]+1)
    sns.countplot(x=i[1],hue='TARGET',data=df)
    plt.yscale('log')
plt.show()

#### Comparing the target variables in the number variables against Target 0 and 1

#10 continous numerical columns
cts_columns=['AMT_ANNUITY','AMT_GOODS_PRICE','CNT_FAM_MEMBERS',
                  'DAYS_LAST_PHONE_CHANGE','DAYS_ID_PUBLISH','DAYS_BIRTH','HOUR_APPR_PROCESS_START',
                  'DAYS_EMPLOYED','AMT_CREDIT','AMT_INCOME_TOTAL']
plt.figure(figsize=(22,25))
for i in (enumerate(cts_columns)):
    plt.subplot(len(cts_columns)//2,2,i[0]+1)
    sns.distplot(df_1[i[1]].dropna(),hist=False,label='Target : default')
    sns.distplot(df_0[i[1]].dropna(),hist=False,label='Target : no default')
plt.show() 

### Correlation
#### Correlation for numerical columns for both cases of target column

#df_1.corr()
df_1.corr().unstack().reset_index().sort_values(by=0,ascending=False)

#Calculating Top 10 Correlated values for defalut 
corr=df_1.corr()
corrdf=corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
corrdf=corrdf.unstack().reset_index()
corrdf.columns=['Var1','Var2','Coorelation']
corrdf.dropna(subset=['Coorelation'],inplace=True)
corrdf['Coorelation']=round(corrdf['Coorelation'],2)
corrdf['Coorelation']=abs(corrdf['Coorelation']) 
corrdf.sort_values(by='Coorelation',ascending=False).head()

#Calculating Top 10 Correlated values for non-defalut
corr=df_0.corr()
corrdf=corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
corrdf=corrdf.unstack().reset_index()
corrdf.columns=['Var1','Var2','Coorelation']
corrdf.dropna(subset=['Coorelation'],inplace=True)
corrdf['Coorelation']=round(corrdf['Coorelation'],2)
corrdf['Coorelation']=abs(corrdf['Coorelation']) 
corrdf.sort_values(by='Coorelation',ascending=False).head()

## Bivariate Analysis
#### Bivariate Categorical plots

t1= pd.crosstab(index=df['TARGET'],columns=df['NAME_CONTRACT_TYPE'])
print(t1)
t1.plot(kind="bar", figsize=(6,6),stacked=False)
plt.xticks(rotation=0)
plt.show()
# High cash loans

t2= pd.crosstab(index=df['TARGET'],columns=df['CODE_GENDER'])
print(t2)
t2.plot(kind="bar", figsize=(6,6),stacked=False)
plt.xticks(rotation=0)
plt.show()
#Females are taking more loans

t3= pd.crosstab(index=df['TARGET'],columns=df['NAME_TYPE_SUITE'])
print(t3)
t3.plot(kind="bar", figsize=(6,6),stacked=False)
plt.xticks(rotation=0)
plt.show()
# People mostly come alone while taking loans

t4= pd.crosstab(index=df['TARGET'],columns=df['NAME_INCOME_TYPE'])
print(t4)
t4.plot(kind="bar", figsize=(6,6),stacked=False)
plt.show()
# Employed people take more loans

t5= pd.crosstab(index=df['TARGET'],columns=df['NAME_HOUSING_TYPE'])
print(t5)
t5.plot(kind="bar", figsize=(6,6),stacked=False)
plt.show()
# People having house tend to take more loans

#### Bivariate Continous plots

cts_columns=['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE',
                  'DAYS_EMPLOYED','DAYS_BIRTH','DAYS_LAST_PHONE_CHANGE','HOUR_APPR_PROCESS_START',
                  'DAYS_ID_PUBLISH','DAYS_REGISTRATION']
plt.figure(figsize=(18,25))
for i in (enumerate(cts_columns)):
    plt.subplot(len(cts_columns)//2,2,i[0]+1)
    sns.boxplot(x='TARGET',y=df[i[1]].dropna(),data=df)
    plt.yscale('log')
plt.show() 

### Previous Application Data

p_df=pd.read_csv('previous_application.csv')
p_df.head()

#### Checking the structure of the data

print(p_df.shape)
p_df.info()
p_df.describe()

# checking missing value %  
round((100*p_df.isnull().sum()/len(p_df)),2)

# AMT_DOWN_PAYMENT,RATE_DOWN_PAYMENT,RATE_INTEREST_PRIMARY,RATE_INTEREST_PRIVILEGED 
p_df=p_df.drop(['AMT_DOWN_PAYMENT','RATE_DOWN_PAYMENT','RATE_INTEREST_PRIMARY','RATE_INTEREST_PRIVILEGED'], axis = 1)
p_df.info()

# converting -ve values to +ve
p_df['DAYS_DECISION']=p_df['DAYS_DECISION'].abs()
p_df['SELLERPLACE_AREA']=p_df['SELLERPLACE_AREA'].abs()
p_df['DAYS_FIRST_DUE']=p_df['DAYS_FIRST_DUE'].abs()
p_df['DAYS_LAST_DUE_1ST_VERSION']=p_df['DAYS_LAST_DUE_1ST_VERSION'].abs()
p_df['DAYS_LAST_DUE']=p_df['DAYS_LAST_DUE'].abs()
p_df['DAYS_TERMINATION']=p_df['DAYS_TERMINATION'].abs()
p_df['DAYS_FIRST_DRAWING']=p_df['DAYS_FIRST_DRAWING'].abs()

#### Univariate Analysis

# function to count plot for categorical variables
def plot(var):
    plt.style.use('ggplot')
    sns.despine
    fig,ax = plt.subplots(1,1,figsize=(10,6))
    sns.countplot(x=var, data=p_df,ax=ax,hue='NAME_CONTRACT_STATUS')
    ax.set_ylabel('Total Counts')
    ax.set_title(f'Distribution of {var}',fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.show()

plot('NAME_CONTRACT_TYPE')

plot('NAME_PAYMENT_TYPE')

plot('NAME_CLIENT_TYPE')

####  Checking the correlation in the PreviousApplication dataset

corr=p_df.corr()
corr_df = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool)).unstack().reset_index()
corr_df.columns=['Column1','Column2','Correlation']
corr_df.dropna(subset=['Correlation'],inplace=True)
corr_df['Abs_Correlation']=corr_df['Correlation'].abs()
corr_df = corr_df.sort_values(by=['Abs_Correlation'], ascending=False)
corr_df.head(10)

#Checking by plotting the relation between numeric vriables
plt.figure(figsize=[18,10])
sns.pairplot(p_df[['AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_GOODS_PRICE','NAME_CONTRACT_STATUS']], 
             diag_kind = 'kde', 
             plot_kws = {'alpha': 0.4, 's': 80, 'edgecolor': 'k'},
             size = 4)
plt.show()

(p_df.NAME_CONTRACT_STATUS.value_counts()/len(p_df)).plot.bar()
plt.show()

#### Merging Application dta and previous application data

# Left join as data from application_data.csv is required
merge_df=pd.merge(df,p_df,how='left',on='SK_ID_CURR',suffixes=('_Current', '_Previous'))
merge_df.head()

#### Univariate Categorical Analysis

ctg_columns=['NAME_CONTRACT_TYPE_Current','NAME_CONTRACT_TYPE_Previous',
                     'NAME_TYPE_SUITE_Current','NAME_TYPE_SUITE_Previous',
                     'WEEKDAY_APPR_PROCESS_START_Current','WEEKDAY_APPR_PROCESS_START_Previous',
                    'AMT_INCOME_TOTAL_bin','AMT_CREDIT_bin','NAME_YIELD_GROUP','NAME_CLIENT_TYPE']
plt.figure(figsize=(20,25))
for i in (enumerate(ctg_columns)):
    plt.subplot(len(ctg_columns)//2,2,i[0]+1)
    sns.countplot(x=i[1],hue='NAME_CONTRACT_STATUS',data=merge_df)
plt.show()

#### Univariate Numerical Analysis

# Univariate Numerical analysis
cts_columns=['AMT_CREDIT_Previous','AMT_CREDIT_Current','AMT_ANNUITY_Current','AMT_ANNUITY_Previous',
                   'AMT_GOODS_PRICE_Current','AMT_GOODS_PRICE_Previous','CNT_FAM_MEMBERS','CNT_CHILDREN',
                  'HOUR_APPR_PROCESS_START_Previous','HOUR_APPR_PROCESS_START_Current']
plt.figure(figsize=(20,25))
for i in (enumerate(cts_columns)):
    plt.subplot(len(cts_columns)//2,2,i[0]+1)
    sns.distplot(merge_df.loc[merge_df.NAME_CONTRACT_STATUS=='Approved',:][i[1]].dropna(),hist=False,label='Approved')
    sns.distplot(merge_df.loc[merge_df.NAME_CONTRACT_STATUS=='Canceled',:][i[1]].dropna(),hist=False,label='Canceled',kde_kws={'bw':0.1})
    sns.distplot(merge_df.loc[merge_df.NAME_CONTRACT_STATUS=='Refused',:][i[1]].dropna(),hist=False,label='Refused',kde_kws={'bw':0.1})
    sns.distplot(merge_df.loc[merge_df.NAME_CONTRACT_STATUS=='Unused offer',:][i[1]].dropna(),hist=False,label='Unused offer')
plt.show() 

#### Bivariate Categorical Analysis

t6= pd.crosstab(index=merge_df['NAME_CONTRACT_STATUS'],columns=merge_df['NAME_CONTRACT_TYPE_Current'])
print(t6)
t6.plot(kind="bar", figsize=(6,6),stacked=False)
plt.show()
#Cash loans are the hghest approved loans

t7= pd.crosstab(index=merge_df['NAME_CONTRACT_STATUS'],columns=merge_df['NAME_INCOME_TYPE'])
print(t7)
t7.plot(kind="bar", figsize=(6,6),stacked=False)
plt.show()

t8= pd.crosstab(index=merge_df['NAME_CONTRACT_STATUS'],columns=merge_df['NAME_EDUCATION_TYPE'])
print(t8)
t8.plot(kind="bar", figsize=(6,6),stacked=False)
plt.show()

t9= pd.crosstab(index=merge_df['NAME_CONTRACT_STATUS'],columns=merge_df['NAME_FAMILY_STATUS'])
print(t9)
t9.plot(kind="bar", figsize=(6,6),stacked=False)
plt.show()

t10= pd.crosstab(index=merge_df['NAME_CONTRACT_STATUS'],columns=merge_df['NAME_HOUSING_TYPE'])
print(t10)
t10.plot(kind="bar", figsize=(6,6),stacked=False)
plt.show()

t11= pd.crosstab(index=merge_df['NAME_CONTRACT_STATUS'],columns=merge_df['NAME_CONTRACT_TYPE_Previous'])
print(t11)
t11.plot(kind="bar", figsize=(6,6),stacked=False)
plt.show()

t12= pd.crosstab(index=merge_df['NAME_CONTRACT_STATUS'],columns=merge_df['NAME_CLIENT_TYPE'])
print(t12)
t12.plot(kind="bar", figsize=(6,6),stacked=False)
plt.show()

#### Continous/Numerical analysis

#Bi-variate continous plots
cts_columns=['AMT_ANNUITY_Current','AMT_ANNUITY_Previous',
                   'AMT_GOODS_PRICE_Current','AMT_GOODS_PRICE_Previous','CNT_FAM_MEMBERS','CNT_CHILDREN',
                  'HOUR_APPR_PROCESS_START_Previous','HOUR_APPR_PROCESS_START_Current',
                   'AMT_CREDIT_Current','AMT_CREDIT_Previous']
                   #'AMT_INCOME_TOTAL']
plt.figure(figsize=(20,25))
for i in (enumerate(cts_columns)):
    plt.subplot(len(cts_columns)//2,2,i[0]+1)
    sns.boxplot(x='NAME_CONTRACT_STATUS',y=merge_df[i[1]].dropna(),data=merge_df)
plt.show()