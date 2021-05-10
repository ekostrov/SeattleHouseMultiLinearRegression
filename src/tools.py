# Imports
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_rainbow
from scipy import stats
import pylab
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import normaltest
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.compat import lzip
# End of Imports


def clean_sqft_basement(sqft):
    if sqft =='?':
        return mean
    return float(sqft)

def print_column_info(column):
    print(f"Unique values:\n {column.unique()}")
    print(f"Value_counts:\n {column.value_counts()}")
    print(f"Number of Null values:\n {column.isnull().sum()}")

def drop_na(data, list_columns):
    for column in list_columns:
        idx = data[data[column].isna()].index
        data.drop(idx, inplace = True)
    
    return data

def formula(df, target):
    columns = df.drop(target, axis=1).columns
    right_side = " + ".join(columns)
    return target + "~" + right_side

def evaluate_model(data, target_column='', multicollinearity=False):
    """
    Overall Evaluation of the Model
    data = Pandas Data Frame
    target_column = Dependent Variable
    """
    f = formula(data, target_column)
    model = sm.OLS.from_formula(f,data=data)
    result=model.fit()
    M=60
    print("-"*M)
    print("MODEL SUMMARY:")
    print("-"*M)
    print(result.summary())
    print("-"*M)
    print("MODEL EVALUATION:")
    print("-"*M)
    print("\t","-"*M)
    print("\t CHECK LINEARITY ASSUMPTIONS:")
    print("\t","-"*M)
    rainbow_test(result)
    print("\t","-"*M)
    print("\t CHECK NORMALITY ASSUMPTIONS:")
    print("\t","-"*M)
    print("\tQQ-PLOT\n")
    stats.probplot(result.resid, dist='norm',plot=pylab)
    pylab.show()
    print("\tDISTRIBUTIONS PLOT OF RESIDUALS\n")
    ax = sns.displot(result.resid,kde=True)
    plt.show()
    #shapiro_wilk(result)
    D_Agostino(result)
    
    print("\t","-"*M)
    print("\t CHECK IF WE HAVE HETEROSCEDASTICTY IN THE MODEL :")
    print("\t","-"*M)
    print("%"*M)
    plot_resid(result)
    print("%"*M)
    h_b_test(result)
    print("\t","-"*M)
    if multicollinearity:
        print("\t","-"*M)
        print("\t CHECK MULTICOLLINEARITY ASSUMPTIONS:")
        print("\t","-"*M)
        new_df = data.drop(target_column, axis=1)
        if 'id' in new_df.keys():
            new_df = new_df.drop('id', axis=1)
        corr = new_df.corr()
        sns.heatmap(corr)
        plt.show()
        print("\n","*"*M,"\nCORRELATON MATRIX\n", corr)
        print("\n","*"*M,"\n")
        vif_data = pd.DataFrame()
        vif_data["feature"] = data.columns
        # calculating VIF for each feature
        vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(len(data.columns))]
        y, X = dmatrices(f, data=data, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)
        

        
    
def plot_resid(result):
    fig3, ax3 = plt.subplots(figsize=(15,10))
    ax3.set(xlabel="Predicted Values",
    ylabel="Residuals (Actual - Predicted)")
    ax3.scatter(x=result.predict(),y=result.resid, color='blue', alpha=0.2);
    plt.show()
    
def rainbow_test(result):
    """
    Accepts:
        result = model.fit() 
    Performs Rainbow Test and prints results
    """
    rainbow_statistic, rainbow_p_value = linear_rainbow(result)
    print("Statistic =", rainbow_statistic, "P_Value =", rainbow_p_value)
    print("The null hypothesis H0 is that the model is linearly predicted by the features,\n alternative hypothesis Ha is that it is not.")
    print(f'stat={rainbow_statistic:.3f}, p={rainbow_p_value:.3f}')
    if  rainbow_p_value > 0.05:
        print(f"We have {rainbow_p_value:.3f} > 0.05. We don't have evidence to reject the H0,\n thus the current model satisfies the linearity assumption.")
    else:
        print(f"We have enough evidence to reject H0, since  {rainbow_p_value:.3f} < 0.05 and coclude that the model doesn't satisfy liearity assumption.")

def h_b_test(result):
    """
    het_breuschpagan test
    """
    names = ['Lagrange multiplier statistic', 'p-value','f-value', 'f p-value']
    test = het_breuschpagan(result.resid, result.model.exog)
    #print(lzip(names, test))
    print("A Breusch-Pagan test uses the following null and alternative hypotheses:\n The null hypothesis (H0): Homoscedasticity is present.\n The alternative hypothesis: (Ha): Homoscedasticity is not present (i.e. heteroscedasticity exists)")
    print(f"We see that the Lagrange multiplier statistic for the test is {test[0]}  and the corresponding p-value is {test[1]:0.5f}.")
    if test[1] < 0.05:
          print("Because this p-value is less than alpha = 0.05, we have enough evidence to reject the null hypothesis.\nWe have sufficient evidence to say that heteroscedasticity is present in the regression model.")
    else:
          print("Because this p-value is bigger than alpha = 0.05, we don't enough evidence to reject the null hypothesis.\n We believe that homoscedasticity is present in the regression model.")
# Tests for Normality
def shapiro_wilk(result):
    """
    Accepts:
        result = model.fit() 
    Performs Shapiro-Wilk Test and prints results
    """
    print("Shapiro-Wilk for Normality")
    stat, p = shapiro(result.resid)
    print(f'stat={stat:.3f}, p={p:.3f}')
    
    if p > 0.05:
        print('Probability is Normal')
    else:
        print('Probability is not Normal')
def D_Agostino(result):
    """
    Accepts:
        result = model.fit() 
    Performs D'Agostino Test and prints results
    """
    print("D_Agostino Test for Normality:")
    stat, p = normaltest(result.resid)
    print(f'stat={stat:.3f}, p={p:.3f}')
    
    if p > 0.05:
        print('Probability is Normal')
    else:
        print('Probability is not Normal')
        
def convert_number_to_month(number):
    look_up = {'1': 'Jan', '2': 'Feb',
               '3': 'Mar', '4': 'Apr',
               '5': 'May', '6': 'Jun',
               '7': 'Jul', '8': 'Aug',
               '9': 'Sep', '10': 'Oct',
               '11': 'Nov', '12': 'Dec'}
    return look_up[str(number)]

def drop_outliers(data, target=''):
    # DROP THE OUTLIERS
    df = data.copy()
    cut= df[target].mean() + 3* df[target].std()
    idx = df[df.sqft_living > cut].index
    df =df.drop(idx)
    cut= df[target].mean() - 3* df[target].std()
    idx = df[df.sqft_living < cut].index
    df =df.drop(idx)
    return df