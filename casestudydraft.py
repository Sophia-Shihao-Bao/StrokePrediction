#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 20:53:15 2023

@author: sophiabao
"""
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from scipy.stats import chi2_contingency
path = '/Users/sophiabao/文件/onlineclasses/DataQuest/CaseStudy2/'
stroke_raw = pd.read_csv(path+'healthcare-dataset-stroke-data.csv')
stroke = pd.read_csv(path+'healthcare-dataset-stroke-data.csv')

stroke['hypertension'] = stroke["hypertension"].replace({1: "Yes", 0: "No"})
stroke['heart_disease'] = stroke["heart_disease"].replace({1: "Yes", 0: "No"})
stroke['stroke'] = stroke['stroke'].replace({1:"Yes", 0:"No"})
stroke = stroke[stroke['gender']!='Other'].copy()

pretty_num_names = {'age':'Age', 'avg_glucose_level':'Average Glucose Level', 'bmi':'Body Mass Index'}
pretty_cat_names = {'gender':'Gender', 'hypertension':'Hypertension','heart_disease':'Heart Disease','ever_married':'Ever Married','work_type':'Work Type',"Residence_type":'Residence Type','smoking_status':'Smoking Status'}
all_pretty_names = {'age':'Age', 'avg_glucose_level':'Average Glucose Level', 'bmi':'Body Mass Index', 'gender':'Gender', 'hypertension':'Hypertension','heart_disease':'Heart Disease','ever_married':'Ever Married','work_type':'Work Type',"Residence_type":'Residence Type','smoking_status':'Smoking Status','stroke':'Stroke'}
category_orders_dict = {'Residence_type':['Urban','Rual'],'work_type':['Private','Self-employed','children','Govt-job','Never_worked'],'smoking_status':['never smoked','Unknown','formerly smoked','smokes'],'gender':['Female','Male']}
num_name = ['age','avg_glucose_level', 'bmi']
cat_name = ['gender','heart_disease','hypertension','ever_married','work_type','Residence_type','smoking_status']


st.set_page_config(layout="wide")
with st.sidebar: 
	selected = option_menu(
		menu_title = 'Navigation Pane',
		options = ['Abstract', 'Background Information', 'Data Cleaning', 
		'Exploratory Analysis', 'Naive Bays Function Explanation','Data Analysis', 'Conclusion', 'Bibliography'],
		menu_icon = 'compass-fill',
		icons = ["bookmark","briefcase","clipboard-data","graph-up",'123',"bar-chart-line", "pen", "blockquote-left"],
		default_index = 0)

if selected == 'Abstract':
	st.title('Abstract')

	st.markdown('This case study uses a dataset provided by FEDESORIANO on Kaggle and analyzes the factors that influence stroks. After making assumptions and conducting simulations to prove them, conclusions were made that strokes are affected by multipule factors. Select from the sidebar to see the full case study.')
	st.markdown('')
	st.markdown('')
	st.markdown('')
	st.markdown('')
	st.caption('Made by Shihao Bao')
    
if selected=="Background Information":
    st.title("Background Information")
    st.markdown("A stroke is a severe medical condition resulting from the disruption of blood flow to the brain, which can lead to the death of brain cells. The Stroke Prediction Dataset, developed by FEDESORIANO, provides data scientists and medical professionals with essential information to identify individuals at high risk of stroke and implement preventive measures. By analyzing various factors such as age, gender, hypertension, and smoking status, predictive models can be developed to aid in early detection and intervention. With the use of data-driven insights, personalized approaches to stroke prevention can be developed to improve patient outcomes. This case study aims to harness the power of data science to make a significant contribution to better healthcare outcomes for those at risk of stroke.")
    st.markdown("I found a dataset on Kaggle about strokes. It is very interesting and included many data about stroke and non-stroke patients. I used the infromation to create a prediction of stroke. Please see the unaltered dataset below:")
    st.dataframe(stroke_raw)
    st.markdown("On the following pages, I will clean the dataset, allow you to make some graphs and make some intresting graphs myself. Then I would draw an conclution on stroke prediction based on the graphs I made and the patterns I found.")
    
    
    
if selected=="Data Cleaning":
    st.title("Data Cleaning")
    st.markdown("I edited the dataset and added new columns into it to make it more easier to understand at get more data from the dataset. Here are my steps.")
    st.subheader("Raw Dataset")
    st.dataframe(stroke)
    





if selected=="Exploratory Analysis":
    st.title("Exploratory Analysis")
    st.markdown("This is where you can experiment with the dataset by making your own plotly charts. You can change the variables that makes the chart using dropdown menus, once changed,the system would automaticly change the chart for you. Specific instructions for diffrent charts are below their subheaders. Have fun trying out scatter plots, histograms and sunbursts! Try and see what intresing patterns you may come up with.")
    
    
    st.subheader("Scatter Plots")
    st.markdown("Scatter plots is a graph in which the values of two variables are plotted along two axes, the pattern of the resulting points revealing any correlation present. You can chose your variables from the respective dropown menu. Have Fun!")
    col1,col2 = st.columns([3,6])
    
    with col1.form("Scatter_plot_form"):
        x1 = st.selectbox(
         'Select the X value of the scatterplot', 
         [pname for name, pname in pretty_num_names.items()])	
    
        y1 = st.selectbox(
         'Select the Y value of the scatterplot',
         [pname for name, pname in pretty_num_names.items()])
        
        logscale1 = st.checkbox("Do you want a log scale?", False, key = 4)
        x1_name = [name for name, pname in all_pretty_names.items() if pname == x1]
        y1_name = [name for name, pname in all_pretty_names.items() if pname == y1]
        submitted = st.form_submit_button("Click to display Scatter plot")
        if submitted:
            scatter_ex = px.scatter(stroke, x=x1_name[0], y=y1_name[0], color='stroke', labels=all_pretty_names, category_orders = category_orders_dict, log_y = logscale1)
            scatter_ex = scatter_ex.update_traces(marker_line_width=1, marker_line_color= "black")
            scatter_ex = scatter_ex.update_layout(yaxis=dict(title_text=y1, title_font_size=16, title_font_color="black"), xaxis=dict(title_text=x1, title_font_size=16, title_font_color="black"))
            col2.plotly_chart(scatter_ex)
    
    
    
    
    st.subheader("Histogram")
    st.markdown("A histogram is a diagram consisting of rectangles whose area is proportional to the frequency of a variable and whose width is equal to the class interval. You can choose your variable in the dropdown menu below. In addition, plotly histograms allows some special additions to be made, those are also in their respective dropdown menus.")
    col3,col4 = st.columns([3,6])
    with col3.form("Histogram_Form"):
        x2 = st.selectbox(
         'Select the X value of the graph',
         [pname for name, pname in all_pretty_names.items() if name in['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status']])
        barmode = st.selectbox("please select the barmode value of the graph, defult is relative", ("relative", "group", "overlay"))
        barnorm = st.selectbox("please select the barnorm value of the graph, defult is none", ("", "percent", "fraction"))
        marginal = st.selectbox("Select the marginal for the graph, defult is none. If set, a subplot is drawn alongside the main plot, visualizing the distribution.", ("", "rug", "box", "violin", "histogram"))
        x2_name = [name for name, pname in all_pretty_names.items() if pname == x2]
        submitted = st.form_submit_button("Click to display Histogram")
        if submitted:        
            histo_ex = px.histogram(stroke, x=x2_name[0], barmode = barmode,barnorm=barnorm,color='stroke',labels=all_pretty_names)
            histo_ex= histo_ex.update_traces(marker_line_width=1, marker_line_color='black')
            histo_ex = histo_ex.update_layout(xaxis=dict(title_text=x2, title_font_size=16, title_font_color='black'))
            col4.plotly_chart(histo_ex)
    
    
    
    
    st.subheader("Box Plot")
    st.markdown("A box plot is a graphical rendition of statistical data based on the minimum, first quartile, median, third quartile, and maximum. It is usefull to show the distrubution of data.")
    col5,col6 = st.columns([3,6])
    
   
    with col5.form("Box_Plot_Form"):
      y3 = st.selectbox("Select the Y value of the graph", [pname for name, pname in pretty_cat_names.items()], key = 2)
      color2 = st.selectbox("Select the Color variable", [pname for name, pname in pretty_num_names.items()], key=3)
      logscale = st.checkbox("Do you want a log scale?",False, key = 100)
      
      y3_name = [name for name, pname in all_pretty_names.items() if pname == y3]
      color2_name = [name for name, pname in all_pretty_names.items() if pname == color2]
       # Every form must have a submit button.
      submitted = st.form_submit_button("Click to display box plot")
      if submitted:
        box_ex = px.box(stroke, x='stroke', y=y3_name[0], color=color2_name[0], labels=all_pretty_names, log_y = logscale)
        box_ex = box_ex.update_traces(marker_line_width=1, marker_line_color='black')
        box_ex = box_ex.update_layout(yaxis=dict(title_text=y3, title_font_size=16, title_font_color='black'))
        col6.plotly_chart(box_ex)

    




    st.subheader("Sunbursts")
    st.markdown("The sunburst chart is ideal for displaying hierarchical data. Each level of the hierarchy is represented by one ring or circle with the innermost circle as the top of the hierarchy. You can choose your levels in the dropdown menus below. Please note that if the two variables are the same ot are similar, the graph would not work. The values that the porportions are based on is also one of the dropdown menus. You can click on the first level of any of the variables to see a more detailed graph.")
    col7,col8=st.columns([3,6])
    
    
    with col7.form("sunburst_form"):
        layer1 = st.selectbox("Select the first layer of the sunburst graph", ('gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status'))
        
        value = st.selectbox("Select the value the graph is gong to be based on", [pname for name, pname in pretty_num_names.items()])
        
        
        layer1_name = [name for name, pname in all_pretty_names.items() if pname == layer1]
        value_name = [name for name, pname in all_pretty_names.items() if pname == value]
        submitted = st.form_submit_button("Click to display Sunburst")
        if submitted:
            sun_ex = px.sunburst(stroke, path=[layer1, 'stroke'], values=value_name[0], labels=all_pretty_names)
            sun_ex = sun_ex.update_traces(marker_line_width=1, marker_line_color='black')
            col8.plotly_chart(sun_ex)
            
            
    st.subheader('Chi-Square Tests')
    st.markdown('Chi-Square Tests are specific to prediction datasets, as we would work out which accociation of variables work together the best.')
    
    def perform_chi_square(df, target, feature):
        cross_tab = pd.crosstab(df[feature], df[target])
        chi2, p_value, _, _ = chi2_contingency(cross_tab)
        return chi2, p_value

    col13,col14 = st.columns([3,6])
    with col13.form('chi-square'):
        target_column = st.selectbox('Select Target Column', [pname for name, pname in pretty_cat_names.items()])
        feature_column = st.selectbox('Select Feature Column', [pname for name, pname in pretty_cat_names.items()])
        
        target_column_name = [name for name, pname in all_pretty_names.items() if pname == target_column]
        feature_column_name = [name for name, pname in all_pretty_names.items() if pname == feature_column]
        
        logscale2 = st.checkbox("Do you want a log scale?",False, key = 101)
        chi2_value, p_value = perform_chi_square(stroke, target_column_name[0], feature_column_name[0])
        st.write(f'Chi-Square value: {chi2_value}')
        st.write(f'p-value: {p_value}')
        chisquare = px.histogram(stroke, x=feature_column_name[0], color=target_column_name[0], barmode='group')
        submitted = st.form_submit_button('Click to display Chi-Square')
        
        
        
        if submitted:
            col14.plotly_chart(chisquare, log_y= logscale2)
    
    
    
    
if selected=="Naive Bays Function Explanation": #Check streamlit st.expander      PUT interactive NB here
    st.title('Naive Bays Function Explanation')
    st.markdown("On this page, I would explain the NB funtion that we use to make predictions for the data analysis.")
    
    def nb(df, target):
        def fit_distribution(data):
         # estimate parameters
         mu = np.mean(data)
         sigma = np.std(data)
         # fit distribution
         dist = stats.norm(mu, sigma)
         return dist
        prior = df[target].value_counts(normalize = True)
        stroke1= df[df[target]=="Yes"].copy()
        stroke0= df[df[target]=="No"].copy()
        prob0_dict = {}
        prob1_dict = {}
        catCols = df.drop(target, axis=1).select_dtypes(include='object').columns
        for column in catCols:
            prob1 = stroke1[column].value_counts(normalize=True)
            prob0 = stroke0[column].value_counts(normalize=True)
            prob1_dict[column]=df[column].replace(dict(prob1))
            prob0_dict[column]=df[column].replace(dict(prob0))
            if column == 'work_type':
                prob1_dict[column] = prob1_dict[column].replace({'Never_worked':0.000000001})
        numeric = df.drop(target, axis=1).select_dtypes(include=['integer','float']).columns
        for column in numeric:
            prob1 = fit_distribution(stroke1[column])
            prob0 = fit_distribution(stroke0[column])
            prob1_dict[column] = prob1.pdf(df[column])
            prob0_dict[column] = prob0.pdf(df[column])
        prob0df = pd.DataFrame(prob0_dict).prod(axis=1)*prior["No"]
        prob1df = pd.DataFrame(prob1_dict).prod(axis=1)*prior["Yes"]
        Predicted = (prob0df < prob1df)*1
        final = pd.concat([df[target], Predicted], axis=1).rename({target:'Actual', 0:'Predicted'}, axis=1).replace({1:"Yes", 0:"No"})
        accuracy = (final['Actual'] == final['Predicted']).mean()
        print(pd.crosstab(final['Actual'], final['Predicted']))
        return final, accuracy
    
    st.subheader("Explaining the Naive Bays Algorithm")
    st.markdown("""Naive Bayes is a classification algorithm that is widely used for prediction tasks. It is based on the concept of conditional probability and assumes independence among the features (or variables) being considered. The algorithm calculates the probability of a particular class given the observed values of the features.

The Naive Bayes algorithm works by first estimating the probabilities of each class in the training data, as well as the conditional probabilities of each feature given each class. It assumes that the features are conditionally independent of each other given the class. This assumption simplifies the calculations and allows for efficient computation.

To classify a new instance, the algorithm applies Bayes' theorem, which calculates the probability of a class given the observed values of the features. It multiplies the prior probability of the class (which is estimated from the training data) with the conditional probabilities of each feature given that class. The algorithm then selects the class with the highest probability as the predicted class for the instance.

The construction of a Naive Bayes classifier typically involves three steps. First, the predictor and target variables are chosen, and the conditional probabilities of the features are estimated using probability density functions appropriate for the type of variable (e.g., binary or numeric). Then, the dataset is split into a training set and a testing set. The training set is used to find the model that provides the most accurate predictions, while the testing set is used to evaluate the performance of the model on unseen data.

Naive Bayes is known for its simplicity, speed, and scalability. It can handle large datasets efficiently and is particularly effective when the independence assumption holds reasonably well. However, the assumption of feature independence can limit its accuracy when the features are highly correlated. Despite this limitation, Naive Bayes remains a popular choice for many classification tasks due to its simplicity and competitive performance in various domains.""")

    st.subheader("Naive Bayes Exploratory Table")

    col11,col12 = st.columns([3,6])
    
    with col11.form('NB'):
        select = st.multiselect('Select the values for the Naive Bayes tests, this is a Multi-Select box', [pname for name, pname in all_pretty_names.items()])
        select_name = [name for name, pname in all_pretty_names.items() if pname in select]+['stroke']
        results, acc = nb(stroke[select_name], 'stroke')
        submitted = st.form_submit_button("Click to display Naive Bayes Test")
        if submitted:
            col12.write('Accuracy: '+str(np.round(acc, 4)))
            col12.dataframe(results)

    st.subheader('Code & Explanation of the Naive Bayes Exploratory Table')
    expander=st.expander('Click to expand')
    expander.code('''def nb(df, target): #The function nb is defined with two parameters: df (a pandas DataFrame) and target
                      #(a string representing the target variable).
    def fit_distribution(data):
     # estimate parameters
     mu = np.mean(data) #3-8. The inner function fit_distribution is defined. It takes a data array
                         #as input, calculates the mean (mu) and standard deviation (sigma) of the data,
                         #creates a normal distribution (dist) using the calculated parameters, and
                         #returns the distribution object.
     sigma = np.std(data)
     # fit distribution
     dist = stats.norm(mu, sigma)
     return dist
    prior = df[target].value_counts(normalize = True) #prior is a pandas Series that contains the value 
        #counts of the target variable in the DataFrame df, normalized to represent the
        #probabilities of each class.
    stroke1= df[df[target]=="Yes"].copy() # 12-13. Two new DataFrames, stroke1 and stroke0, 
    #are created by filtering df based on the values of the target variable. 
    #stroke1 contains rows where the target variable is "Yes," and stroke0 contains
    #rows where the target variable is "No." These DataFrames are copied to avoid
    #modifying the original DataFrame.
    stroke0= df[df[target]=="No"].copy()
    prob0_dict = {} #15-16. Two empty dictionaries, prob0_dict and prob1_dict, are initialized to
    #store the probabilities for each category in categorical columns.
    prob1_dict = {}
    catCols = df.drop(target, axis=1).select_dtypes(include='object').columns #The catCols variable
    #is assigned the column names of df excluding the target column, and only selecting
    #columns of type 'object' (categorical columns).
    for column in catCols: #19-23. A loop iterates over each column name in catCols.
    #For each column, the probabilities of each category are calculated and stored 
    #in prob1 and prob0 variables using value_counts(normalize=True) on the
    #respective stroke1 and stroke0 DataFrames. Then, the original values in the column
    #of df are replaced with their corresponding probabilities using replace(dict(prob1))
    #and replace(dict(prob0)), and these modified columns are assigned to prob1_dict[column]
    #and prob0_dict[column], respectively. Additionally, for the specific column
    #'work_type', the category 'Never_worked' is replaced with a very small value to avoid
    #division by zero issues. The numeric variable is assigned the column names of df
    #excluding the target column, and only selecting columns of types
    #'integer' and 'float' (numeric columns).
        prob1 = stroke1[column].value_counts(normalize=True)
        prob0 = stroke0[column].value_counts(normalize=True)
        prob1_dict[column]=df[column].replace(dict(prob1))
        prob0_dict[column]=df[column].replace(dict(prob0)) 
        if column == 'work_type':
            prob1_dict[column] = prob1_dict[column].replace({'Never_worked':0.000000001})
    numeric = df.drop(target, axis=1).select_dtypes(include=['integer','float']).columns
    #27-32. Another loop iterates over each column name in numeric. For each column,
    #the data in stroke1 and stroke0 corresponding to that column are used as input
    #to the fit_distribution function, which returns a distribution object.Then, 
    #the probability density function (PDF) of the original values in the column of
    #df is calculated using the pdf method of the distribution objects. 
    #These probabilities are stored in prob1_dict[column] and prob0_dict[column], respectively.
        prob1 = fit_distribution(stroke1[column]) #34-35. Two new DataFrames, prob0df and prob1df,
        #are created. prob0df calculates the product of the probabilities in each row of
        #prob0_dict, and multiplies it by the prior probability of the "No" class. Similarly, prob1df
        #calculates the product of the probabilities in each row of prob1_dict,
        #and multiplies it by the prior probability of the "Yes" class.
    for column in numeric: 
        prob0 = fit_distribution(stroke0[column])
        prob1_dict[column] = prob1.pdf(df[column])
        prob0_dict[column] = prob0.pdf(df[column])
    prob0df = pd.DataFrame(prob0_dict).prod(axis=1)*prior["No"]
    prob1df = pd.DataFrame(prob1_dict).prod(axis=1)*prior["Yes"]
    Predicted = (prob0df < prob1df)*1 #The Predicted variable is assigned the result of a
    #comparison between prob0df and prob1df, using the less-than operator <. 
    #The resulting boolean Series is converted to 0s and 1s by multiplying it with 1.
    final = pd.concat([df[target], Predicted], axis=1).rename({target:'Actual', 0:'Predicted'}, axis=1).replace({1:"Yes", 0:"No"}) 
    #The final DataFrame is created by concatenating
    #the df[target] (the actual values) and Predicted columns. 
    #The column names are renamed to 'Actual' and 'Predicted' using the
    #rename method, and the values in the DataFrame are replaced: 1 is replaced with "Yes" 
    #and 0 with "No" using the replace method.
    print(pd.crosstab(final['Actual'], final['Predicted'])) #The cross-tabulation of
    #the 'Actual' and 'Predicted' columns in the final DataFrame is printed using
    #the crosstab function from pandas.
    return final #The final DataFrame is returned from the function.''')    
    
    
    

if selected=="Data Analysis":
    st.title("Data Analysis")
    st.markdown("On this page, I would show you some of my conclusions that I drew from the dataset and graphs that I made.")
    
    st.subheader('Accuacy')
    st.markdown("""When assessing the performance of a Naive Bayes model, accuracy plays a crucial role. It provides a measure of how well the model predicts the correct class labels. To establish a baseline for comparison, we can determine the accuracy we would achieve by simply predicting the most common class for all observations in the dataset, without performing any calculations.

To calculate the baseline accuracy, we can utilize the 'value_counts(normalize=True)' method from the Pandas library. This method allows us to determine the percentage of occurrences for each category of the 'Profit Indicator' variable. By identifying the most frequent category, we can ascertain the accuracy we would obtain by blindly predicting that class for all instances.

For instance, let's assume that the most frequent category of 'Profit Indicator' in our dataset is 'negative profit', accounting for 95% of the observations. This implies that if we were to predict 'positive profit' for every instance, we would achieve an accuracy of 95%. Consequently, any Naive Bayes model we develop should strive to exceed this baseline accuracy in order to be considered effective.""")
    st.table(stroke['stroke'].value_counts(normalize=True))

    st.subheader('Using Chi-Square to Show Relevancy')
    st.markdown('''Given that we have selected the 'Stroke' as our target variable for prediction, we want to determine the significance of the other variables in relation to it. The following dataframe contains the categorical variables, along with their corresponding chi-square values and p-values with respect to the 'Stroke'.''')
    
    def get_key(val):
        for k,v in pretty_num_names.items():
            if val==v:
                return k
        for k,v in cat_name.items():
            if val==v:
                return k

    chi_data_analysis = pd.DataFrame([[i] + list(stats.chi2_contingency(pd.crosstab(stroke[i],                              stroke['stroke']))[0:2]) for i in cat_name], columns=['Predictor', 'Chi-Square Stat', 'P-Value'])
    

    chi_data_analysis = chi_data_analysis.sort_values('Chi-Square Stat',    ascending=False).reset_index(drop=True)
    chi_data_analysis['Predictor'] = chi_data_analysis['Predictor'].replace(pretty_cat_names)
    st.dataframe(chi_data_analysis)
    st.markdown('''
The dataframe provides important insights for our analysis by presenting the magnitude of Chi-Square values and P-values. These metrics are crucial in determining the relevance of variables and selecting the most accurate model.

The Chi-Square value of a variable indicates its level of association with the target variable. A higher Chi-Square value suggests a stronger relationship with the target. Conversely, a larger p-value implies less relevance. Upon examining the dataframe, it becomes evident that the 'Gender' and 'Residence type' columns have relatively low significance in relation to stroke. Therefore, we can prioritize them less in our analysis. Conversely, the 'Heart Disease' column exhibits the highest relevance to stroke, indicating its importance. As a result, we should consider 'Heart Disease' as our first choice in the analysis.''')
    



    melted_num=pd.melt(stroke,id_vars=['stroke'],value_vars=[get_key(i) for i in list(pretty_num_names.values())])
    melted_num=melted_num[melted_num['variable']!='stroke']
    corr_df=stroke[list(pretty_num_names.keys())].corr().round(4)
    st.subheader("Facet plots for selecting numeric predictors")
    fig6=px.box(melted_num,x='stroke',y='value',color='stroke',facet_col='variable',facet_col_wrap=3,facet_col_spacing=0.06,width=1000,height=500)
    fig6.for_each_annotation(lambda a: a.update(text=f'<b>{pretty_num_names[a.text.split("=")[-1]]}</b>', font_size=14))
    fig6.update_yaxes(matches=None,title='',showticklabels=True)
    fig6.update_xaxes(showticklabels=False,title_text='')
    fig6.add_annotation(x=-0.05,y=0.4,text="<b>Value of Each Variable</b>", textangle=-90, xref="paper", yref="paper",font_size=14
)
    st.markdown("This facet plot is very important as it allows us to directly see the means of each variable with respect to positive and negative profits. Therefore, the greater the difference in means between the two categories, the more influence the variable has on the target variable, which we take into account when deciding on our predictors.")    
    
    st.plotly_chart(fig6)
    st.markdown("")
    num_name_pretty = [pretty_num_names[col] for col in stroke[num_name].columns]
    imshow=px.imshow(stroke[num_name].corr(),text_auto=True,width=600,height=600,color_continuous_scale='inferno',title="<b>Correlation of Each Numeric Variable")
    imshow.update_traces()
    imshow.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(len(num_name_pretty))),
                                 ticktext=num_name_pretty),
                      yaxis=dict(tickmode='array', tickvals=list(range(len(num_name_pretty))),
                                 ticktext=num_name_pretty))
    st.plotly_chart(imshow)
    
    st.markdown('''This heatmap visually represents the correlations between each numeric variable in our dataset. The strength of correlation is shown through varying color depths. Strong correlations are depicted by red colors, while weak correlations are represented by blue colors. It's important to note that the Naive Bayes algorithm assumes independence between predictor variables. However, in some cases, numeric variables may show a strong positive correlation as evident from their dark red color on the heatmap. This suggests that these variables are closely related and should not be used together as predictors.

To identify significant and independent variables for the target variable 'Stroke' we analyze both the heatmap and the Chi-Square chart. The Chi-Square chart helps us identify variables with high significance, while the heatmap helps us check for independence between variables. Variables with the highest Chi-Square and the largest difference in means are preferred as predictors. Additionally, we should ensure that these selected variables have lighter colors on the heatmap, indicating low correlation with each other and preserving the assumption of independence.''')

    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.markdown('''After considering the Chi-Square, facet plot and the heatmap, I have decided to use the following variables as predictors: 
                1. Hypertension
                2. Ever Married
                3. Work Type
                4. Smoking Status
                5. Residence Type
                6. Genger
                7. Age''')
    st.markdown('Other variables are highly corelated with at least one of these variables, therefore we would not be choosing them. However, I am not able to raise the accuacy level above the baseline perentage.')


if selected=="Conclusion":
    st.title("Conclusion")
    st.markdown("In conclustion, strokes are dangrous diseases that happen when there is a blokage in the brain. Studing the causes of strokes could save many peoples lifes from predicting them with models. Our model had succeeded in predicting 95% of the stroke cases and  with improvement, could have an even higher accuacy.")

if selected=="Bibliography":
    st.title("Bibliography")
    st.markdown("""Works Cited “Airplane Crashes and Fatalities.” Www.kaggle.com, www.kaggle.com/datasets/thedevastator/airplane-crashes-and-fatalities.""") # Add harry case study to bibliography.