# Importing libraries and packages
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# List of all symptoms
l1 = ['anorexia', 'abdominal_pain', 'anaemia', 'abortions', 'acetone', 'aggression', 'arthrogyposis',
      'ankylosis', 'anxiety', 'bellowing', 'blood_loss', 'blood_poisoning', 'blisters', 'colic',
      'Condemnation_of_livers', 'coughing', 'depression', 'discomfort', 'dyspnea', 'dysentery',
      'diarrhoea', 'dehydration', 'drooling', 'dull', 'decreased_fertility', 'diffculty_breath',
      'emaciation', 'encephalitis', 'fever', 'facial_paralysis', 'frothing_of_mouth', 'frothing',
      'gaseous_stomach', 'highly_diarrhoea', 'high_pulse_rate', 'high_temp', 'high_proportion',
      'hyperaemia', 'hydrocephalus', 'isolation_from_herd', 'infertility', 'intermittent_fever',
      'jaundice', 'ketosis', 'loss_of_appetite', 'lameness', 'lack_of-coordination', 'lethargy',
      'lacrimation', 'milk_flakes', 'milk_watery', 'milk_clots', 'mild_diarrhoea', 'moaning',
      'mucosal_lesions', 'milk_fever', 'nausea', 'nasel_discharges', 'oedema', 'pain', 'painful_tongue',
      'pneumonia', 'photo_sensitization', 'quivering_lips', 'reduction_milk_vields', 'rapid_breathing',
      'rumenstasis', 'reduced_rumination', 'reduced_fertility', 'reduced_fat', 'reduces_feed_intake',
      'raised_breathing', 'stomach_pain', 'salivation', 'stillbirths', 'shallow_breathing',
      'swollen_pharyngeal', 'swelling', 'saliva', 'swollen_tongue', 'tachycardia', 'torticollis',
      'udder_swelling', 'udder_heat', 'udder_hardeness', 'udder_redness', 'udder_pain', 'unwillingness_to_move',
      'ulcers', 'vomiting', 'weight_loss', 'weakness']

# List of diseases
disease = ['mastitis', 'blackleg', 'bloat', 'coccidiosis', 'cryptosporidiosis', 'displaced_abomasum',
           'gut_worms', 'listeriosis', 'liver_fluke', 'necrotic_enteritis', 'peri_weaning_diarrhoea',
           'rift_valley_fever', 'rumen_acidosis', 'traumatic_reticulitis', 'calf_diphtheria', 'foot_rot',
           'foot_and_mouth', 'ragwort_poisoning', 'wooden_tongue', 'infectious_bovine_rhinotracheitis',
           'acetonaemia', 'fatty_liver_syndrome', 'calf_pneumonia', 'schmallen_berg_virus', 'trypanosomosis',
           'fog_fever']

l2 = [0] * len(l1)

# Reading the Cattle training Dataset .csv file
df = pd.read_csv("training.csv")
DF = pd.read_csv('training.csv', index_col='prognosis')

df.replace({'prognosis': {'mastitis': 0, 'blackleg': 1, 'bloat': 2, 'coccidiosis': 3, 'cryptosporidiosis': 4,
                           'displaced_abomasum': 5, 'gut_worms': 6, 'listeriosis': 7, 'liver_fluke': 8,
                           'necrotic_enteritis': 9, 'peri_weaning_diarrhoea': 10, 'rift_valley_fever': 11,
                           'rumen_acidosis': 12, 'traumatic_reticulitis': 13, 'calf_diphtheria': 14,
                           'foot_rot': 15, 'foot_and_mouth': 16, 'ragwort_poisoning': 17, 'wooden_tongue': 18,
                           'infectious_bovine_rhinotracheitis': 19, 'acetonaemia': 20, 'fatty_liver_syndrome': 21,
                           'calf_pneumonia': 22, 'schmallen_berg_virus': 23, 'trypanosomosis': 24, 'fog_fever': 25}},
           inplace=True)

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df1, nGraphShown, nGraphPerRow, nGraphRow):
    nunique = df1.nunique()
    df1 = df1[[col for col in df1 if nunique[col] > 1 and nunique[col] < 50]]
    nRow, nCol = df1.shape
    columnNames = list(df1)
    plt.figure(num=None, figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80, facecolor='w', edgecolor='k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df1.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot(kind='bar')
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()
# Scatter and density plots
def plotScatterMatrix(df1, plotSize, textSize):
    df1 = df1.select_dtypes(include=[np.number])
    df1 = df1.dropna('columns')
    df1 = df1[[col for col in df1 if df1[col].nunique() > 1]]
    columnNames = list(df1)
    if len(columnNames) > 10:
        columnNames = columnNames[:10]
    df1 = df1[columnNames]
    ax = pd.plotting.scatter_matrix(df1, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df1.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center',
                           va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

plotPerColumnDistribution(df, 10, 5, 2)
plotScatterMatrix(df, 20, 10)

X = df[l1]
y = df[["prognosis"]]
np.ravel(y)
y = y.values

# Plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[l1[0]], X[l1[1]], X[l1[2]], c='r', marker='o')
ax.set_xlabel(l1[0])
ax.set_ylabel(l1[1])
ax.set_zlabel(l1[2])
plt.show()

# Training the models
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

clf3 = tree.DecisionTreeClassifier()
clf3 = clf3.fit(X, y)

clf4 = RandomForestClassifier()
clf4 = clf4.fit(X, np.ravel(y))

gnb = GaussianNB()
gnb = gnb.fit(X, np.ravel(y))

knn = KNeighborsClassifier()
knn = knn.fit(X, np.ravel(y))

# GUI Section
from tkinter import *

root = Tk()
root.configure(background='black')
root.title("Cattle Disease Prediction")
root.geometry('800x400')

Label(root, text="Enter the name of the Cattle", font=('arial', 16, 'bold'), fg="yellow", bg="black").grid(row=0, column=0)
Symptom1 = Entry(root, width=60, bg="yellow")
Symptom1.grid(row=0, column=1)

Label(root, text="Select Symptoms of Cattle", font=('arial', 16, 'bold'), fg="yellow", bg="black").grid(row=1, column=0)

i = 1
for k in range(len(l1)):
    i += 1
    Checkbutton(root, text=l1[k], fg="yellow", bg="black", variable=l2[k]).grid(row=i, sticky=W)

def predict_and_display(predicted):
    for l in range(len(disease)):
        if predicted[0] == l:
            t1.delete("1.0", END)
            t1.insert(END, disease[l])

def DecisionTree():
    inputSymptoms = [l2[k].get() for k in range(len(l1))]
    predicted = clf3.predict([inputSymptoms])
    predict_and_display(predicted)

def RandomForest():
    inputSymptoms = [l2[k].get() for k in range(len(l1))]
    predicted = clf4.predict([inputSymptoms])
    predict_and_display(predicted)

def NaiveBayes():
    inputSymptoms = [l2[k].get() for k in range(len(l1))]
    predicted = gnb.predict([inputSymptoms])
    predict_and_display(predicted)

def KNN():
    inputSymptoms = [l2[k].get() for k in range(len(l1))]
    predicted = knn.predict([inputSymptoms])
    predict_and_display(predicted)

Button(root, text="Decision Tree Prediction", command=DecisionTree, bg="cyan").grid(row=i + 1, sticky=W)
Button(root, text="Random Forest Prediction", command=RandomForest, bg="cyan").grid(row=i + 2, sticky=W)
Button(root, text="Naive Bayes Prediction", command=NaiveBayes, bg="cyan").grid(row=i + 3, sticky=W)
Button(root, text="K-Nearest Neighbors Prediction", command=KNN, bg="cyan").grid(row=i + 4, sticky=W)
t1 = Text(root, height=2, width=30, font=('arial', 16, 'bold'), fg="yellow", bg="black")
t1.grid(row=15, column=1)
root.mainloop()

