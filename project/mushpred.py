import seaborn as sns
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics as mat
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.model_selection import train_test_split as tts

st.set_page_config(page_title="Mushroom Classification", page_icon="🍄", layout="wide")

st.title("🍄 Mushroom Data Analysis and Classification")

mdf = pd.read_csv("mushrooms.csv")
st.header("Mushroom Dataset")
st.dataframe(mdf.head())

le = LabelEncoder()
for column in mdf.columns:
    mdf[column] = le.fit_transform(mdf[column])

st.subheader("Encoded Dataset")
st.dataframe(mdf.head())


x = mdf.drop('class', axis=1)
y = mdf['class']

mushroom_model = dtc(criterion='entropy', random_state=0)


xtrain, xtest, ytrain, ytest = tts(x, y, test_size=0.2, random_state=42)
mushroom_model.fit(xtrain, ytrain)


ypred = mushroom_model.predict(xtest)


st.header("Classification Report")
st.table(mat.classification_report(ytest, ypred, output_dict=True))


st.header("Prediction")
n1 = st.number_input("Enter cap-shape value")
n2 = st.number_input("Enter cap-surface value")
n3 = st.number_input("Enter cap-color value")
n4 = st.number_input("Enter bruises value")
n5 = st.number_input("Enter odor value")
n6 = st.number_input("Enter gill-attachment value")
n7 = st.number_input("Enter gill-spacing value")
n8 = st.number_input("Enter gill-size value")
n9 = st.number_input("Enter gill-color value")
n10 = st.number_input("Enter stalk-shape value")
n11 = st.number_input("Enter stalk-root value")
n12 = st.number_input("Enter stalk-surface-above-ring value")
n13 = st.number_input("Enter stalk-surface-below-ring value")
n14 = st.number_input("Enter stalk-color-above-ring value")
n15 = st.number_input("Enter stalk-color-below-ring value")
n16 = st.number_input("Enter veil-type value")
n17 = st.number_input("Enter veil-color value")
n18 = st.number_input("Enter ring-number value")
n19 = st.number_input("Enter ring-type value")
n20 = st.number_input("Enter spore-print-color value")
n21 = st.number_input("Enter population value")
n22 = st.number_input("Enter habitat value")


sample1 = [[
    int(n1), int(n2), int(n3), int(n4), int(n5), int(n6), int(n7),
    int(n8), int(n9), int(n10), int(n11), int(n12), int(n13),
    int(n14), int(n15), int(n16), int(n17), int(n18), int(n19),
    int(n20), int(n21), int(n22)
]]

if st.button("Predict Mushroom Class"):
    target_sp = mushroom_model.predict(sample1)
    proba = mushroom_model.predict_proba(sample1)
    st.write("Probability:", proba)
    st.write("Prediction (0 = Edible, 1 = Poisonous):", target_sp)

    if target_sp == 1:
        st.write("This mushroom is likely Poisonous.")
    else:
        st.write("This mushroom is likely Edible.")
