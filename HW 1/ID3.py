    #   Homework Assignment 01: Decision Trees
#   Name: Ishan Joshi
#   USC ID- 3451334123


import pandas as pd

from pprint import pprint
import numpy as np

# Taking input data and processing it to create Dataframe

Input_file = pd.read_csv('dt_data.txt', header=None, skipinitialspace=True,engine='python')


Input_file.to_csv(r'dt_data.csv', index=None)
col_names = Input_file[0:1]
col_names = [(s).replace("(", "").replace(")", "") for s in col_names.unstack()]
Input_file.columns = col_names
Input_file = Input_file[1:len(Input_file)]
output_label = Input_file["Enjoy"]
Input_file["Enjoy"] = Input_file["Enjoy"].apply(lambda x: str(x).replace(";", ""))
Input_file["Occupied"] = Input_file["Occupied"].apply(lambda x: str(x).replace(": ", ""))
Input_file["Occupied"] = Input_file["Occupied"].apply(lambda  x: str(x).replace("0", "")
                                                      .replace("1", "")
                                                      .replace("2", "")
                                                      .replace("3", "")
                                                      .replace("4", "")
                                                      .replace("5", "")
                                                      .replace("6", "")
                                                      .replace("7", "")
                                                      .replace("8", "")
                                                      .replace("9", ""))

feature_cols = []
for i in range(len(col_names) - 1):
    feature_cols.append(col_names[i])

#   Input query for prediction
query = {'Occupied':'Moderate','Price':'Cheap','Music':'Loud','Location':'City-Center','VIP':'No','Favorite Beer':'No'}

# Calculating initial Entropy of Output Label
def Entropy_Calc(Label_RV):
    Label_RV = Label_RV
    initial_entropy = 0
    prob = []
    #Calculating probability
    unique_array, counts = np.unique(Label_RV, return_counts=True)
    for i in range (len(unique_array)):
        prob.append(counts[i] / np.sum(counts))
        initial_entropy += (-prob[i]) * (np.log2(prob[i]))
    return (initial_entropy)


# Calculating Entropy of Attributes
def Info_Gain(data, split_attribute, target_value = "Enjoy"):
    split_attribute_vals, split_attribute_count = np.unique(data[split_attribute], return_counts= True)
    original_entropy = Entropy_Calc(Label_RV= output_label)
    Weighted_Entropy = 0
    for i in range(len(split_attribute_vals)):
        split_node = data[data[split_attribute] == split_attribute_vals[i]][target_value]
        split_entropy = Entropy_Calc(Label_RV=split_node)
        Weighted_Entropy += ((split_attribute_count[i] / np.sum(split_attribute_count)) * split_entropy)
    Information_Gain = original_entropy - Weighted_Entropy
    return(Information_Gain)



#   Building the Tree
def Tree_Build(features, sub_data, Input_file=Input_file, target_value="Enjoy", parent_node =None):
    current_val = np.unique(sub_data[target_value])
    if ((len(current_val)) <= 1):
        temp_val = sub_data[target_value]
        temp_val_vals = np.unique(temp_val)[0]
        return temp_val_vals
    elif len(sub_data) == 0:
        return np.unique(Input_file[target_value])[np.argmax(np.unique(Input_file[target_value], return_counts=True)[1])]
    elif (len(features) == 0):
        return (parent_node)
    else:
        parent_node = np.unique(sub_data[target_value])[np.argmax(np.unique(sub_data[target_value],return_counts=True)[1])]
        Entropy_arr = []
        for i in range (len(features)):
            ent = Info_Gain(data=sub_data, split_attribute=features[i])
            Entropy_arr.append(ent)
        max_info_gain_pass_index = np.argmax(Entropy_arr)
        best_first_feature = features[max_info_gain_pass_index]
        best_first_feature_vals, best_first_feature_counts = np.unique(sub_data[best_first_feature], return_counts=True)
        tree = {best_first_feature : {}}

        features = [i for i in features if i != best_first_feature]

        for i in (best_first_feature_vals):
            sub_val = i
            split_data = sub_data[sub_data[best_first_feature] == sub_val]
            sub_tree = Tree_Build(features=features, sub_data=split_data, Input_file=Input_file, target_value="Enjoy", parent_node=parent_node)
            tree[best_first_feature][sub_val] = sub_tree
        


    return tree

#   Prediction for a given input query
def predict(query, tree):
    if type(tree) is dict:
        i = list(tree.keys())[0]
        if i in query:
            return predict(query=query, tree=tree[i][query[i]])
        else:
            return i
    else:
        return tree







#   Printing the Tree
output = Tree_Build(features= feature_cols, sub_data= Input_file)
pprint(output)



#   Printing the prediction for the given input query
prediction = predict(query= query, tree=output)
print("The prediction for the given query is:")
pprint(prediction)
