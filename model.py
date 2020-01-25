from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
#Import os package
import os
import pickle

def read_data():
    # set the path of the raw data
    raw_file_path = os.path.join(os.path.pardir, 'data', 'raw')
    raw_data_file_path = os.path.join(raw_file_path, 'gender_classification_data.xlsx')
    raw_data = pd.read_excel(raw_data_file_path)
    
    return raw_data


def  process_data(raw_data):
            raw_data.loc[raw_data['Gender']=='Female', 'Shoe_Size'].mean()
            raw_data.iloc[4:5, 2:3] = raw_data.iloc[4:5, 2:3].fillna(35)
            raw_data.loc[raw_data['Height']==146, 'Weight']
            raw_data.iloc[14:15, 1:2] = raw_data.iloc[14:15, 1:2].fillna(50)
            raw_data.loc[raw_data['Height']==180, 'Shoe_Size'].mean()
            raw_data.iloc[18:19, 2:3] = raw_data.iloc[18:19, 2:3].fillna(42)
            raw_data.loc[raw_data['Height']==178, 'Weight'].mean()
            raw_data.iloc[37:38, 1:2] = raw_data.iloc[37:38, 1:2].fillna(78)
            
            raw_data.loc[raw_data['Weight']>125, 'Weight']
            raw_data.drop(raw_data.index[73], inplace = True)
            
            # Dropping redundant column
            raw_data.drop(['Height Inches'], axis = 1, inplace=True)
            return raw_data
            
def write_data(raw_data):
            processed_data_path = os.path.join(os.path.pardir, 'data', 'processed')
            write_processed_data_path = os.path.join(processed_data_path, 'processed_gender_classificaiton_data.xlsx')
            raw_data.to_excel(write_processed_data_path, index=False)
            processed_data = pd.read_excel(write_processed_data_path)
            return processed_data

def running_model(raw_data):
            train, test = train_test_split(processed_data, test_size = 0.2)
            features = ['Height', 'Weight', 'Shoe_Size']
            X_train = train[features]
            y_train = train['Gender']
            X_test = test[features]
            y_test = test['Gender']
            # Creating the model
            clf = tree.DecisionTreeClassifier()
            # Training the model
            clf = clf.fit(X_train, y_train)
            # Saving model to disk
            pickle.dump(clf, open('../model.pkl','wb'))
            # Loading model to compare the results
            model = pickle.load(open('../model.pkl','rb'))
            print(model.predict(X_test))


if __name__ == '__main__':
                  raw_data = read_data()
                  raw_data = process_data(raw_data)
                  processed_data = write_data(raw_data)
                  running_model(processed_data)
