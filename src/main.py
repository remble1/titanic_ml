import pandas as pd
from titanic_logo import logo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")
df_gs = pd.read_csv("data/gender_submission.csv")
pd.set_option('display.max_columns', None)

def titanic_eda(df):
    print(logo)
    # names
    df = df.drop('Name', axis=1)
    df = df.drop('Cabin', axis=1)
    df = df.drop('Ticket', axis=1)
    df = df.drop('Embarked', axis=1)

    df.rename(columns={'Fare': 'TPrice'}, inplace=True)

    # map names
    sex_mapping = {'male': 1, 'female': 0}
    df['Sex'] = df['Sex'].map(sex_mapping)

    # types
    def is_float(value):
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False


    def is_integer(value):
        try:
            int(value)
            return True
        except (ValueError, TypeError):
            return False

    # change type from int to float
    df.dropna(subset=['Age'], inplace=True)
    df['Age'] = df['Age'].astype(int)
        
    is_passanger_integer = df['PassengerId'].apply(is_integer).all()
    try:
        is_survived_integer = df['Survived'].apply(is_integer).all()
    except:
        print("That df dont have Survived")
    is_pclass_integer = df['Pclass'].apply(is_integer).all()
    is_sex_integer = df['Sex'].apply(is_integer).all()
    is_age_integer = df['Age'].apply(is_integer).all()
    is_sibsp_integer = df['SibSp'].apply(is_integer).all()
    is_parch_integer = df['Parch'].apply(is_integer).all()
    is_tprice_integer = df['TPrice'].apply(is_float).all()

    print(f"'Passager' contain only int: {is_passanger_integer}")
    try:
        print(f"'Survived' contain only int: {is_survived_integer}")
    except:
        print("That df dont have Survived")
    print(f"'Pclass' contain only int: {is_pclass_integer}")
    print(f"'Sex' contain only int: {is_sex_integer}")
    print(f"'Age' contain only int: {is_age_integer}")
    print(f"'Sibsp' contain only int: {is_sibsp_integer}")
    print(f"'Parch' contain only int: {is_parch_integer}")
    print(f"'Tprize' contain only int: {is_tprice_integer}")
    print("Finall data:")
    print(df.head())
    return df

train = titanic_eda(df_train)
print("train data")

test = titanic_eda(df_test)
print("test data")


x = train.drop('Survived', axis=1)  
y = train['Survived'] 
print(x)
print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


classifier = LogisticRegression()
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


predictions = classifier.predict(test)

print(predictions, "Result on test dataframe")
