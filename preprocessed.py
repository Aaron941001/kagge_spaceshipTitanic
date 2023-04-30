import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load data into a Pandas DataFrame

for data in ['train.csv', 'test.csv']:
    df = pd.read_csv(data)

    df = df.drop(columns=['Name'])

    ### Cabin
    def split_cabin(x):
        if len(str(x).split('/')) < 3:
         return ['Missing', 'Missing', "Missing"]
        else:   
            return str(x).split('/')

    df['TempCabin'] = df['Cabin'].apply(lambda x: split_cabin(x))
    df['Deck'] = df['TempCabin'].apply(lambda x: x[0])
    df['Side'] = df['TempCabin'].apply(lambda x: x[2])

    # Fill missing values in numerical columns with median and in categorical columns with 'FALSE'
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    cat_cols = ['CryoSleep', 'VIP']
    for col in num_cols:
        imputer = SimpleImputer(strategy='median')
        df[col] = imputer.fit_transform(df[[col]])
    for col in cat_cols:
        df[col] = df[col].fillna(False)

    df['Destination'] = df['Destination'].fillna('NONE')
    df['HomePlanet'] = df['HomePlanet'].fillna('NONE')

    # Fill missing values in RoomService, FoodCourt, ShoppingMall, Spa, and VRDeck with 0
    df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)

    # Create a new column for total spending
    df['TotalSpending'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']


    # One-hot encode HomePlanet, CryoSleep, and VIP columns
    encoder = OneHotEncoder()
    encoded_cols = pd.DataFrame(encoder.fit_transform(df[['HomePlanet', 'CryoSleep', 'VIP', 'Destination', 'Side', 'Deck']]).toarray(), columns=encoder.get_feature_names(['HomePlanet', 'CryoSleep', 'VIP', 'Destination', 'Side', 'Deck']))

    df = pd.concat([df, encoded_cols], axis=1)

    # Drop the original columns
    df = df.drop(columns=['HomePlanet', 'CryoSleep', 'VIP', 'Destination', 'Cabin', 'Deck', 'Side', 'TempCabin'])

    # Preview the updated DataFrame
    print(df.tail())
    missing_values = df.isna().sum()
    print(missing_values)
    df.to_csv(data, index=False)
