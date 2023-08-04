# Import necessary libraries
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Read data
data_stats = pd.read_csv('data.csv')

# Convert "Date" column to a timestamp
data_stats["Date"] = pd.to_datetime(data_stats["Date"], format="%d/%m/%Y").astype('int64') / 10**9

# Convert categorical columns to numerical values
team_labels = {}
for col in ['HomeTeam', 'AwayTeam']:
    le = LabelEncoder()
    data_stats[col] = le.fit_transform(data_stats[col])
    team_labels.update({index: label for index, label in enumerate(le.classes_)})

data_stats['FTR'] = data_stats['FTR'].replace({'H': 1, 'A': 2, 'D': 0})

# Create empty lists to store the rolling averages
home_stats = []
away_stats = []

# loop over each row in the dataframe
for i, row in data_stats.iterrows():
    # get the home and away teams for this match
    home_team = row["HomeTeam"]
    away_team = row["AwayTeam"]
    
    # filter the dataframe to get the last 5 matches for each team
    home_matches = data_stats[(data_stats["HomeTeam"] == home_team) | (data_stats["AwayTeam"] == home_team)].iloc[:-1].tail(5)
    away_matches = data_stats[(data_stats["HomeTeam"] == away_team) | (data_stats["AwayTeam"] == away_team)].iloc[:-1].tail(5)
    
    # calculate the rolling averages for each statistic for each team
    home_avg = home_matches[["FTHG", "HS", "HST", "HF", "HC", "HY", "HR"]].mean()
    away_avg = away_matches[["FTAG", "AS", "AST", "AF", "AC", "AY", "AR"]].mean()
    
    # append the rolling averages to our lists
    home_stats.append(home_avg)
    away_stats.append(away_avg)

# create new columns in the dataframe for the rolling averages
data_stats[["HomeAvgGoals", "HomeAvgShots", "HomeAvgShotsOnTarget", "HomeAvgFouls", "HomeAvgCorners", "HomeAvgYellowCards", "HomeAvgRedCards"]] = pd.DataFrame(home_stats)
data_stats[["AwayAvgGoals", "AwayAvgShots", "AwayAvgShotsOnTarget", "AwayAvgFouls", "AwayAvgCorners", "AwayAvgYellowCards", "AwayAvgRedCards"]] = pd.DataFrame(away_stats)

# drop the original statistic columns
data_stats.drop(["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"], axis=1, inplace=True)

# Split dataset into training and testing sets based on Date column
split_date = pd.Timestamp('2023-05-15').timestamp()
train_stats = data_stats[data_stats['Date'] <= split_date]
upcoming_stats = data_stats[data_stats['Date'] > split_date].drop(['FTR'], axis=1)

# Hyperparameter tuning using GridSearchCV
params = {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 1.0}

X_train_stats = train_stats.drop(['FTR'], axis=1)
y_train_stats = train_stats['FTR']
model_stats = xgb.XGBClassifier(**params)
model_stats.fit(X_train_stats, y_train_stats)

# Reset the index column to start from 0
upcoming_stats = upcoming_stats.reset_index(drop=True)


# Print the percentage of success for each outcome
print("UPCOMING stats")
for index, row in upcoming_stats.iterrows():
    home_team = team_labels[row['HomeTeam']]
    away_team = team_labels[row['AwayTeam']]
    probs = model_stats.predict_proba(upcoming_stats.iloc[[index]])[0]
    success_rate_H = round(probs[1]*100,2)
    success_rate_A = round(probs[2]*100,2)
    success_rate_D = round(probs[0]*100,2)
    print(f"{home_team} vs {away_team}: 1 = {success_rate_H}%, X = {success_rate_D}%, 2 = {success_rate_A}%")
