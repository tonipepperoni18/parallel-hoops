
# Parallel Hoops
## Overview

Parallel Hoops is sports data analytics project that compares NBA and WNBA players using statistical profiles and similarity modeling. 

The goal of the project is to answer questions such as:

+ Which NBA player is the most statistically similar to A'ja Wilson?
+ Which WNBA players most close resembles Victor Wembanyama?
+ How do players compare whenn measure relative to their own leagues

Instead of relying on raw statistics, Parallel Hoops normalizes player performance within each league before generating comparisons. This allows players from different leagues to be evaluated on the same scale and compared based on role, play style, and overall impact. 

## Project Goals

The project was built to:

+ Practice real-world engineering workflows
+ Develop sports analytics and machine learning skills
+ Create a reusable player comparison engine
+ Explore cross-league NBA and WNBA player comparisons
+ Build a portfolio-quality data science project

## Features
  ### Data Pipeline
+ Automated CSV loading

+ Data cleaning and standardization

+ Player profile generation

+ Feature engineering 

+ League-specific normalization

+ SQLite database storage

  ### Similarity Engine
+ Weighted player comparison model
+ Category- based similarity scoring
+ Cross-league NBA/WNBA comparisons
+ Offensive, defensive, and impact anaylsis
+ Position-aware filtering

## Statistical Categories

### Offense (35%)
#### Scoring (17.5%)
+ Points 
+ True Shooting
+ Effective Field Goal Percentage
+ Free throw rate
+ Shot distribution
+ Shooting effiency

#### Playmaking (17.5%)
+ Assist 
+ AST %
+ Turnover %
+ Offensive Rating
+ Offensive Win Shares


### Defense (35%)
#### Individual (17.5%)
+ Steals 
+ Blocks
+ STL %
+ BLK %

#### Team Defense(17.5%)
+ Defensive Rating
+ Defensive Win Shares
+ Defensive Rebound %

### Intangibles (35%)

+ PER 
+ Win shares
+ Impact Metrics



## Methodology

Parallel Hoops uses league normalization to compare players across leagues. For each statistical category, player performance is converted into a standarized score using a z-score:

``` 
z= (player_stat - league_average) / league_standard_deviation
```

This allows the model to compare players based on how dominant they are relative to their own league rather than comparing raw statistics directly. 

After normalization, a weighted similarity engine generated player comparisons using offensive, defensive, and impact metrics. 

## Tech stack

#### Languages
+ Python

#### Libraries
+ Pandas
+ NumpPy
+ Scikit-Learn
+ SciPy
+ SQLite

#### Tools
+ Git
+ VsCode



## Current Status
### Version 1 Mk 1 (We are here)
### Version 1 Mk 2
+ Archetype Classification (Two Way star, Playmaking Big, Shot creator )
+ Player comparison dashboard
+ Visualization Tools
+ Historical season comparisons
+ Web application deployment
  

## Examples

Who is the closest NBA comparison to A'ja Wilson?

```
find_league_comps(
    "A'ja Wilson",
    combined_df,
    target_league="NBA",
    top_n=20
)
```


