# EDA - Exploratory Data Analysis on Spotify 2023 Dataset  
### **Christian Justin M. Fernando | 2ECE-B | ECE2112**

---

## 🎧 Project Overview

In this project, we will conduct an **Exploratory Data Analysis (EDA)** on a dataset containing detailed information about the **Most Streamed Spotify Songs of 2023**. This dataset offers a comprehensive view of the music scene in 2023, capturing essential metrics of top songs, such as **stream counts**, **musical attributes** (e.g., BPM, energy, danceability), and **artist performance**.

The dataset is sourced from Kaggle and can be accessed [here](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023). 🎶

By analyzing these variables, we aim to uncover trends, patterns, and insights that shed light on what drives a song's success on Spotify in 2023.

---

## 🔍 Analysis Framework

To structure this analysis, we will follow a systematic approach:

- **📂 Explore the Dataset**:  
  Examine the dataset’s structure, check for missing values, and analyze data types. Gain a clear understanding of the available features and their relationships.

- **📊 Summary Statistics**:  
  Calculate key metrics, including total streams, release dates, and musical attributes like BPM, danceability, and energy. This will provide an overview of the dataset’s most important characteristics.

- **📈 Visualize Trends**:  
  Create insightful visualizations, including bar charts, histograms, and scatter plots, to identify trends in song popularity and feature distributions. Each plot will be carefully labeled for clarity and ease of interpretation.

- **💡 Correlation Analysis**:  
  Investigate relationships between numerical variables (e.g., streams vs. tempo, streams vs. energy). This step will reveal insights into which features are most strongly correlated with high streaming performance.

- **💭 Insights & Recommendations**:  
  Synthesize the findings from the analysis to generate actionable insights. Based on the data, we’ll explore which factors contribute most to a track’s popularity and offer recommendations for artists, producers, and marketers.

---
## Questions and Data Analysis - EDA on Spotify 2023 Dataset

```python
#Import all the necessary libraries for the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
💡 This code imports essential libraries for Exploratory Data Analysis (EDA) on the Spotify 2023 dataset. It uses NumPy for numerical operations, Pandas for data manipulation, and Matplotlib/Seaborn for data visualization.

```python
#Load the `spotify-2023.csv` file
#Use `latin1` to ensure that special characters are registered
df = pd.read_csv('spotify-2023.csv', encoding = 'latin1')
```
💡 The dataset (spotify-2023.csv) is loaded with the latin1 encoding to correctly handle special characters. The purpose of this setup is to prepare the data for analysis, enabling exploration of trends, patterns, and insights from popular tracks, artists, and musical features in 2023.

### 1. Overview of Dataset 📋
- **How many rows and columns does the dataset contain?**
```python
#Getting the size of the dataset
print("Rows:", df.shape[0], "Columns:", df.shape[1])
```
💡 The dataset contains 953 rows and 24 columns, with each row representing a track and each column capturing a specific attribute, such as stream count, release date, artist, genre, and musical features. This structure enables analysis of trends and relationships in Spotify's most streamed songs of 2023.

- **What are the data types of each column? Are there any missing values?**
```python
#Checking the data types of each column
print(df.dtypes)
```
![image](https://github.com/user-attachments/assets/ab443fff-246f-4da3-a951-e039bd292ae6)

```python
#Checking if there are any missing values
print(df.isnull().sum())
```
![image](https://github.com/user-attachments/assets/9f2ae17d-fecd-40f9-9738-41b8e016f1b4)

💡 The dataset contains seven columns classified as objects and 17 columns as int64. Most columns contain complete data, with notable exceptions being in_shazam_charts and key, which have some missing values. Specifically, `in_shazam_charts` has 50 missing values, and `key` has 95 missing values. The `streams` column is also represented as an `object`, likely due to formatting issues, which may require conversion to a numerical type for further analysis. Overall, the dataset is mostly complete, but these missing values and data types will need to be addressed before analysis.

---

### 2. Basic Descriptive Statistics 📊
- **What are the mean, median, and standard deviation of the streams column?**  
```python
#Converts the 'streams' column to numeric values
#This changes any non-numeric values with Not a Number(NaN)
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')

#Calculate and print the mean
print("Mean:", df['streams'].mean())
#Calculate and print the median
print("Median:", df['streams'].median())
#Calculate and print the standard deviation
print("Standard Deviation:", df['streams'].std())
```
![image](https://github.com/user-attachments/assets/bb1253df-ab1b-4d63-9643-d2cdad87d1cd)

💡 The mean of 514,137,424.94 reflects the average number of streams across the dataset, indicating that tracks are generally streamed around 514 million times. The median of 290,530,915 is lower than the mean, suggesting a right-skewed distribution where a few tracks have significantly higher stream counts, pulling the average up. The standard deviation of 566,856,949.04 indicates high variability, meaning there are large differences in stream counts across tracks. Overall, the data shows that while some tracks are extremely popular, most receive considerably fewer streams.


- **What is the distribution of `released_year` and `artist_count`? Are there any noticeable trends or outliers?**
```python
#Histogram for 'released_year'
plt.figure(figsize=(10, 6))
sns.histplot(df['released_year'], bins=20, color='skyblue')  
plt.title('Histogram of Released Year')
plt.xlabel('Released Year')
plt.ylabel('Number of Released Tracks')
plt.show()
```
![image](https://github.com/user-attachments/assets/d5edce2e-8081-4871-9eec-008506ee34ae)

```python
#Histogram for 'artist_count'
plt.figure(figsize=(10, 6))
sns.histplot(df['artist_count'], bins=20, color='orange') 
plt.title('Histogram of Artist Count')
plt.xlabel('Number of Artists')
plt.ylabel('Count')
plt.show()
```
![image](https://github.com/user-attachments/assets/8d07151f-0681-414b-92a4-b8e6577184b1)

💡 The distribution of the released year shows a clear trend of increasing track releases over time, particularly in recent years. While there are a few releases dating back to the 1930s, the majority of tracks were released from the early 2000s onward. The highest release counts are observed in 2022 (402 tracks) and 2023 (175 tracks), with significant spikes in 2020 (37 tracks) and 2021 (119 tracks). The mean release year is around 2018, and the data suggests a concentration of releases in the last few years, with a notable decrease in the number of older releases.

The artist count distribution indicates that most tracks feature 1 to 2 artists, with 587 tracks having a single artist and 254 tracks having two artists. There are fewer tracks with more than two artists, with only a small number of tracks having 5 or more artists. The mean artist count is 1.56, reflecting the dominance of solo and duet performances in the dataset. The data shows a clear trend toward collaborations, but solo artists remain predominant.

```python
#Creating side-by-side plots to determine outliers
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

#Box plot for released_year on the first subplot (left side)
sns.boxplot(x=df['released_year'], color='lightgreen', ax=ax[0])
ax[0].set_title('Box Plot of Released Year')
ax[0].set_xlabel('Released Year')

#Box plot for artist_count on the second subplot (right side)
sns.boxplot(x=df['artist_count'], color='lightcoral', ax=ax[1])
ax[1].set_title('Box Plot of Artist Count')
ax[1].set_xlabel('Number of Artists')

#Adjust layout to avoid overlap
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/2e8856fc-8e47-4998-9420-57bf68a5eb08)

💡 There is a clear trend of increasing track releases in recent years, especially from 2020 to 2022, with a peak in 2022 (402 tracks). Older years show very few releases, with 1930 and 1942 as notable outliers. The sharp rise in recent years could reflect industry shifts, such as the influence of streaming. Most tracks feature 1 or 2 artists, reflecting the dominance of solo and duet performances. Outliers include a small number of tracks with 5 or more artists, suggesting multi-artist collaborations, but these are relatively rare.

---

### 3. Top Performers 🏆
- **Which track has the highest number of streams? Display the top 5 most streamed tracks.**  
```python
#Sort the DataFrame by 'streams' in descending order and select the top 5 rows
top_5_tracks = df.sort_values(by='streams', ascending=False)[['track_name', 'streams']].head(5)

#Pie chart to visualize the top 5 tracks based on number of streams
plt.figure(figsize=(8, 8))
plt.pie(top_5_tracks['streams'], labels=top_5_tracks['track_name'], autopct='%1.1f%%', colors=['skyblue', 'orange', 'green', 'red', 'purple'])
plt.title('Top 5 Tracks by Streams')
plt.show()
```
![image](https://github.com/user-attachments/assets/7a7a3686-cba3-4aa0-a149-e8cfce0228a1)

💡 The top 5 tracks by streams are:
  <li><strong>Blinding Lights</strong> – 3.70 billion streams</li>
  <li><strong>Shape of You</strong> – 3.56 billion streams</li>
  <li><strong>Someone You Loved</strong> – 2.89 billion streams</li>
  <li><strong>Dance Monkey</strong> – 2.86 billion streams</li>
  <li><strong>Sunflower</strong> – 2.81 billion streams</li>

These tracks showcase the dominance of pop music, with <strong>Blinding Lights</strong> and <strong>Shape of You</strong> at the top. The presence of <strong>Sunflower</strong> highlights the influence of movie soundtracks on streaming trends.</p>

- **Who are the top 5 most frequent artists based on the number of tracks in the dataset?**
```python
#Count number of tracks per artist and extract top 5
top_5 = df['artist(s)_name'].value_counts().head()

#Plot horizontal bar chart
plt.figure(figsize=(9, 6)) 

#Horizontal bar plot with custom colors
top_5.plot(kind='barh', color=['#FFC0CB', '#FFB6C1', '#FF69B4', '#FF1493', '#DB7093'])  
plt.title('Top 5 Artists by Number of Tracks')  
plt.xlabel('Number of Tracks')  
plt.ylabel('Artist(s) Name') 
plt.show() 
```
![image](https://github.com/user-attachments/assets/74a86e3e-95fb-4a70-8de8-2f2ec6faaaf1)

💡 The top 5 artists in the dataset, based on the number of tracks, are:
  <li><strong>Taylor Swift</strong> – 34 tracks</li>
  <li><strong>The Weeknd</strong> – 22 tracks</li>
  <li><strong>Bad Bunny</strong> – 19 tracks</li>
  <li><strong>SZA</strong> – 19 tracks</li>
  <li><strong>Harry Styles</strong> – 17 tracks</li>
</ul>

<p>These artists dominate the dataset, with <strong>Taylor Swift</strong> leading by a significant margin. The others, including <strong>The Weeknd</strong>, <strong>Bad Bunny</strong>, and <strong>SZA</strong>, also show strong representation, reflecting their continued popularity in the music industry.</p>

---

### 4. Temporal Trends 📅
- **Analyze the trends in the number of tracks released over time. Plot the number of tracks released per year.**  
```python
#Example of counting tracks per year (ensure this is already done)
number_track_peryear = df.groupby('released_year')['track_name'].count()

#Create a figure for the plot
plt.figure(figsize=(10, 6))

#Create a line plot with grid for better visualization
number_track_peryear.plot(kind='line', color='green', marker='*')  

#Title and labels
plt.title('Number of Tracks Released Per Year')
plt.xlabel('Released Year')
plt.ylabel('Number of Tracks')

#Display grid for better readability
plt.grid(True)

#Show the plot
plt.show()
```
![image](https://github.com/user-attachments/assets/614c2ed2-d818-48d6-8523-e849b8fc0308)

💡 The graph shows the number of tracks released each year. Notable trends include a significant increase in releases starting from the 2010s, with a sharp rise in 2021 and 2022, marking a peak in the number of tracks. The years 2021 and 2022 saw particularly high release counts, with 119 and 402 tracks, respectively. Before the 2000s, the number of releases per year was relatively low, with a gradual increase over time. The trend indicates a growing music production volume, especially in recent years, likely driven by digital platforms and increased access to music distribution.

- **Does the number of tracks released per month follow any noticeable patterns? Which month sees the most releases?**  
```python
#Counting and sorting the number of tracks released per month from the dataset
track_month = df['released_month'].value_counts().sort_index()

#Dictionary for months
months = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 
          8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}

#Mapping the month number to the corresponding name
track_month = track_month.rename(months)

#Plotting the number of tracks released per month
plt.figure(figsize=(10, 6))

#Plot with 'hue' parameter and a custom color palette
sns.barplot(x=track_month.index, y=track_month.values, hue=track_month.index, palette='coolwarm', legend=False)
plt.title("Number of Tracks Released Per Month")
plt.xlabel("Month")
plt.ylabel("Number of Tracks")
plt.xticks(rotation=45)
plt.show()
```

![image](https://github.com/user-attachments/assets/ad3256a4-7454-487a-8fda-5f2c382d887a)


💡 The dataset reveals the distribution of tracks released throughout the year, with notable peaks in January, May, and December, each having over 100 releases. January leads with 134 tracks, followed by May with 128, and December with 75. The months of August, September, and July have the lowest release counts, ranging between 46 and 62 tracks. Overall, there is a fairly consistent number of releases throughout the year, with early and late months seeing higher activity, likely due to seasonal music releases and end-of-year campaigns.

---

### 5. Genre and Music Characteristics 🎶
- **Examine the correlation between streams and musical attributes like bpm, danceability_%, and energy_%. Which attributes seem to influence streams the most?**
```python
#Extracting specific columns
attributes = df[['streams', 'bpm', 'danceability_%', 'energy_%']]

#Correlation matrix
correlation_matrix = attributes.corr()

#Plotting heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Between Streams and Musical Attributes')
plt.show()
```
![image](https://github.com/user-attachments/assets/8f7facbf-3153-4f02-acd2-cc72709a6070)

💡 The correlation matrix reveals weak relationships between the variables. Streams show a very minimal negative correlation with BPM (-0.0024), danceability (-0.1055), and energy (-0.0261), indicating that streams are not strongly influenced by these musical features. Danceability and energy have a moderate positive correlation (0.1981), suggesting that tracks with higher energy levels tend to be more danceable. Additionally, BPM has a slight negative correlation with danceability (-0.1471), indicating that faster tempos may not always correspond to higher danceability. Overall, these correlations suggest that other factors may be more influential in determining the popularity (streams) of a track.

- **Is there a correlation between danceability_% and energy_%? How about valence_% and acousticness_%?**  
```python
#Select relevant columns
attributes = df[['danceability_%', 'energy_%', 'valence_%', 'acousticness_%']]

#Compute the correlation matrix
correlation_matrix = attributes.corr()

#Display the correlation matrix
print(correlation_matrix)
```
![image](https://github.com/user-attachments/assets/6bafed72-a207-44af-bc2f-d1219a933292)

💡 The correlation matrix indicates several interesting relationships between musical attributes. Danceability has a moderate positive correlation with energy (0.1981), suggesting that more energetic tracks tend to be more danceable. Valence (musical positivity) shows a stronger positive correlation with danceability (0.4085), indicating that happier, more positive tracks are often more danceable. However, acousticness has a negative correlation with both energy (-0.5773) and danceability (-0.2362), suggesting that more acoustic, less energetic tracks tend to be less danceable. These findings highlight how different musical features interact, with energy and danceability being positively linked, while acousticness plays a contrasting role.

---

### 6. Platform Popularity 📱
- **How do the numbers of tracks in `spotify_playlists`, `spotify_charts`, and `apple_playlists` compare? Which platform seems to favor the most popular tracks?**  
```python
#Convert columns to numeric and sum the number of tracks per platform
numtrack_playlists = df[['in_spotify_playlists', 'in_deezer_playlists', 'in_apple_playlists']].apply(pd.to_numeric, errors='coerce').sum()

#Prepare the data for the violin plot 
df_melted = df[['in_spotify_playlists', 'in_deezer_playlists', 'in_apple_playlists']].apply(pd.to_numeric, errors='coerce')
df_melted = df_melted.melt(var_name='Platform', value_name='Number of Tracks')

#Create a violin plot to visualize the distribution of tracks by platform
plt.figure(figsize=(8, 6))
sns.violinplot(x='Platform', y='Number of Tracks', data=df_melted, hue='Platform', palette=["#28a745", "#34d058", "#7bed7f"], legend=False)
plt.title("Distribution of Tracks Across Platforms")
plt.xlabel("Platform")
plt.ylabel("Number of Tracks")
plt.show()
```
![image](https://github.com/user-attachments/assets/f8647d09-7446-4a58-b933-b01081b5ceda)

💡 The data reveals the following distribution of tracks across different music platforms:</p>
<ul>
  <li><strong>Spotify Playlists</strong>: 4,955,719 tracks, indicating that Spotify is the dominant platform for the tracks in this dataset.</li>
  <li><strong>Deezer Playlists</strong>: 95,913 tracks, significantly fewer than Spotify, suggesting a smaller share of tracks featured on Deezer.</li>
  <li><strong>Apple Playlists</strong>: 64,625 tracks, indicating a smaller presence in playlists compared to both Spotify and Deezer.</li>
</ul>
This shows that Spotify is the most prominent platform for tracks in this dataset, while Deezer and Apple Music have much fewer tracks featured in their playlists.</p>

---

### 7. Advanced Analysis 🔎
- **Based on the streams data, can you identify any patterns among tracks with the same key or mode (Major vs. Minor)?**  
```python
# Calculate average streams per mode and key
stream_data = df.groupby(['mode', 'key'])['streams'].mean().reset_index()

# Sort by streams in descending order
sorted_stream = stream_data.sort_values('streams', ascending=False)

# Print the sorted average streams
print("Average streams by mode and key\n", sorted_stream)

# Create a horizontal bar plot with pink color palette
plt.figure(figsize=(8, 6))
sns.barplot(x='streams', y='key', hue='mode', data=sorted_stream, palette='pink')
plt.xlabel("Average Streams")
plt.ylabel("Key")
plt.title("Average Streams by Mode and Key")
plt.legend(title='Mode')
plt.show()
```
![image](https://github.com/user-attachments/assets/232ca9a1-e92e-4cc0-886f-f508f8ceebb0)

![image](https://github.com/user-attachments/assets/3ccf0ccc-c62a-476e-895e-a0de1cc906da)


💡 Tracks in Major mode have higher average streams than those in Minor mode. For instance, the highest streaming track, in E Major, has around 760 million streams, while Minor mode tracks peak at about 595 million. Major mode tracks like those in D# Major and C# Major also outperform their Minor mode counterparts. This indicates that Major mode tracks tend to be more popular, likely due to their more upbeat tonal qualities.

- **Do certain genres or artists consistently appear in more playlists or charts? Perform an analysis to compare the most frequently appearing artists in playlists or charts.**
```python
#Convert selected columns to numeric, coerce errors to NaN
columns_to_convert = ['in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists', 
                      'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts']
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

#Group by artist and sum appearances across platforms
artist_appearances = df.groupby('artist(s)_name')[columns_to_convert].sum()
artist_appearances['Total_appearances'] = artist_appearances.sum(axis=1)

#Sort by total appearances and get the top 5 artists
top_artists = artist_appearances.sort_values('Total_appearances', ascending=False).head()

#Plot as pie chart
plt.figure(figsize=(8, 8))
plt.pie(top_artists['Total_appearances'], labels=top_artists.index, autopct='%1.1f%%', colors=sns.color_palette("viridis", len(top_artists)))
plt.title("Top 5 Artists Across All Platforms")
plt.show()
```
![image](https://github.com/user-attachments/assets/dad70320-39b9-421d-a228-6c6c52a1b70f)

💡 The top 5 artists with the highest total appearances across all platforms are:</p>
<ul>
  <li><strong>The Weeknd</strong>: 150,273 appearances</li>
  <li><strong>Taylor Swift</strong>: 140,755 appearances</li>
  <li><strong>Ed Sheeran</strong>: 133,407 appearances</li>
  <li><strong>Harry Styles</strong>: 115,338 appearances</li>
  <li><strong>Eminem</strong>: 88,523 appearances</li>
</ul>
These artists lead in terms of the number of tracks included across various music platforms, indicating their widespread popularity and presence in playlists, charts, and other music-related content.</p>

---

# About the author

The author is currently a second-year student pursuing a Bachelor's degree in Electronics Engineering. They have a strong passion for technology and are dedicated to applying their knowledge to real-world projects. In addition to their academic pursuits, the author continually seeks to expand their skill set. They are excited about the opportunities in the electronics field and aim to contribute to innovative solutions in the industry.

---

# Acknowledgement

I would like to express my gratitude to the following professors for their continuous guidance and support during this experiment:

👩‍🏫 Engr. Ma. Madecheen S. Pangaliman, MSc <br/>
👨‍🏫 Engr. Nico John Leo S. Lobos, MSc, ECE, ECT

---

# References:</h3>
<ul>
  <li>ECE2112 Lesson Materials</li>
  <li>Python Software Foundation. (n.d.). <i>Python 3 reference documentation</i>. Retrieved from <a href="https://docs.python.org/3/reference/index.html" target="_blank">https://docs.python.org/3/reference/index.html</a></li>
  <li>Waskom, M. (2020). <i>Seaborn: statistical data visualization</i>. Retrieved from <a href="https://seaborn.pydata.org/index.html" target="_blank">https://seaborn.pydata.org/index.html</a></li>
  <li>Python Programming. (2019, September 18). <i>Python programming tutorials: Python crash course for beginners</i> [Video]. YouTube. Retrieved from <a href="https://www.youtube.com/watch?v=m1FEHPz90oI" target="_blank">https://www.youtube.com/watch?v=m1FEHPz90oI</a></li>
  <li>Hunter, J. D. (2007). <i>Matplotlib: A 2D graphics environment</i>. Computing in Science & Engineering, 9(3), 90-95. Retrieved from <a href="https://matplotlib.org/stable/api/index.html" target="_blank">https://matplotlib.org/stable/api/index.html</a></li>
</ul>

---

### Feel free to let the author know if there is anything they can do to improve the code. Thank you!



