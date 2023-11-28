# Paper aquaculture IoT

1. Data Preparation:
    •Gather all the papers related to IoT technologies in fisheries and aquaculture.
    •Preprocess the text data by removing stop words, stemming, and lemmatization.
    •Convert the text data into a format suitable for LDA analysis, such as a document-term matrix.
2. LDA Model Training:
•	Choose the number of topics (clusters) you want the model to identify. This may require some experimentation.
•	Train the LDA model on your preprocessed dataset.
3. Explore Topics:
•	Examine the results of the LDA model to understand the distribution of topics across your papers.
•	Identify the most significant words associated with each topic.
4. Assign Topics to Papers:
•	Assign each paper to the topic that is most dominant in its content.
•	This step helps in categorizing papers based on the identified themes.
5. Visualize Results:
•	Create visualizations to represent the results of your LDA analysis.
•	Tools like word clouds, bar charts, or network graphs can help visualize the relationships between topics and words.
6. Interpretation:
•	Analyze the results to understand the main themes emerging from your dataset.
•	Look for patterns, connections, or trends within the identified topics.
7. Refinement:
•	Refine your LDA model if needed. Adjust the number of topics or revisit the preprocessing steps based on the initial results.
8. Write-Up:
•	Document your findings and interpretations.
•	Include visualizations and key insights derived from the LDA analysis.
Tips:
•	Experiment with the Number of Topics: Try different numbers of topics to find the most meaningful and coherent grouping.
•	Iterative Process: LDA analysis is often an iterative process. Refine parameters and preprocessing as needed.
•	Validate Results: Consider manually reviewing a subset of papers to validate the accuracy of the topic assignments.
Tools:
•	Python libraries such as gensim or scikit-learn can be used for LDA analysis.
•	Visualization tools like pyLDAvis can assist in interpreting and presenting the results.



```python
pip install pycountry
```

    Collecting pycountry
      Downloading pycountry-22.3.5.tar.gz (10.1 MB)
         ---------------------------------------- 0.0/10.1 MB ? eta -:--:--
         ---------------------------------------- 0.0/10.1 MB ? eta -:--:--
         --------------------------------------- 0.0/10.1 MB 435.7 kB/s eta 0:00:24
          --------------------------------------- 0.2/10.1 MB 2.0 MB/s eta 0:00:06
         ----- ---------------------------------- 1.3/10.1 MB 7.6 MB/s eta 0:00:02
         ---------- ----------------------------- 2.7/10.1 MB 13.2 MB/s eta 0:00:01
         ----------------- ---------------------- 4.4/10.1 MB 16.7 MB/s eta 0:00:01
         ---------------------------- ----------- 7.3/10.1 MB 22.2 MB/s eta 0:00:01
         --------------------------------------  10.1/10.1 MB 28.2 MB/s eta 0:00:01
         --------------------------------------  10.1/10.1 MB 28.2 MB/s eta 0:00:01
         --------------------------------------  10.1/10.1 MB 28.2 MB/s eta 0:00:01
         --------------------------------------  10.1/10.1 MB 28.2 MB/s eta 0:00:01
         --------------------------------------  10.1/10.1 MB 28.2 MB/s eta 0:00:01
         --------------------------------------- 10.1/10.1 MB 18.5 MB/s eta 0:00:00
      Installing build dependencies: started
      Installing build dependencies: finished with status 'done'
      Getting requirements to build wheel: started
      Getting requirements to build wheel: finished with status 'done'
      Preparing metadata (pyproject.toml): started
      Preparing metadata (pyproject.toml): finished with status 'done'
    Requirement already satisfied: setuptools in c:\users\kapoorab\appdata\local\anaconda3\lib\site-packages (from pycountry) (68.0.0)
    Building wheels for collected packages: pycountry
      Building wheel for pycountry (pyproject.toml): started
      Building wheel for pycountry (pyproject.toml): finished with status 'done'
      Created wheel for pycountry: filename=pycountry-22.3.5-py2.py3-none-any.whl size=10681895 sha256=9ee0f849e2b083d9e36cd4168b2b72560e7ec2f70077ffc67fb5c18aeb3570a9
      Stored in directory: c:\users\kapoorab\appdata\local\pip\cache\wheels\cd\29\8b\617685ed7942656b36efb06ff9247dbe832e3f4f7724fffc09
    Successfully built pycountry
    Installing collected packages: pycountry
    Successfully installed pycountry-22.3.5
    Note: you may need to restart the kernel to use updated packages.



```python
pip install geopy
```

    Collecting geopy
      Obtaining dependency information for geopy from https://files.pythonhosted.org/packages/e5/15/cf2a69ade4b194aa524ac75112d5caac37414b20a3a03e6865dfe0bd1539/geopy-2.4.1-py3-none-any.whl.metadata
      Downloading geopy-2.4.1-py3-none-any.whl.metadata (6.8 kB)
    Collecting geographiclib<3,>=1.52 (from geopy)
      Downloading geographiclib-2.0-py3-none-any.whl (40 kB)
         ---------------------------------------- 0.0/40.3 kB ? eta -:--:--
         ---------- ----------------------------- 10.2/40.3 kB ? eta -:--:--
         ---------------------------- --------- 30.7/40.3 kB 435.7 kB/s eta 0:00:01
         ---------------------------- --------- 30.7/40.3 kB 435.7 kB/s eta 0:00:01
         -------------------------------------- 40.3/40.3 kB 273.5 kB/s eta 0:00:00
    Downloading geopy-2.4.1-py3-none-any.whl (125 kB)
       ---------------------------------------- 0.0/125.4 kB ? eta -:--:--
       ---------------------------------------  122.9/125.4 kB 7.5 MB/s eta 0:00:01
       ---------------------------------------- 125.4/125.4 kB 2.5 MB/s eta 0:00:00
    Installing collected packages: geographiclib, geopy
    Successfully installed geographiclib-2.0 geopy-2.4.1
    Note: you may need to restart the kernel to use updated packages.



```python
# Import necessary libraries
import metaknowledge as mk
import nltk
import networkx as nx
import pandas as pd
import pandoc
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from geopy.geocoders import Nominatim
#import pycountry
import scipy
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


```

    [nltk_data] Downloading package stopwords to /Users/rishi/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!


# Example of accessing nodes and edges in the co-citation network

# Making a network of co-citations of journals
coCiteJournals = RC.networkCoCitation(nodeType='journal', dropNonJournals=True)
print(mk.graphStats(coCiteJournals))






```python
# Making a citation network
citationsA = RC.networkCitation(nodeType='year', keyWords=['A'], directed=True)
print(mk.graphStats(citationsA))

# Making a co-author network
coAuths = RC.networkCoAuthor()
print(mk.graphStats(coAuths))





# Post-processing graphs
minWeight = 3
maxWeight = 10
processedCoCiteJournals = mk.dropEdges(coCiteJournals, minWeight, maxWeight, dropSelfLoops=True)

```

    Nodes: 8
    Edges: 9
    Isolates: 0
    Self loops: 1
    Density: 0.160714
    Transitivity: 0.142857
    Nodes: 334
    Edges: 881
    Isolates: 1
    Self loops: 0
    Density: 0.0158422
    Transitivity: 0.995147



```python
# Import necessary libraries
import metaknowledge as mk
import networkx as nx
import matplotlib.pyplot as plt
import metaknowledge.contour.plotting as mkv

# Load a RecordCollection from a file
RC = mk.RecordCollection('savedrecs (5).txt')

# Making a co-citation network
CoCitation = RC.networkCoCitation()

# Visualize the co-citation network

# Export the co-citation network
mk.writeGraph(CoCitation, "CoCitationNetwork")

# Read the exported graph back into Python
ExportedCoCitation = mk.readGraph("CoCitationNetwork_edgeList.csv", "CoCitationNetwork_nodeAttributes.csv")

# Print the graph statistics
print(mk.graphStats(ExportedCoCitation))
```

    Nodes: 3152
    Edges: 126796
    Isolates: 0
    Self loops: 10
    Density: 0.0255329
    Transitivity: 0.701685



```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Read text from a TXT file
file_path = 'savedrecs-5.txt'

with open(file_path, 'r') as file:
    sample_text = file.read()

from nltk.tokenize import sent_tokenize, word_tokenize


tokenize_sentence = sent_tokenize(sample_text)

#print (tokenize_sentence)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# define the language for stopwords removal
stopwords = set(stopwords.words("english"))
print ("""{0} stop words""".format(len(stopwords)))

tokenize_words = word_tokenize(sample_text)
filtered_sample_text = [w for w in tokenize_words if not w in stopwords]

# print ('\nOriginal Text:')
# print ('------------------\n')
# print (sample_text)
# print ('\n Filtered Text:')
# print ('------------------\n')
# print (' '.join(str(token) for token in filtered_sample_text))

```

    [nltk_data] Downloading package punkt to /Users/rishi/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to /Users/rishi/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package wordnet to /Users/rishi/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!


    179 stop words



# Stop Words Removal
Often, there are a few ubiquitous words which would appear to be of little value in helping the purpose of analysis but increases the dimensionality of feature set, are excluded from the vocabulary entirely as the part of stop words removal process. There are two considerations usually that motivate this removal.

1. Irrelevance: Allows one to analyze only on content-bearing words. Stopwords, also called empty words because they generally do not bear much meaning, introduce noise in the analysis/modeling process
2. Dimension: Removing the stopwords also allows one to reduce the tokens in documents significantly, and thereby decreasing feature dimension

**Challenges:**

Converting all characters into lowercase letters before stopwords removal process can introduce ambiguity in the text, and sometimes entirely changing the meaning of it. For example, with the expressions "US citizen" will be viewed as "us citizen" or "IT scientist" as "it scientist". Since both *us* and *it* are normally considered stop words, it would result in an inaccurate outcome. The strategy regarding the treatment of stopwords can thus be refined by identifying that "US" and "IT" are not pronouns in the above examples, through a part-of-speech tagging step.


Tokenization:

It breaks down a paragraph into individual words.
For example, "I love coding" becomes ["I", "love", "coding"].
Stemming:

It trims words to their base form.
For instance, "running" becomes "run".
It helps in simplifying words for analysis.
Lemmatization:

Similar to stemming but smarter.
It reduces words to their essential form based on their meaning.
For example, "better" becomes "good".


```python
# Import Libraries
import metaknowledge as mk
import networkx as nx
import matplotlib.pyplot as plt
import metaknowledge.contour.plotting as mkv

# Load a RecordCollection from a file
RC = mk.RecordCollection('savedrecs (5).txt')

# Co-Citation Network
CoCitation = RC.networkCoCitation()
mk.writeGraph(CoCitation, "CoCitationNetwork")
ExportedCoCitation = mk.readGraph("CoCitationNetwork_edgeList.csv", "CoCitationNetwork_nodeAttributes.csv")
print(mk.graphStats(ExportedCoCitation))

# Citation Network
Citation = RC.networkCitation()

```

    Nodes: 3152
    Edges: 126796
    Isolates: 0
    Self loops: 10
    Density: 0.0255329
    Transitivity: 0.701685



```python
import pandas as pd
# Read the CSV file into a pandas DataFrame

```

Step 4: Prepare text for LDA analysis
Next, let’s work to transform the textual data in a format that will serve as an input for training LDA model. We start by tokenizing the text and removing stopwords. Next, we convert the tokenized object into a corpus and dictionary.



```python

# Load the filtered dataset
df_filtered = pd.read_csv("20231127WOSResults115.csv")

df_filtered.columns
```




    Index(['Publication Type', 'Authors', 'Book Authors', 'Book Editors',
           'Book Group Authors', 'Author Full Names', 'Book Author Full Names',
           'Group Authors', 'Article Title', 'Source Title', 'Book Series Title',
           'Book Series Subtitle', 'Language', 'Document Type', 'Conference Title',
           'Conference Date', 'Conference Location', 'Conference Sponsor',
           'Conference Host', 'Author Keywords', 'Keywords Plus', 'Abstract',
           'Addresses', 'Affiliations', 'Reprint Addresses', 'Email Addresses',
           'Researcher Ids', 'ORCIDs', 'Funding Orgs', 'Funding Name Preferred',
           'Funding Text', 'Cited References', 'Cited Reference Count',
           'Times Cited, WoS Core', 'Times Cited, All Databases',
           '180 Day Usage Count', 'Since 2013 Usage Count', 'Publisher',
           'Publisher City', 'Publisher Address', 'ISSN', 'eISSN', 'ISBN',
           'Journal Abbreviation', 'Journal ISO Abbreviation', 'Publication Date',
           'Publication Year', 'Volume', 'Issue', 'Part Number', 'Supplement',
           'Special Issue', 'Meeting Abstract', 'Start Page', 'End Page',
           'Article Number', 'DOI', 'DOI Link', 'Book DOI', 'Early Access Date',
           'Number of Pages', 'WoS Categories', 'Web of Science Index',
           'Research Areas', 'IDS Number', 'Pubmed Id', 'Open Access Designations',
           'Highly Cited Status', 'Hot Paper Status', 'Date of Export',
           'UT (Unique WOS ID)', 'Web of Science Record'],
          dtype='object')




```python

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Load the filtered dataset
df_filtered = pd.read_csv("20231127WOSResults115.csv")

# Replace 'Publication Year' and 'Title' with the correct field names representing the publication year and title, respectively
publication_year_field = 'Publication Year'
title_field = 'Article Title'

# Filter the dataset to include only the papers published in 2021

# Filter the dataset to include only the papers from 2000 to 2023
papers_published_between_2000_and_2023 = df_filtered[(df_filtered[publication_year_field] >= 2000) & (df_filtered[publication_year_field] <= 2023)]

#papers_published_in_2021 = df_filtered[df_filtered[publication_year_field] == 2021]

# Combine the 'Title' and 'Abstract' fields into a single text column (Optional: If 'Abstract' is available in the dataset)
# papers_published_in_2021['Text'] = papers_published_in_2021[title_field] + " " + papers_published_in_2021['Abstract']

# Tokenize the text and remove stopwords
stop_words = set(stopwords.words('english'))

# Keyword analysis on the titles
keyword_occurrences = Counter()
for title in papers_published_between_2000_and_2023[title_field]:
    tokens = word_tokenize(title.lower())
    keywords = [token for token in tokens if token.isalpha() and token not in stop_words]
    keyword_occurrences.update(keywords)

# Get the top 10 most frequent keywords as major themes
major_themes = keyword_occurrences.most_common(20)

# Print the major themes
print("Major Themes in Papers Published between 2000 and 2023:")
for i, (keyword, occurrences) in enumerate(major_themes):
    print(f"{i+1}. {keyword}: Occurrences = {occurrences}")


```

    Major Themes in Papers Published between 2000 and 2023:
    1. aquaponics: Occurrences = 33
    2. aquaponic: Occurrences = 26
    3. systems: Occurrences = 21
    4. system: Occurrences = 19
    5. production: Occurrences = 12
    6. fish: Occurrences = 10
    7. water: Occurrences = 9
    8. nitrogen: Occurrences = 9
    9. plant: Occurrences = 8
    10. lettuce: Occurrences = 8
    11. hydroponic: Occurrences = 8
    12. effects: Occurrences = 7
    13. commercial: Occurrences = 7
    14. growth: Occurrences = 7
    15. sustainability: Occurrences = 6
    16. recirculating: Occurrences = 5
    17. comparison: Occurrences = 5
    18. use: Occurrences = 5
    19. effect: Occurrences = 5
    20. hydroponics: Occurrences = 5


## November 27, 2023



```python
import metaknowledge as mk

# Load the dataset of academic publications
data = mk.RecordCollection("savedrecs (5).txt")

# Create a dictionary to store publication types and their citation counts
publication_type_counts = {}

# Calculate the citation counts for each publication type in the dataset
for record in data:
    publication_type = record['PT']
    if publication_type in publication_type_counts:
        publication_type_counts[publication_type] += 1
    else:
        publication_type_counts[publication_type] = 1

# Filter the data to include only entries with 4 or more citations
data_filtered = [record for record in data if publication_type_counts[record['PT']] >= 4]

# Recalculate the citation counts for the filtered data
publication_type_counts_filtered = {}

for record in data_filtered:
    publication_type = record['PT']
    if publication_type in publication_type_counts_filtered:
        publication_type_counts_filtered[publication_type] += 1
    else:
        publication_type_counts_filtered[publication_type] = 1

# Sort the publication types based on their citation counts in descending order to get the most influential types
sorted_publication_types = sorted(publication_type_counts_filtered.items(), key=lambda x: x[1], reverse=True)

# Print the top influential publication types
top_influential_publication_types = sorted_publication_types[:20]
for i, (publication_type, citation_count) in enumerate(top_influential_publication_types):
    print(f"{i+1}. {publication_type}: Citation Count = {citation_count}")

```

    1. J: Citation Count = 82
    2. C: Citation Count = 5



```python
# Create a dictionary to store publication types and their citation counts
publication_type_counts = {}

# Calculate the citation counts for each publication type in the dataset
for record in data:
    publication_type = record['PT']
    if publication_type in publication_type_counts:
        publication_type_counts[publication_type] += 1
    else:
        publication_type_counts[publication_type] = 1

# Filter the data to include only entries with publication years between 2000 and 2022
data_filtered = [record for record in data if 2000 <= int(record['PY']) <= 2022]

# Recalculate the citation counts for the filtered data
publication_type_counts_filtered = {}

for record in data_filtered:
    publication_type = record['PT']
    if publication_type in publication_type_counts_filtered:
        publication_type_counts_filtered[publication_type] += 1
    else:
        publication_type_counts_filtered[publication_type] = 1

# Sort the publication types based on their citation counts in descending order to get the most influential types
sorted_publication_types = sorted(publication_type_counts_filtered.items(), key=lambda x: x[1], reverse=True)

# Print the top influential publication types
top_influential_publication_types = sorted_publication_types[:20]
for i, (publication_type, citation_count) in enumerate(top_influential_publication_types):
    print(f"{i+1}. {publication_type}: Citation Count = {citation_count}")

```

    1. J: Citation Count = 74
    2. C: Citation Count = 5
    3. B: Citation Count = 1



```python
# Import necessary libraries
import metaknowledge as mk
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline
import metaknowledge.contour.plotting as mkv

# Load a RecordCollection from a file
RC = mk.RecordCollection('savedrecs (5).txt')

# Create a co-citation network
coCites = RC.networkCoCitation()
print(mk.graphStats(coCites, makeString=True))

# Create a co-citation network focusing on journals
coCiteJournals = RC.networkCoCitation(nodeType='journal', dropNonJournals=True)
print(mk.graphStats(coCiteJournals))

# Visualize the co-citation journal network using spring layout
# nx.draw_spring(coCiteJournals)

# Create a citation network based on keywords
citationsA = RC.networkCitation(nodeType='year', keyWords=['aquaculture', 'technology', 'aquaponics', 'IoT'])
print(mk.graphStats(citationsA))

# Create a co-author network
coAuths = RC.networkCoAuthor()
print(mk.graphStats(coAuths))

# Post-process the co-author network
minWeight = 2
maxWeight = 10
mk.dropEdges(coAuths, minWeight, maxWeight, dropSelfLoops=True)
mk.dropNodesByDegree(coAuths, 1)

# Visualize the processed co-author network
# nx.draw_spring(coAuths)

# Export the graph to files
mk.writeGraph(coAuths, "FinalJournalCoCites")

# Read the graph back into Python
FinalJournalCoCites = mk.readGraph("FinalJournalCoCites_edgeList.csv", "FinalJournalCoCites_nodeAttributes.csv")
print(mk.graphStats(FinalJournalCoCites))

```

    Nodes: 3152
    Edges: 126796
    Isolates: 0
    Self loops: 10
    Density: 0.0255329
    Transitivity: 0.701685
    Nodes: 557
    Edges: 14699
    Isolates: 0
    Self loops: 158
    Density: 0.0949266
    Transitivity: 0.412293
    Nodes: 83
    Edges: 554
    Isolates: 0
    Self loops: 9
    Density: 0.0813988
    Transitivity: 0.157455
    Nodes: 311
    Edges: 860
    Isolates: 2
    Self loops: 0
    Density: 0.0178405
    Transitivity: 0.836871
    Nodes: 37
    Edges: 51
    Isolates: 0
    Self loops: 0
    Density: 0.0765766
    Transitivity: 0.81203


# Literature on Drones and their use in fisheries


```python
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the text that you want to analyze.
text = "This is a great book!"

# Create a ToneAnalyzer object.
analyzer = SentimentIntensityAnalyzer()

# Analyze the text.
scores = analyzer.polarity_scores(text)

# Print the results.
print("Overall sentiment:", scores['compound'])
print("Positive sentiment:", scores['pos'])
print("Negative sentiment:", scores['neg'])
print("Neutral sentiment:", scores['neu'])
```

    Overall sentiment: 0.6588
    Positive sentiment: 0.594
    Negative sentiment: 0.0
    Neutral sentiment: 0.406



```python
papers= pd.read_csv('droneLitWoS56.csv', encoding='latin1')
#Add code to process text columns together
papers.head()



```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Publication Type</th>
      <th>Authors</th>
      <th>Book Authors</th>
      <th>Book Editors</th>
      <th>Book Group Authors</th>
      <th>Author Full Names</th>
      <th>Book Author Full Names</th>
      <th>Group Authors</th>
      <th>Article Title</th>
      <th>Source Title</th>
      <th>...</th>
      <th>Web of Science Index</th>
      <th>Research Areas</th>
      <th>IDS Number</th>
      <th>Pubmed Id</th>
      <th>Open Access Designations</th>
      <th>Highly Cited Status</th>
      <th>Hot Paper Status</th>
      <th>Date of Export</th>
      <th>UT (Unique WOS ID)</th>
      <th>Web of Science Record</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>J</td>
      <td>Provost, EJ; Butcher, PA; Coleman, MA; Kelaher...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Provost, Euan J.; Butcher, Paul A.; Coleman, M...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Assessing the viability of small aerial drones...</td>
      <td>FISHERIES MANAGEMENT AND ECOLOGY</td>
      <td>...</td>
      <td>Science Citation Index Expanded (SCI-EXPANDED)</td>
      <td>Fisheries</td>
      <td>OP0OK</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023-11-27</td>
      <td>WOS:000557866800001</td>
      <td>View Full Record in Web of Science</td>
    </tr>
    <tr>
      <th>1</th>
      <td>J</td>
      <td>Kopaska, J</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Kopaska, Jeff</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Drones-A Fisheries Assessment Tool?</td>
      <td>FISHERIES</td>
      <td>...</td>
      <td>Science Citation Index Expanded (SCI-EXPANDED)</td>
      <td>Fisheries</td>
      <td>AM8HY</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023-11-27</td>
      <td>WOS:000340115400009</td>
      <td>View Full Record in Web of Science</td>
    </tr>
    <tr>
      <th>2</th>
      <td>J</td>
      <td>Provost, EJ; Butcher, PA; Coleman, MA; Bloom, ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Provost, Euan J.; Butcher, Paul A.; Coleman, M...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Aerial drone technology can assist compliance ...</td>
      <td>FISHERIES MANAGEMENT AND ECOLOGY</td>
      <td>...</td>
      <td>Science Citation Index Expanded (SCI-EXPANDED)...</td>
      <td>Fisheries</td>
      <td>ME7WU</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023-11-27</td>
      <td>WOS:000544866100008</td>
      <td>View Full Record in Web of Science</td>
    </tr>
    <tr>
      <th>3</th>
      <td>J</td>
      <td>Bloom, D; Butcher, PA; Colefax, AP; Provost, E...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Bloom, Daniel; Butcher, Paul A.; Colefax, Andr...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Drones detect illegal and derelict crab traps ...</td>
      <td>FISHERIES MANAGEMENT AND ECOLOGY</td>
      <td>...</td>
      <td>Science Citation Index Expanded (SCI-EXPANDED)...</td>
      <td>Fisheries</td>
      <td>IL0SO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023-11-27</td>
      <td>WOS:000477010400001</td>
      <td>View Full Record in Web of Science</td>
    </tr>
    <tr>
      <th>4</th>
      <td>J</td>
      <td>Winkler, AC; Butler, EC; Attwood, CG; Mann, BQ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Winkler, Alexander C.; Butler, Edward C.; Attw...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>The emergence of marine recreational drone fis...</td>
      <td>AMBIO</td>
      <td>...</td>
      <td>Science Citation Index Expanded (SCI-EXPANDED)...</td>
      <td>Engineering; Environmental Sciences &amp; Ecology</td>
      <td>YP0LB</td>
      <td>34145559.0</td>
      <td>Green Published</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023-11-27</td>
      <td>WOS:000663256000002</td>
      <td>View Full Record in Web of Science</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 72 columns</p>
</div>




```python

```
