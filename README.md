# COVID-19_research
Ever since the outbreak of the novel Coronavirus (SARS-CoV-2) in late November 2019, the virus has spread to 199 and territories across the globe, has infected more than 550K individuals and caused more than 26,000 deaths so far. This by far has been the worst outbreak of our lifetime and will forever be remembered as a defining moment for humankind. The scientific community across borders had joined hands in pushing research about this novel Coronavirus with the ultimate goal of trying to understand how the virus spreads, what is the immune response to the virus, what are the clinical manifestations of the disease, how SARS-CoV-2 is different from previous Coronaviruses etc., all with the aim of developing ways to stop the spread of this disease. As a result, we now have access to over 37,000 Coronaviruses related research articles containing huge amounts of data just waiting to be mined. This is my small contribution to the effort.

As a researcher, whenever I begin a project the first thing I do is read previous work on the topic and get a good understanding of the current state of the work. This is usually not a huge problem and I take my time developing my knowledgebase. In case of novel coronavirus-related research, because of the urgency of the matter we need to hit the ground running and therefore need to streamline the process of selecting research articles that we should read first.

## PART1:
### Paper sorter
This code parses through all the research material available (peer-reviewed and on Arxiv) in the databases below and creates a pandas dataframe containing the title, abstract and the body text of these papers. We then enter search keywords and the code will identify papers that mention the keyword in the text. We then generate a graph depicting the number of manuscripts that contain the keyword, sort of an indication of the "popularity" of the keyword.

Next, we create connection dataframes which contain papers which mention all combinations of "two search keywords" together and save these as separate files and generate a network graph showing these connections. Finally, we can select a keyword of interest to get all the papers that mention the selected keyword and other keywords together. These plots, and tables together will help identify research articles that contain keywords of interest, and therefore can save time and help organize our reading efforts.

![Mentions plot](https://github.com/pranaydogra/COVID-19_research/blob/master/mentions_plot.png)
![SARS-CoV-2 connectivity](https://github.com/pranaydogra/COVID-19_research/blob/master/sarscov2_small.png)

This selected dataset can then be used for further computational analysis pipelines.

## PART2:
### Term specific paper clustering
Once we have created the term specific data frames i.e. grouped all the papers based on whether they mention a keyword or not usign the previous script, the next step is to analyze the content of the papers. There are several ways to go about doing this, presented here is my approach. I first vectorize the body text from all the articles and then use that to cluster the papers based on content. Then I perform dimensionality reduction via UMAP and TSNE. Finally I project the data in 2D as interactive Bokeh plots to identify groups of papers, which hopefully have the same broad message. Lastly I have tried to identfy the most common words in the abstracts of each cluster to help one select the group of papers they want to focus on. There is also a step that saves papers in each cluster as individual .csv files containing the paper id and the title.

![Single cell lit cluster](https://github.com/pranaydogra/COVID-19_research/blob/master/single_cell_normal.png)
![Most common words](https://github.com/pranaydogra/COVID-19_research/blob/master/Most%20common%20words%20in%20louvain%20cluster%2011_lem_lexicon.png)

### Databases
Download data from: https://pages.semanticscholar.org/coronavirus-research

**NOTE:** Unzip all the files from the different sources into a single folder and provide the path to this folder when asked for "read_dir"

**To learn more about the Kaggle COVID-19 competition please visit:**
https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge
