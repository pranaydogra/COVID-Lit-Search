# COVID-19_research
Ever since the outbreak of the novel Coronavirus (SARS-CoV-2) in late November 2019, the virus has spread to 199 and territories across the globe. The scientific community had joined hands in pushing resarch about this novel Coronavirus with the to understand how the virus spreads, what is the immune respose to the virus, what are the cinical manifestations of the disease, how SATS-CoV-2 different from previous Coronairuses etc. all with the aim of developing ways to stop the spread of this disease. This is my small contribution to the effort.

As a researcher, whenever I begin a project the first thing I do is read previous work on the topic and get a good understanding of the current state of the work. This is uaually not a huge problem and I take my time developing my knowledgebase. In case of novel coronavirus-related resarch, becasue of the urgency of the matter we neeed to hit the ground running and therefore need to streamline the process of selecting resaerch articles that we should read first. This code parses through all the research material available (peer-reviewed and on Arxiv) in the databases below and creates a pandas dataframe containing the title, abstract and the body text of these papers. The code outputs a graph depicting the number of manuscripts that contain the keyword, an indication of the "popularity" of the keyword.

One can then enter search keywords and the code will identify papers that mention the keyword in the text. We then create connection dataframes which contains papers which mention all combinations of two search keywords together, save these as separate files and generate a network graph showing these connections. Finally we can select a keyword get all the papers that mention the selected keyword and other keywords together. These plots together will help identify research articles that contain keywords of interest, and thefore save time and help organize our reading efforts.

This selected dataset can then be used for further computational analysis pipelines.

Databses link: https://pages.semanticscholar.org/coronavirus-research

Unzip all the files from the different sources into a single folder and provide the path to it when asked for "read_dir"\

To read more about the Kaggle COVID-19 competition please visit:
https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge
