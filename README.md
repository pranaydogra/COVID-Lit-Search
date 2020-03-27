# COVID-19_research
Ever since the outbreak of the novel Coronavirus (SARS-CoV-2) in late November 2019, the virus has spread to 199 and territories across the globe, has infected more than 550K individuals and caused more than 26,000 deaths so far. This by far has been the worst outbreak of our lifetime, and will forever be remembered as a defining moment for humankind. The scientific community across borders had joined hands in pushing resarch about this novel Coronavirus with the ultimate goal of trying to understand how the virus spreads, what is the immune respose to the virus, what are the cinical manifestations of the disease, how SATS-CoV-2 different from previous Coronairuses etc. all with the aim of developing ways to stop the spread of this disease. As as result, we now have access to over 29,000 Coronaviruses related reasearch articles containing huge amounts of data just waiting to be mined. This is my small contribution to the effort.

As a researcher, whenever I begin a project the first thing I do is read previous work on the topic and get a good understanding of the current state of the work. This is uaually not a huge problem and I take my time developing my knowledgebase. In case of novel coronavirus-related resarch, becasue of the urgency of the matter we neeed to hit the ground running and therefore need to streamline the process of selecting resaerch articles that we should read first.

This code parses through all the research material available (peer-reviewed and on Arxiv) in the databases below and creates a pandas dataframe containing the title, abstract and the body text of these papers. We then enter search keywords and the code will identify papers that mention the keyword in the text. We then generate a graph depicting the number of manuscripts that contain the keyword, sort of an indication of the "popularity" of the keyword.

Next we create connection dataframes which contain papers which mention all combinations of "two search keywords" together and save these as separate files and generate a network graph showing these connections. Finally we can select a keyword of interest to get all the papers that mention the selected keyword and other keywords together. These plots, and tables together will help identify research articles that contain keywords of interest, and thefore can save time and help organize our reading efforts.

This selected dataset can then be used for further computational analysis pipelines.

Databses link: https://pages.semanticscholar.org/coronavirus-research

NOTE: Unzip all the files from the different sources into a single folder and provide the path to this folder when asked for "read_dir"

To learn more about the Kaggle COVID-19 competition please visit:
https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

