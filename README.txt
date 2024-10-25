__Stimuler la découverte de contenus pour renforcer les espaces d’information démocratiques__

This project aims to deepen the understanding of news consumption patterns by analyzing the topics and user needs of articles visited by users. The data utilized includes real articles and logs from the french Swiss public media service RTS (https://www.rts.ch/), enabling the development of a recommendation and nudging system tailored to the needs of democratic news spaces.

__Objective__
The main objective is to create a more precise recommendation system that aligns with ethical standards, reinforcing democratic information spaces. This goal is pursued by categorizing users based on their interactions with various article topics and user needs. 
For additional information on the user needs classification used, see https://smartocto.com/research/userneeds/.

__Nudging Framework__
Three types of nudging strategies are implemented, each with a foundation in ethical considerations surrounding their legitimacy and purpose.

__Technical Summary__
Data Handling:

Data from articles and user logs is uploaded, cleaned, and preprocessed.
Article data includes two primary features: Topic and User Need, as defined by RTS.

Community Detection:

Users are grouped into communities based on the frequency of (topic, user need) pairs associated with each visited article.
Given the sparsity of the dataset, classical clustering methods were not effective; instead, graph-theoretical techniques are applied.
Different community detection algorithms are evaluated using conductance and modularity metrics to determine the most meaningful user groupings.

Nudging Implementation:

After community detection, users are assigned to specific communities, and nudging strategies are applied based on these groupings.
Refer to the Demo folder for a sample nudging implementation.


The data is not included in this repository for privacy reasons. However, further insights on the methodology and analysis can be found in the accompanying paper (in progress).

