# Projet-Cassiopee [96] - made by:

GOIS Marianna

YAMAMOTO Gabriel

XIE Zihang

# About:

A Panic Button App which can be used in emergency situations to quickly alert authorities or emergency contacts inside Telecom SudParis. The summary image of the project is below:

![alt text](https://github.com/GaiaFL/Projet-Cassiopee/blob/main/poster.jpeg?raw=true)

# Database:
Inside the Database directory, contains the Access Points of some floors of the central building of the campus in Évry - France.

# Code: 
Inside the code directory, the code is split in a panic button conceived on Android Studio (zip file Button), responsible to get user's information related to the current wifi connection, and a proxy server in Python (tcp_proxy.py) that loads the database and it contains the KNN algorithm to predict the user's location.

# Primary Results:
More than 60% of accuracy in the predicition, considering a test of 170 alert messages in 3 buildings with 4 floors.
![alt text](https://github.com/GaiaFL/Projet-Cassiopee/blob/main/Code/Graphics/confusion%20matrix.jpeg?raw=true)
