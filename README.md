# ddp_project_COPRO
This is the public repository for the Co-Pro Project Management Software by Doğa Su Kıralioğlu, Ahmet Evyapan, and Serhan Kodaman.

Co-Pro Project Management Software can be replicated and used by anyone. Co-Pro software is mainly composed of two parts: the website and the mobile application. Both of these components can be set up and used by following these steps:

### The Website

1- The website is written in Python. In order to run the code for the website, a Python version of 3.7 or later is required. 

2- Installing the necessary libraries for Python: All of the libraries that are required for running Co-Pro can be installed through the following commands:
streamlit: pip install streamlit
pandas: pip install pandas
plotly: pip install plotly
streamlit-authenticator: pip install streamlit-authenticator
pillow: pip install pillow
streamlit-aggrid: pip install streamlit-aggrid
scikit-learn: pip install scikit-learn
numpy: pip install numpy

3- Downloading the necessary files: All of the necessary files that are needed to run the Co-Pro Website can be found in the following Git-Hub repository, labeled as “Website Files”:
	https://github.com/archisu/ddp_project_COPRO

4- Understanding the code: The main Python file for Co-Pro website is composed of three parts. The first part is Website and Functionality, followed by Part 2: File Uploads and Gantt Chart, and lastly Part 3: Machine Learning. You can navigate within the code with the help of comments and make adjustments in the parts that you would like. 

5- Activating the virtual environment: Inside your terminal, activate the neccessary virtual environment.

6- Running the code: First, open a terminal. With the command “streamlit run main.py” run the main Python file. After this, you will be redirected into Co-Pro webpage. If you are not redirected, you can open the page manually by following the link that appears in the terminal after running Streamlit. 

7- Congratulations! You have successfully set up the website for Co-Pro. 

8- After the set-up: In order to make adjustments and improve Co-Pro, you can directly make the changes on the main.py code. These changes will be instantly reflected on the website while Streamlit is running. If you can not see the changes, visit the website and choose “Rerun” option from the top right side of the page.

### The Mobile Application

1- Import Required Libraries
import 'package:flutter/material.dart';
import 'package:flutter/gestures.dart';
import 'dart:ui';
import 'package:google_fonts/google_fonts.dart';
import 'package:myapp/utils.dart';

2- Create the Main Function
In Flutter, the main function is the starting point of your app. We use the runApp function to run our app, and pass the COPROApp widget as an argument.

3- Create the COPROApp Widget
The COPROApp widget is a StatelessWidget, which means its state does not change during the lifetime of the app.
We use the MaterialApp widget to create the basic structure of our app. We set the title to 'CO-PRO APP', and set the theme to ThemeData(primarySwatch: Colors.blue) so that the app background will have a blue color scheme.
The home property is set to COPROAppHomePage(title: 'CO-PRO Home Page'), which is the widget that will be displayed when the app starts.
We also set up a route in the routes property, which will allow us to navigate to the ReschedulePage when the user presses the ‘RESCHEDULE’ button on the home page.

4- Create the COPROAppHomePage Widget
The COPROAppHomePage widget is a StatefulWidget, which means its state can change during the lifetime of the app.
The title property is passed in as a parameter and will be used to display the title of the app bar on the home page.

5- Create a new class called "ReschedulePage" that extends "StatelessWidget". This class will define the contents of the Reschedule Page.
In the "build" method, return a "Scaffold"* widget that contains an "AppBar" widget and a "body" widget. The "AppBar" widget should have a title of "Reschedule Page". The "body" widget should be a "Center" widget that contains a "Text" widget with the message "This is the Reschedule Page".
*Scaffold : is a widget in the Flutter framework's Material library that provides a basic layout structure. It implements the basic material design visual layout structure.




