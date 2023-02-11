# Co-Pro Project Management Software
# Doğa Su Kıralioğlu, Ahmet Evyapan, and Serhan Kodaman
# February 2023

# Import necessary libraries

from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
from lib2to3.pytree import convert
import pickle
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit_authenticator as stauth
from db_fxn import create_table, add_data, view_all_data, get_task, view_unique_tasks, edit_task_data,delete_data
from  PIL import Image
import io 
from st_aggrid import AgGrid
import sklearn.preprocessing as pre
import copy
import sklearn.ensemble as ens
import numpy as np
import datetime

# Set random seed
np.random.seed = 11

# Initialize RandomForestRegressor model with random state = 11
model = ens.RandomForestRegressor(random_state=11)

# Read the style.css file and add its contents to the report as a stylesheet
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


#User Authentication
names = ['Serhan Kodaman', 'Ahmet Evyapan']
usernames = ['admin', 'user']


# Flag to check if the current user is admin
admin_flag = False


#Load hashed passwords
file_path = Path(__file__).parent / 'hashed_pw.pkl'
with file_path.open('rb') as file:
    hashed_passwords = pickle.load(file)

# Initialize the authentication module
authenticator = stauth.Authenticate(names,usernames,hashed_passwords, 'project PRO','abcdef',cookie_expiry_days=365)
name,authentication_status,username = authenticator.login('Login','main')

# Check if the user is admin
if username == "admin":
    admin_flag=True

if authentication_status == False:
    st.error('Username/password is incorrect')
if authentication_status == None:
    st.warning('Please enter your username and password')

# Part1: Website and Functionality

# Main function only executes if the user is authenticated
if authentication_status:


        def main():
            st.title('CO-PRO')

            # List of menu options for the user
            menu = ['Create Task', 'Update', 'Progress Track', 'Delete','Upload Your Project','Predict Finish Dates']
            authenticator.logout('Logout','sidebar')
            st.sidebar.title(f'Welcome {name}')
            choice = st.sidebar.selectbox('Menu',menu)


            create_table()
            if choice == 'Create Task':
                st.subheader('Add Tasks')
                #Layout
                col1,col2 = st.columns(2)
                with col1:
                    task = st.text_area('Task To Do')
                    task_cost = st.text_area('Task Cost:')
                    cost_currency = st.selectbox('Currency',['Euro','USD'])
                with col2:
                    task_status = st.selectbox('Status',['Not Started','In Progress','Completed'])
                    task_due_date = st.date_input('Due Date')
                
                
                if st.button('Add Task'):
                    add_data(task,task_status,task_due_date,task_cost,cost_currency)
                    st.success('Successfully Added Task: {}'.format(task))


            elif choice == 'Progress Track':
                st.subheader('View Tasks')
                result = view_all_data()
                st.write(result)
                df = pd.DataFrame(result,columns=['Task','Status','Due Date','Task Cost','Cost Currency'])
                with st.expander('View All Data'):
                    st.dataframe(df)
                with st.expander('Task Status'):
                    task_df = df['Status'].value_counts().to_frame()
                    task_df = task_df.reset_index()
                    st.dataframe(task_df)

                    p1 = px.pie(task_df, names = 'index', values ='Status' )
                    st.plotly_chart(p1)
                
            elif choice == 'Update':
                # Checking if the user has admin permission
                if admin_flag:
                    st.subheader('Edit/Update Tasks')
                    result = view_all_data()
                    # Converting the result into a pandas dataframe
                    df = pd.DataFrame(result,columns=['Task','Status','Due Date','Task Cost','Cost Currency'])
                    with st.expander('Current Tasks'):
                        st.dataframe(df)
                    
                        list_of_task = [i[0] for i in view_unique_tasks()]
                    

                        selected_task = st.selectbox('Task to Edit', list_of_task)
                        selected_result = get_task(selected_task)
                        st.write(selected_result)

                        if selected_result:
                            # Task details
                            task = selected_result[0][0]
                            task_status = selected_result[0][1]
                            task_due_date = selected_result[0][2]
                            task_cost = selected_result[0][3]

                            # Splitting the interface into two columns
                            col1,col2 = st.columns(2)
                            # First column for entering the new task details
                            with col1:
                                new_task = st.text_area('Task To Do',task)
                                new_task_cost = st.text_area('task cost:')
                                cost_currency = st.selectbox('Currency',['Euro','USD'])
                            # Second column for entering the new task status and due date
                            with col2:
                                new_task_status = st.selectbox(task_status,['Not Started','In Progress','Completed'])
                                new_task_due_date = st.date_input(task_due_date)
                                
                    
                        if st.button('Update Task'):
                            edit_task_data(new_task,new_task_status,new_task_due_date,new_task_cost,task,task_status,task_due_date,task_cost)
                            st.success('Successfully Updated Task:{}'.format(task,new_task))
                # If the user doesn't have admin permission
                else:
                    st.write("Permisson Denied")

                result2 = view_all_data()
                df2 = pd.DataFrame(result2,columns=['Task','Status','Due Date','Task Cost','Task Currency'])
                with st.expander('Updated Data'):
                    st.dataframe(df2)
                    



            elif choice == 'Delete':
                # Check if the user is an admin
                if admin_flag:
                    st.subheader('Delete Task')
                    result=view_all_data()
                    #Convert data into a pandas dataframe
                    df=pd.DataFrame(result,columns=['Task','Status','Due Date','Task Cost','Task Currency'])
                    with st.expander('Current Data'):
                        st.dataframe(df)
                    list_of_task = [i[0] for i in view_unique_tasks()]
                    

                    selected_task = st.selectbox('Task to Delete', list_of_task)
                    st.warning('Do you want to delete {}?'.format(selected_task))
                    #Button to trigger the delete operation
                    if st.button('Delete Task'):
                        delete_data(selected_task)
                        st.success('Task has been successfully deleted')
                else:
                    st.write("Permisson Denied")
                
# Part2: File Uploads and Gantt Chart

            elif choice == 'Upload Your Project':
                
                st.subheader('Step 1: Download the project plan template')
                image = Image.open(r"template.png") #Template screenshot provided as an example
                st.image(image)
                #Allow users to download the template
                @st.cache
                def convert_df(df):
                    return df.to_csv().encode('utf-8')
                df=pd.read_csv(r"template.csv")
                csv = convert_df(df)
                st.download_button(
                    label="Download Template",
                    data=csv,
                    file_name='project_template.csv',
                    mime='text/csv',
                )
                #Add a file uploader to allow users to upload their project plan file
                st.subheader('Step 2: Upload your project plan file')

                uploaded_file = st.file_uploader("Fill out the project plan template and upload your file here. After you upload the file, you can edit your project plan within the app.", type=['csv'])
                if uploaded_file is not None:
                    Tasks=pd.read_csv(uploaded_file)
                    Tasks['Start'] = Tasks['Start'].astype('datetime64')
                    Tasks['Finish'] = Tasks['Finish'].astype('datetime64')
                    
                    grid_response = AgGrid(
                        Tasks,
                        editable=True, 
                        height=500, 
                        width='100%',
                        )

                    updated = grid_response['data']
                    df = pd.DataFrame(updated) 
                    
                else:
                    st.warning('You need to upload a csv file.') 
                
                st.subheader('Step 3: Generate Gantt chart')
                
                Options = st.selectbox("View Gantt Chart by:", ['Team','Completion Pct','Category of the Task'],index=0)
                if st.button('Generate Gantt Chart'): 
                    fig = px.timeline(
                                    df, 
                                    x_start="Start", 
                                    x_end="Finish", 
                                    y="Task",
                                    color=Options,
                                    hover_name="Category of the Task"
                                    )

                    fig.update_yaxes(autorange="reversed")                
                    
                    fig.update_layout(
                                    title='Project Plan Gantt Chart',
                                    hoverlabel_bgcolor='#DAEEED',   #Change the hover tooltip background color to a universal light blue color. If not specified, the background color will vary by team or completion pct, depending on what view the user chooses
                                    bargap=0.05,
                                    height=750,              
                                    xaxis_title="", 
                                    yaxis_title="",                   
                                    title_x=0.5,                    #Make title centered                     
                                    xaxis=dict(
                                            tickfont_size=7,
                                            tickangle = 270,
                                            rangeslider_visible=True,
                                            side ="top",            #Place the tick labels on the top of the chart
                                            showgrid = True,
                                            zeroline = True,
                                            showline = True,
                                            showticklabels = True,
                                            tickformat="%x\n",      
                                            )
                                )
                    
                    fig.update_xaxes(tickangle=0, tickfont=dict(family='Rockwell', color='blue', size=10))

                    st.plotly_chart(fig, use_container_width=True)  #Display the plotly chart in Streamlit

                    st.subheader('Export Gantt chart to HTML') #Allow users to export the Plotly chart to HTML
                    buffer = io.StringIO()
                    fig.write_html(buffer, include_plotlyjs='cdn')
                    html_bytes = buffer.getvalue().encode()
                    st.download_button(
                        label='Export to HTML',
                        data=html_bytes,
                        file_name='Gantt.html',
                        mime='text/html'
                    ) 
                else:
                    st.write('---') 

# Part3: Machine Learning

            else: 
                if admin_flag:

                    # User selects a csv file
                    uploaded_file = st.file_uploader('Choose a file', type = ['csv'])
                    if uploaded_file is not None:
                        # Read the csv file into a dataframe
                        df_raw = pd.read_csv(uploaded_file)
                    
                        ### --------Preprocessing--------

                        df = df_raw.drop(columns=["Unnamed: 9"])
                        df = df.rename(columns={"Category of the Task":"Task_Category", "Completion Pct":"Completion_Pct", "Current Delay":"Current_Delay", 
                        "%100 Completed Delay":"Completed_Delay",'Dependent?':'Dependent'})

                        for col in ["Completion_Pct", "Current_Delay", "Completed_Delay"]:
                            df[col] = df[col].str.rstrip("%")

                        pd.to_datetime(df["Finish"], dayfirst=True)

                         # Convert the "Start" and "Finish" columns to datetime format and store the difference as "Task_Duration"

                        df["Start"] = pd.to_datetime(df["Start"], dayfirst=True)
                        df["Finish"] = pd.to_datetime(df["Finish"], dayfirst=True)
                        df["Task_Duration"] = (df.Finish - df.Start)
                        df["Task_Duration"] = df["Task_Duration"].astype('timedelta64[D]').astype(int)
                        df["Start_Month"] = df.Start.dt.month
                        df["Finish_Month"] = df.Finish.dt.month

                        df_map = df.copy().loc[:, ["Task", "Start", "Finish" ]]
                        df_map.head()

                        df = df.drop(columns=["Start", "Task"])

                        ordinal_encoder = pre.OrdinalEncoder()
                        df[["Task_Category", "Team"]] = ordinal_encoder.fit_transform(df[["Task_Category", "Team"]]).astype(int)

                        object_cols = df.select_dtypes("object").columns
                        df[object_cols] = df[object_cols].apply(pd.to_numeric, downcast="integer")

                        # Split the data into training and prediction sets

                        df_train = df.loc[df.Completion_Pct == 100] #.reset_index(drop=True)
                        train_indexes = df_train.index

                        df_pred = df.loc[df.Completion_Pct < 100] #.reset_index(drop=True)
                        pred_indexes = df_pred.index
                        finish_dates = copy.deepcopy(df_pred.Finish)

                        ### --------train/pred data split --------

                        X_train = df_train.drop(columns=["Completed_Delay", "Finish"])
                        y_train = df_train.Completed_Delay
                        X_pred = df_pred.drop(columns=["Completed_Delay", "Finish"])

                        ### --------Model Fit & Predict --------
                        model.fit(X=X_train, y=y_train)
                        predicted_delays = model.predict(X_pred)
                        X_pred["Completed_Delay"] = predicted_delays

                        ### --------Postprocessing --------
                        delay_in_days = np.ceil((X_pred.Completed_Delay / 100) * (X_pred.Task_Duration))
                        delay_in_days = delay_in_days.apply( datetime.timedelta )
                        predicted_finish_dates = finish_dates + delay_in_days

                        X_pred["Predicted_Finish_Dates"] = predicted_finish_dates
                        X_pred = X_pred.rename(columns={
                            "Task_Category": "Category of the Task", "Completion_Pct": "Completion Pct",
                            "Current_Delay": "Current Delay", "Completed_Delay": "%100 Completed Delay", "Predicted_Finish_Dates": "Predicted Finish Dates"
                        })

                        X_pred[["Task", "Category of the Task", "Start", "Finish", "Team"]] = df_raw.loc[X_pred.index, ["Task", "Category of the Task", "Start", "Finish", "Team"]]

                        X_pred = X_pred.drop(columns=["Start_Month", "Finish_Month"])

                        first_cols = ["Task", "Start", "Finish", "Predicted Finish Dates"]
                        cols = X_pred.columns.tolist()
                        last_cols = [col for col in cols if col not in first_cols]
                        all_cols = first_cols + last_cols
                        X_pred = X_pred[all_cols]
                        X_pred = X_pred.reset_index(drop=True)
                        X_pred["Predicted Finish Dates"] = X_pred["Predicted Finish Dates"].dt.strftime('%d-%b-%y')
                        st.write("File has been processed")
                        #X_pred.to_csv('predicted_file.csv', index=False)
                        st.dataframe(X_pred)                                      
                    
                    # @st.cache
                    # def convert_df(df):
                    #     # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    #     return df.to_csv("predictions.xlsx", index=False).encode('utf-8')


                    @st.cache
                    def convert2excel(df):
                        output = BytesIO()
                        writer = pd.ExcelWriter(output, engine='xlsxwriter')
                        df.to_excel(writer, index=False, sheet_name='Sheet1')
                        # workbook = writer.book
                        # worksheet = writer.sheets['Sheet1']
                        # format1 = workbook.add_format({'num_format': '0.00'}) 
                        # worksheet.set_column('A:A', None, format1)  
                        writer.save()
                        processed_data = output.getvalue()
                        return processed_data

                    if uploaded_file is not None:
                        predictions_excel = convert2excel(X_pred)

                        download_button = st.download_button(
                            label = "Download Predictions as excel file.",
                            data = predictions_excel,
                            file_name = "predictions.xlsx",
                        )

                        if download_button:
                            st.write("Prediction csv file has been downloaded.")

                else:
                    st.write("Permission Denied")    
                    
        if __name__ == '__main__':
            main()

