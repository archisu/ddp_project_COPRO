import sqlite3
conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()


#Database
#Table
#Field/Columns
#DataType

def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS taskstable(task TEXT, task_status TEXT, task_due_date DATE, task_cost REAL,cost_currency TEXT)')

def add_data(task,task_status,task_due_date,task_cost,cost_currency):
    c.execute('INSERT INTO taskstable(task,task_status,task_due_date,task_cost,cost_currency) VALUES(?,?,?,?,?)' , (task,task_status,task_due_date,task_cost,cost_currency))
    conn.commit()

def view_all_data():
    c.execute('SELECT * FROM taskstable')
    data=c.fetchall()
    return data

def view_unique_tasks():
    c.execute('SELECT DISTINCT task FROM taskstable')
    data=c.fetchall()
    return data


def get_task(task):
    c.execute('SELECT * FROM taskstable WHERE task = "{}"'.format(task))
    data=c.fetchall()
    return data

def edit_task_data(new_task,new_task_status,new_task_date,new_task_cost,task,task_status,task_due_date,task_cost):
    c.execute('UPDATE taskstable SET task =?,task_status=?,task_due_date=?,task_cost=? WHERE task=? and task_status=? and task_due_date=? and task_cost=?', (new_task,new_task_status,new_task_date,new_task_cost,task,task_status,task_due_date,task_cost))
    conn.commit()
    data = c.fetchall()
    return data

def delete_data(task):
    c.execute('DELETE FROM taskstable WHERE task="{}"'.format(task))
    conn.commit()