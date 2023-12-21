# Since the user is interested in the "window.__xrRequests" object within the script tags,
# we will find all script tags and search for the one that contains the "window.__xrRequests" object.

import os
import re
import json
import javalang
import mysql.connector
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
from bs4 import BeautifulSoup 

# %% Environment variables

host='127.0.0.1'
user='root'
password='root'
database = 'jpetstore-6.0.2'
port = '3308'
directory_path = "/home/amir/Desktop/PJ/MonoMicro/jpetstore-6-jpetstore-6.0.2"  # Replace with your directory
repo_path = '/home/amir/Desktop/PJ/MonoMicro/jpetstore-6'
xrebel_logs = '/home/amir/Desktop/PJ/MonoMicro/jpetstore-xrebel-logs.html'
xrebel_tree = '/home/amir/Desktop/PJ/MonoMicro/jpetstore-xrebel-logs.json'

# %% Define database functions for connecting,executing queries and printing results
def connect_to_database(host, user, password, database,port):
    """Create a connection to a MySQL database."""
    cnx = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port = port
    )
    return cnx

def execute_query(cnx, query):
    """Execute a SQL query and return the results as a list of tuples."""
    cursor = cnx.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    return results

def print_results(results):
    """Print the results of a SQL query."""
    for row in results:
        print(row)

def close_database_connection(cnx):
    """Close the connection to a MySQL database."""
    cnx.close()

# %% Extract conseptual coupling features from code + enums feature of structural coupling
class_id_to_name ={}

def get_all_classes():
    cnx = connect_to_database(host, user, password, database,port)
        
    query = '''
    select 
    lc.class_id,
    lc.class_name
    from list_class lc
        '''
    results = execute_query(cnx, query)
    close_database_connection(cnx)
    return results

# %% Extract conseptual coupling features from code + enums feature of structural coupling
all_classes = get_all_classes()

def parse_java_file(file_path):
    with open(file_path, 'r') as file:
        java_code = file.read()
    return javalang.parse.parse(java_code)

# find enums within the directories of the project
def find_enums(directory):
    enums_list = []
    i=1
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)
                try:
                    java_tree = parse_java_file(file_path)

                    # Extract package name
                    package_name = java_tree.package.name if java_tree.package else "default"

                    # Get all enum declarations
                    enums = [node for _, node in java_tree.filter(javalang.parser.tree.EnumDeclaration)]
                    if len(enums)>0: #Found any enum
                        enum_info = [(e.name,package_name,len(all_classes)+i) for e in enums]
                        enums_list += enum_info
                        i+=1
                    
                except Exception as e:
                    print(e)

    return enums_list

# add enums to the list of classes
enums = find_enums(directory_path)
for enum in enums:
    class_id_to_name[enum[2]] = enum[0].lower()
    all_classes.append((enum[2],enum[0].lower()))

# %% Create Class Matrix

class_names = [class_obj[1] for class_obj in all_classes]

zero_matrix = np.zeros((len(class_names), len(class_names)), dtype='object')


# Create a pandas DataFrame from the matrix
class_matrix = pd.DataFrame(zero_matrix, columns=class_names, index=class_names)

# %%

# Function to extract window.__xrRequests object from the script content
def extract_xr_requests(script):
    # Look for the window.__xrRequests pattern and capture the JSON object
# If window.__xrRequests is an array
    xr_requests_pattern = re.compile(r'window\.__xrRequests\s*=\s*(\[.*\])\s*;', re.DOTALL)
    match = xr_requests_pattern.search(script)
    if match:
        # Extract the JSON-like string and parse it into a Python object
        json_string = match.group(1)
        # We'll handle any JSON parsing exceptions that may occur
        try:
            traces= '{"logs":'+json_string +"}"
            # traces = traces.replace('"','\"')
            data = json.loads(traces)
            
            return data
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
    return None

# Read the entire content of the HTML file
with open(xrebel_logs, 'r') as file:
    html_content = file.read()

soup = BeautifulSoup(html_content, 'html.parser')


script_tags = soup.find_all('script')

# Initialize an empty list to hold our extracted data
xr_requests_data = []

# Iterate over all script tags and attempt to extract the xrRequests data
for script in script_tags:
    if script.string and "window.__xrRequests" in script.string:
        data = extract_xr_requests(script.string)
        if data:
            xr_requests_data.append(data)

xr_requests_data


def extract_table_names(sql_query):
    # Normalize the query to simplify regex matching
    query = sql_query.upper()

    # Regex patterns for different SQL clauses
    patterns = [
        r'FROM\s+([a-zA-Z0-9_]+)(?:\s+[a-zA-Z0-9_]+)?(?:,\s*([a-zA-Z0-9_]+)(?:\s+[a-zA-Z0-9_]+)?)?',  
        r'JOIN\s+([a-zA-Z0-9_]+)',                                                               
        r'UPDATE\s+([a-zA-Z0-9_]+)',                                                             
        r'INSERT\s+INTO\s+([a-zA-Z0-9_]+)'                                                       
    ]

    # Set to store unique table names
    table_names = set()

    # Apply each regex pattern and update the set of table names
    for pattern in patterns:
        for match in re.finditer(pattern, query):
            # Adding matched groups to the set
            table_names.update([m for m in match.groups() if m])

    return list(table_names)


def extract_dependecies_from_execution_trace(traces: list, queries: list, parents: list, elements: list):
    

    if(len(traces) == 0):
        return
    
    for trace in traces:
        if('traces' not in trace.keys()):
            continue

        class_name_parts = trace['className'].split('.')
        class_name = ''
        if(len(class_name_parts) > 1):
            class_name = class_name_parts[0].lower()
        else:
            class_name = trace['className'].lower()

            if('$' in class_name):
                class_name_parts = class_name.split('$')
                class_name = ''
                if(len(class_name_parts) > 1):
                    class_name = class_name_parts[0].lower()
                else:
                    class_name = class_name_parts

        if("preparedstatement" in class_name):
            last_element = elements[-1]
            for query in queries:
                tables = extract_table_names(sql_query=query['query'])
                for table in tables:
                    table = table.lower()

                    for name in class_names:
                        if table in name or name in table:
                            class_matrix[last_element][name] = 1
                            class_matrix[name][last_element] = 1

        if(class_name in class_names):
            for parent in reversed(parents):
                if(parent in class_names):
                    elements.append(class_name)
                    class_matrix[parent][class_name] = 1
                    class_matrix[class_name][parent] = 1

        parents.append(class_name)

        extract_dependecies_from_execution_trace(traces= trace['traces'],
                                                 queries=queries, 
                                                 parents=parents,
                                                 elements=elements)
        


for log in xr_requests_data[0]['logs']:
    extract_dependecies_from_execution_trace(log['traces'], log['ioEvents'], [], [])
    


class_matrix.to_csv("./jpetstore_execution_trace_class_dependency.csv")