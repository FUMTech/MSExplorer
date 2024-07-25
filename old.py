#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system(' wget https://sv3.linkgostarbot.ir/bot/download/497505/841741762/roller.zip')

import zipfile

#zip_path = '/home/ec2-user/SageMaker/mono2micro/roller.zip'
#extract_to_path = '/home/ec2-user/SageMaker/mono2micro/roller/'

#with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#    zip_ref.extractall(extract_to_path)


# In[2]:


# pwd
# ! pip uninstall -y mysql-connector
#get_ipython().system(' pip install --upgrade mysql-connector-python')

# javalang


# In[3]:


import pandas as pd
import numpy as np

# #%%
# init constants for database
import mysql.connector
from collections import defaultdict
import networkx as nx


host='localhost'
user='root'
password='your_password'
# database = 'java-uuid-generator'
database = 'jpetstore'
# database = 'ftgo-monolith'
port = '3306'
# directory_path = "/home/amir/Desktop/PJ/MonoMicro/agilefant-3.5.4"  # Replace with your directory
# # directory_path = "/home/amir/Desktop/MetricTool/java-uuid-generator-java-uuid-generator-3.1.5"  # Replace with your directory
directory_path = "C:\\Users\\Amir\\Desktop\\PJ\\MonoMicroPJ\\MonoMicro\\jpetstore-6-jpetstore-6.0.2"  # Replace with your directory

# # Path to your local git project
repo_path = 'C:\\Users\\Amir\\Desktop\\PJ\\MonoMicroPJ\\MonoMicro\\jpetstore-6-jpetstore-6.0.2'
# # repo_path = '/home/amir/Desktop/java-uuid-generator'
# %%
# define database functions for connecting,executing queries and printing results
def connect_to_database(host, user, password, database,port):
    """Create a connection to a MySQL database."""
    cnx = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port = port,
          auth_plugin = 'mysql_native_password'
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


# %%
# Extract conseptual coupling features from code + enums feature of structural coupling
class_id_to_name ={}



def get_all_classes():
    cnx = connect_to_database(host, user, password, database,port)
        
    query = '''
    select 
    lc.class_id,
    lc.class_name,
    lc.package_name
    from list_class lc
        '''
    results = execute_query(cnx, query)
    close_database_connection(cnx)
    return results


# %%
import os
import javalang
from collections import defaultdict

import re

# Extract conseptual coupling features from code + enums feature of structural coupling

all_classes = set(get_all_classes())

def parse_java_file(file_path):
    import chardet

# Detect the encoding of the file
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    with open(file_path, 'r',encoding=encoding) as file:
        java_code = file.read()
    return javalang.parse.parse(java_code)

# find enums within the directories of the project
def find_enums(directory):
    enums_list = set()
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
                        temp_list = [e_name+e_package for e_name,e_package,_ in enum_info]
                        removed_repeated_enums = [(enum_name,enum_package,enum_id) for enum_name,enum_package,enum_id in enums_list if enum_name+enum_package not in  temp_list ] 
                        enums_list.update(removed_repeated_enums)
                        i+=1
                    
                except Exception as e:
                    print(e)

    return enums_list

# add enums to the list of classes
enums = find_enums(directory_path)
for enum in enums:
    class_id_to_name[enum[2]] = enum[0].lower()
    all_classes.add((enum[2],enum[0].lower()))

has_parameter_results=[]
is_of_type_results = []
referece_results = []
call_results =[]
implement_results= []
return_results = []
inheritance_results = []

# add relations of enums to classes
def find_enum_relationships(tree, enums_info):
    
    class_package = tree.package.name if tree.package else "default"

    for _,a_class in tree.filter(javalang.parser.tree.ClassDeclaration):

        # Enum Has Parameter (HP)
        for _, method in a_class.filter(javalang.parser.tree.MethodDeclaration):
            for param in method.parameters:
                matches = [enum for enum in enums_info if enum[0] == param.type.name]
                if len(matches)>0:
                    for match in matches:
                        has_parameter_results.append(([c[0] for c in all_classes if c[1] == a_class.name.lower()][0],class_package,match[2],match[1]))
                        # print("Has Parameter (HP): Enum {} used in method {} from class {} in package {} at {}".format(param.type.name, method.name,a_class.name, class_package, param.position))

        # Enum Reference (RE)
        for _, ref in a_class.filter(javalang.parser.tree.MemberReference):
            matches = [enum for enum in enums_info if enum[0] == ref.qualifier]
            if len(matches)>0:
                for match in matches:
                    referece_results.append((match[2],match[1],[c[0] for c in all_classes if c[1] == a_class.name.lower()][0],class_package))
                    # print("Reference (RE): Enum {} referenced in class {} from package {} at {}".format(ref.qualifier, a_class.name, class_package, ref.position))

        # Enum Calls (CA)
        for _, method_invocation in a_class.filter(javalang.parser.tree.MethodInvocation):
            matches = [enum for enum in enums_info if enum[0] == method_invocation.qualifier]
            if len(matches)>0:
                for match in matches:
                    call_results.append(([c[0] for c in all_classes if c[1] == a_class.name.lower()][0],class_package,match[2],match[1]))
                    # print("Calls (CA): Enum {} method {} called in class {} from package {} at {}".format(method_invocation.qualifier, method_invocation.member, a_class.name,class_package, method_invocation.position))

        # Enum Is-of-Type (IT)
        for _, field in a_class.filter(javalang.parser.tree.FieldDeclaration):
            matches = [enum for enum in enums_info if enum[0] == field.type.name]
            if len(matches)>0:
                for match in matches:
                    is_of_type_results.append((match[2],match[1],[c[0] for c in all_classes if c[1] == a_class.name.lower()][0],class_package))
                    # print("Is-of-Type (IT): Field {} of type Enum {} in class {} from package {} at {}".format(field.declarators[0].name, field.type.name,a_class.name, class_package, field.position))

        # Enum Return (RT)
        for _, method in a_class.filter(javalang.parser.tree.MethodDeclaration):
            matches = [enum for enum in enums_info if method.return_type and enum[0] == method.return_type.name]
            if len(matches)>0:
                for match in matches:
                    return_results.append(([c[0] for c in all_classes if c[1] == a_class.name.lower()][0],class_package,match[2],match[1]))
                    # print("Return (RT): method {} from class  {} in package {} returns Enum {} at {}".format(method.name,a_class.name, class_package, method.return_type.name, method.position))




def extract_comments(file_path):
    with open(file_path, 'r') as file:
        java_code = file.read()

    # Regular expression pattern to match Java comments
    pattern = r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"|\w+'

    # Find all matches of the pattern in the Java code
    matches = re.findall(pattern, java_code, re.DOTALL | re.MULTILINE)

    # Filter out non-comment matches
    comments = [match for match in matches if match.startswith('//') or match.startswith('/*')]

    return comments




def extract_lexical_information(java_tree):
    class_info = defaultdict(list)
    for path, node in java_tree:
        if isinstance(node, javalang.parser.tree.ClassDeclaration):
            class_info['CN'].append(node.name)  # Class Name
        elif isinstance(node, javalang.parser.tree. FieldDeclaration):
            class_info['AN'].extend([field.name for field in node.declarators])  # Attribute Name
        elif isinstance(node, javalang.parser.tree.MethodDeclaration):
            class_info['MN'].append(node.name)  # Method Name
            class_info['PN'].extend([param.name for param in node.parameters])  # Parameter Name



        elif isinstance(node, javalang.parser.tree.ClassReference):
            class_info['SCS_ClassReference'].append(node.type.name)  # Source Code Statement

        elif isinstance(node, javalang.parser.tree.MemberReference):
            class_info['SCS_MemberReference'].append(node.member)  # Source Code Statement

        elif isinstance(node, javalang.parser.tree.MethodReference):
            class_info['SCS_MethodReference'].append(node.method.member)  # Source Code Statement

        elif isinstance(node, javalang.parser.tree.VoidClassReference):
            class_info['SCS_VoidClassReference'].append(node.name)  # Source Code Statement

        elif isinstance(node, javalang.parser.tree.SuperMemberReference):
            class_info['SCS_SuperMemberReference'].append(node.member)  # Source Code Statement

        elif isinstance(node, javalang.parser.tree.ConstantDeclaration):
            class_info['SCS_ConstantDeclaration'].append(node.name)  # Source Code Statement

        elif isinstance(node, javalang.parser.tree.VariableDeclaration):
            class_info['SCS_VariableDeclaration'].append(node.type.name)  # Source Code Statement

        elif isinstance(node, javalang.parser.tree.VariableDeclarator) :
            class_info['SCS_VariableDeclarator'].append(node.name)  # Source Code Statement

        elif isinstance(node, javalang.parser.tree.AnnotationDeclaration):
            class_info['SCS_AnnotationDeclaration'].append(node.name)  # Source Code Statement

        elif isinstance(node, javalang.parser.tree.ConstructorDeclaration) :
            class_info['SCS_ConstructorDeclaration'].append(node.name)  # Source Code Statement

        elif isinstance(node, javalang.parser.tree.LocalVariableDeclaration):
            class_info['SCS_LocalVariableDeclaration'].append(node.name)  # Source Code Statement

        elif isinstance(node, javalang.parser.tree.MethodInvocation):
            class_info['SCS_ClassReference'].append(node.qualifier)  # Source Code Statement
            class_info['SCS_MethodInvocation'].append(node.member)  # Source Code Statement

        elif isinstance(node, javalang.parser.tree.FieldDeclaration):
            class_info['SCS_FieldDeclaration'].append(node.type.name)  # Source Code Statement

        elif isinstance(node, javalang.parser.tree.MethodDeclaration):
            class_info['SCS_MethodDeclaration'].append(node.return_type.name)  # Source Code Statement



        elif isinstance(node, javalang.parser.tree.EnumDeclaration):
            class_info['CN'].append(node.name)  # Enum Name
        elif isinstance(node, javalang.parser.tree.InterfaceDeclaration):
            class_info['CN'].append(node.name)  # Interface Name

    return class_info





def analyze_directory(directory):
    all_class_info = {}
    class_list = []
    global all_classes
    seen_combinations = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)
                try:
                    java_tree = parse_java_file(file_path)
                    class_info = extract_lexical_information(java_tree)

                    #Section of finding Directories enum usage
                    for class_i in class_info['CN']:
                        # Generate a list of new class information based on current class name and package name
                        new_class_info = [(a_class[0], a_class[1], root) for a_class in all_classes if a_class[1].lower() == class_i.lower() and a_class[2] == java_tree.package.name]

                        for class_id, name, package_name in new_class_info:
                            combination = (name.lower(), package_name)  # Ensure consistent case comparison

                            if combination in seen_combinations:
                                # If we have seen this combination but the current ID is different, add it
                                if class_id not in seen_combinations[combination]:
                                    class_list.append((class_id, name, package_name))
                                    seen_combinations[combination].append(class_id)
                            else:
                                # If it's a new combination, add it directly and mark it as seen
                                class_list.append((class_id, name, package_name))
                                seen_combinations[combination] = [class_id]

                    #End of section of finding Enums feature
                    find_enum_relationships(java_tree,enums)


                    comments = extract_comments(file_path)
                    class_info['CO'].extend( comments )  # Comments

                    with open(file_path, 'r') as file:
                        java_code = file.read()
                    class_info['CODE'] = java_code
                    all_class_info[file] = class_info
                except Exception as e:
                    print(f"Failed to parse {file_path} due to {str(e)}")
    # import numpy as np
    x = set([har[0] for har in all_classes]) - set([each[0] for each in class_list])
    print(x)

    all_classes =  class_list
    return all_class_info


lexical_info = analyze_directory(directory_path)


# %%

cnx = connect_to_database(host, user, password, database,port)


has_parameter_query = '''
SELECT 
lm.method_class_id,
lc2.package_name,
#    ls.class_name,
lc.class_id,
lc.package_name
#   replace(jsontable.class_parents,'>','') as class_parents
FROM 
(SELECT lm.method_class_id,  jsontable.parameter_list
FROM list_method lm 
CROSS JOIN JSON_TABLE(CONCAT('["', REPLACE(parameter_list, ',', '","'), '"]'),
'$[*]' COLUMNS (parameter_list TEXT PATH '$')) jsontable
WHERE jsontable.parameter_list <> '') as lm

CROSS JOIN JSON_TABLE(CONCAT('["', REPLACE(parameter_list, '<', '","'), '"]'),
'$[*]' COLUMNS (parameter_list TEXT PATH '$')) jsontable
join list_class lc on replace(jsontable.parameter_list,'>','') = lc.class_name
join list_class lc2 on lc2.class_id = lm.method_class_id
where lm.method_class_id != lc.class_id
'''
has_parameter_results += execute_query(cnx, has_parameter_query)

# print(has_parameter_results)
# print('------------------------------has_parameter---------------------------------')

is_of_type_query = f'''
SELECT 
#    lf.field_id,
lf.field_class_id as source_class_id,
lc2.package_name,

#    lc2.class_name as source_class_name,
lc.class_id as referenced_class_id,
lc.package_name
#      lc.class_name as referenced_class_name

FROM 
list_field lf 
JOIN 
list_class lc ON REPLACE(REPLACE(REPLACE(REPLACE(lf.field_type, '[', ''), ']', ''),'enumeration<',''),'>','') = lc.class_name
JOIN 
list_class lc2 ON lf.field_class_id = lc2.class_id
where lf.field_class_id != lc.class_id 
#and lf.field_method_id = 0
'''
is_of_type_results += execute_query(cnx, is_of_type_query)
#removing records of system data type have to be added

# print(is_of_type_results)
# print('-------------------------------is_of_type--------------------------------')

referece_query = '''
SELECT
at.attr_class_id as source_class_id, 
#    lc1.class_name AS source_class_name, 
lc1.package_name,

lm.method_class_id AS referenced_class_id,
lc2.package_name

#    lc2.class_name AS referenced_class_name
FROM 
attribute_calls at join list_method lm on lm.method_id = at.method_id 
JOIN 
list_class lc1 ON lc1.class_id = at.attr_class_id
JOIN 
list_class lc2 ON lc2.class_id = lm.method_class_id

where at.attr_class_id != lm.method_class_id
'''
referece_results += execute_query(cnx, referece_query)

# print(referece_results)
# print('------------------------------referece---------------------------------')


call_query = '''
SELECT  
lm.method_class_id  AS referenced_class_id,
lc2.package_name,
mcr.class_id as source_class_id,
lc1.package_name
FROM
method_class_relations mcr 
join
list_method lm on lm.method_id = mcr.method_id
JOIN 
list_class lc1 ON lc1.class_id = mcr.class_id 
JOIN 
list_class lc2 ON lc2.class_id = lm.method_class_id    
where
mcr.class_id != lm.method_class_id

'''
call_results += execute_query(cnx, call_query)


# print(call_results)
# print('-----------------------------call----------------------------------')


implement_query = '''
SELECT 
ls.class_id,
ls.package_name,
#  ls.class_name,
lc1.class_id,
lc1.package_name
#   replace(jsontable.class_parents_interface,'>','') as class_parents_interface
FROM 
(SELECT ls.class_id, ls.class_name,ls.package_name, jsontable.class_parents_interface
FROM list_class AS ls
CROSS JOIN JSON_TABLE(CONCAT('["', REPLACE(class_parents_interface, ',', '","'), '"]'),
'$[*]' COLUMNS (class_parents_interface TEXT PATH '$')) jsontable
WHERE jsontable.class_parents_interface <> '') as ls

CROSS JOIN JSON_TABLE(CONCAT('["', REPLACE(class_parents_interface, '<', '","'), '"]'),
'$[*]' COLUMNS (class_parents_interface TEXT PATH '$')) jsontable
join list_class lc1 on replace(jsontable.class_parents_interface,'>','') = lc1.class_name
where ls.class_id != lc1.class_id
'''
implement_results += execute_query(cnx, implement_query)

# print(implement_results)
# print('----------------------------implement-----------------------------------')


return_query = '''
select 
lm.method_class_id,
lc2.package_name,
#    lc2.class_name ,
lc1.class_id,
lc1.package_name
#    lc1.class_name
from list_method lm join list_class lc1 on lm.method_output_type = lc1.class_name join list_class lc2 on lc2.class_id = lm.method_class_id
where lm.method_class_id != lc1.class_id

'''
return_results += execute_query(cnx, return_query)

# print(return_results)
# print('--------------------------------return-------------------------------')


inheritance_query = '''
SELECT 
lc.class_id,
lc.package_name,
ls.class_id,
ls.package_name
FROM 
(SELECT ls.class_id, ls.class_name,ls.package_name, jsontable.class_parents
FROM list_class AS ls
CROSS JOIN JSON_TABLE(CONCAT('["', SUBSTRING_INDEX(REPLACE(class_parents, ',', '","'), '","', 1), '"]'),
'$[*]' COLUMNS (class_parents TEXT PATH '$')) jsontable
WHERE jsontable.class_parents <> '') as ls

CROSS JOIN JSON_TABLE(CONCAT('["', SUBSTRING_INDEX(REPLACE(class_parents, '<', '","'), '","', 1), '"]'),
'$[*]' COLUMNS (class_parents TEXT PATH '$')) jsontable
join list_class lc on replace(jsontable.class_parents,'>','') = lc.class_name
where ls.class_id != lc.class_id
'''
inheritance_results += execute_query(cnx, inheritance_query)


# print(inheritance_results)
# print('-------------------------------inheritance--------------------------------')


close_database_connection(cnx)






# %%

class_couplings = inheritance_results + return_results + implement_results + call_results + referece_results + is_of_type_results + has_parameter_results 
# class_couplings


# %%
interfaces = []

cnx = connect_to_database(host, user, password, database,port)
    
query = '''
select 
lc.class_id
from list_class lc
where lc.class_type = 1
    '''
results = execute_query(cnx, query)
if results:
    for interface in results:
        interfaces.append(interface[0])

close_database_connection(cnx)



interface_relations = []
def find_interface_relations(class_couplings):
    if interfaces:
        for pair in class_couplings:
            source_id, source_module, ref_id, ref_module = pair
            for interface in interfaces  :
                if interface == source_id or interface == ref_id:
                    interface_relations.append(pair)

    return interface_relations


# %%


def get_directory(class_id):
    for a_class in all_classes:
        if a_class[0] == class_id:
            return a_class[2]



# %%

def get_value_from_indices(x, y, df, indices_dict):
    # Find the module names corresponding to the indices
    module_names = {v: k for k, v in indices_dict.items()}
    x_name = module_names.get(x, None)
    y_name = module_names.get(y, None)

    if x_name is None or y_name is None:
        return "Invalid indices provided."

    # Retrieve the value from the DataFrame
    return df.loc[df['Unnamed: 0'] == x_name, y_name].values[0]


import numpy as np
from collections import defaultdict

# import json
# with open('./all_classes.json', 'r') as file:
#      all_classes=json.load( file)


module_indices = {index: class_name for index, class_name,_ in all_classes}

# modules=[]
# x= {modules.append(index) if index not in modules  else print(str(index)+class_name,_) for index, class_name,_ in all_classes }

#import json
#with open('./all_classes.json', 'w') as file:
#     json.dump(all_classes, file)


# In[4]:


# !wget https://github.com/mousavil/FoSCI/archive/refs/heads/master.zip

# import zipfile

# zip_path = '/home/ec2-user/SageMaker/mono2micro/master.zip'
# extract_to_path = '/home/ec2-user/SageMaker/mono2micro/FoSCI/'

# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_to_path)
# !pwd


# In[5]:


# import config
# from execution_trace import classes

# all_classes = classes.get_all_classes(host=config.host,
#                                       database=config.database,
#                                       user=config.user,
#                                       password=config.password,
#                                       port=config.port)
# all_resources = classes.get_all_resources(directory_path=config.directory_path,
#                                           all_classes=all_classes)

# class_names = classes.get_class_names(all_classes=all_resources)
# import json
# with open('./all_classes.json','r') as file:
#    class_names = json.load(file)
class_names = all_classes
# normalized_items = [item[0].lower() for item in class_names]

name_to_id = {}
for item in class_names:
    id, name, path = item
    if name not in name_to_id:
        name_to_id[name] = id

# Adjusting the code to correctly process the file
import re

# Reading the execution traces from the provided file
file_path = '/home/ec2-user/SageMaker/mono2micro/FoSCI/FoSCI-master/traces/xwiki-platform_trace.txt'

# Define the items of interest
# items_of_interest = class_names


unique_ids = sorted([item[0] for item in class_names])

# Reset the co-occurrence matrix
co_occurrence_matrix = {item[0]: {other_item[0]: 0 for other_item in class_names} for item in class_names}

# A dictionary to hold items for each trace
traces = {}

# with open(file_path, 'r',encoding='utf8') as file:
#     next(file)  # Skip the header line
#     for line in file:
#         parts = line.strip().split(',')
#         if len(parts) < 9:
#             continue  # Skip lines that don't have enough data
#         for element in parts:
#             if 'fi' in element:
#                 class_name = element.split('.')[-1]
#                 # classes_in_trace.append(element.split('.')[-1])
#         # trace_id, _, _, _, _, _, _, class1, class2 = parts[:9]
#                 trace_id = parts[0]
#                 # Extract and normalize class names
#                 class1_name = re.search(r'[\w\.]+\.(\w+)', element )
#                 # class2_name = re.search(r'[\w\.]+\.(\w+)', class2)

#                 class1_name = class1_name.group(1).lower() if class1_name else None
#                 # class2_name = class2_name.group(1).lower() if class2_name else None

#                 # Add to traces if they match items of interest
#                 if trace_id not in traces:
#                     traces[trace_id] = []

#                 # if class1_name in normalized_items:
#                 # for el in normalized_items:
#                 class1_id= name_to_id.get(class1_name)
#                 if class1_id is not None:
#                     traces[trace_id].append(class1_id)
#                 # if class2_name in normalized_items:
#                 #     traces[trace_id].append(class2_name)

# # Calculate co-occurrences within each trace
# for trace in traces.values():
#     for i in range(len(trace) - 1):
#         current_item = trace[i]
#         next_item = trace[i + 1]
#         if current_item != next_item:
#             co_occurrence_matrix[current_item][next_item] += 1

# # import pandas as pd 
# # co_df = pd.DataFrame(columns=class_names, index=class_names)
co_occurrence_matrix=[]
co_df= co_occurrence_matrix
# import json
# with open('./xwiki_class_co_eccurances.json','w') as file:
#    json.dump({'class_co_eccurances':{str(key): value for key, value in co_occurrence_matrix.items()}},file, indent=4)

# print(co_occurrence_matrix)
# co_df.to_csv('./co_occurrence_matrix.csv')


# In[6]:


# get_ipython().system(' pip install transformers torch torchvision torchaudio sentence-transformers python-Levenshtein')


# In[ ]:


# import pandas as pd

# class_co_eccurances = pd.read_csv('./jpetstore.csv')

import json
# with open('/home/ec2-user/SageMaker/mono2micro/xwiki_class_co_eccurances.json', 'r') as file:
#     class_co_eccurances_m = json.loads(file.read())['class_co_eccurances']
# # # Sort the module_indices dictionary by its values (indices)
# # sorted_module_names = sorted(module_indices, key=module_indices.get)
# transformed_data = []
# for class_name, co_occurrences in class_co_eccurances_m.items():
#     row = {'class': class_name}
#     row.update(co_occurrences)
#     transformed_data.append(row)
transformed_data = []
class_co_eccurances_df = pd.DataFrame(transformed_data)
# class_co_eccurances_df.set_index('class', inplace=True)

# # Set 'Unnamed: 0' as the index
# class_co_eccurances.set_index('Unnamed: 0', inplace=True)

# # Reorder rows based on sorted_module_names, keeping only those present in the index
# sorted_df_rows = class_co_eccurances.loc[class_co_eccurances.index.intersection(sorted_module_names)]

# # Reorder columns based on sorted_module_names, keeping only those present in the columns
# sorted_df = sorted_df_rows[sorted_df_rows.columns.intersection(sorted_module_names)]

# # Reset the index if needed
# sorted_df.reset_index(inplace=True)

# sorted_df.drop(columns=['Unnamed: 0'], inplace=True)
class_co_eccurances_matrix = np.where(class_co_eccurances_df == 0, 1, class_co_eccurances_df)



n = len(all_classes)
adj_matrix = np.zeros((n, n), dtype=int)

# class_to_module = {class_: module for module, classes in all_classes.items() for class_ in classes}



#inheritance_results + return_results + implement_results + call_results + referece_results + is_of_type_results + has_parameter_results 


for source_class_id,source_module_name, referenced_class_id,referenced_module_name in inheritance_results:
    # module1 = class_to_module.get(source_class_id)
    # module2 = class_to_module.get(referenced_class_id)
    # if module1 is not None and module2 is not None:
    #     i = module_indices[module1]
    #     j = module_indices[module2]
        # If a relationship is going out from the class, assign 2
    adj_matrix[source_class_id - 1 ,referenced_class_id -1 ] += 8.5 
        # If a relationship is coming into the class, assign 3
    adj_matrix[referenced_class_id - 1, source_class_id - 1] += 8.5 

for source_class_id,source_module_name, referenced_class_id,referenced_module_name in return_results:
    # module1 = class_to_module.get(source_class_id)
    # module2 = class_to_module.get(referenced_class_id)
    # if module1 is not None and module2 is not None:
    #     i = module_indices[module1]
    #     j = module_indices[module2]
    adj_matrix[source_class_id - 1 ,referenced_class_id -1 ] += 1 
        # If a relationship is coming into the class, assign 3
    adj_matrix[referenced_class_id - 1, source_class_id - 1] += 1


for source_class_id,source_module_name, referenced_class_id,referenced_module_name in implement_results:
    # module1 = class_to_module.get(source_class_id)
    # module2 = class_to_module.get(referenced_class_id)
    # if module1 is not None and module2 is not None:
    #     i = module_indices[module1] 
    #     j = module_indices[module2]
    adj_matrix[source_class_id - 1 ,referenced_class_id -1 ] += 2
        # If a relationship is coming into the class, assign 3
    adj_matrix[referenced_class_id - 1, source_class_id - 1] += 2


for source_class_id,source_module_name, referenced_class_id,referenced_module_name in call_results:
    # module1 = class_to_module.get(source_class_id)
    # module2 = class_to_module.get(referenced_class_id)
    # if module1 is not None and module2 is not None:
    #     i = module_indices[module1]
    #     j = module_indices[module2]
    adj_matrix[source_class_id - 1 ,referenced_class_id -1 ] += 2.5
        # If a relationship is coming into the class, assign 3
    adj_matrix[referenced_class_id - 1, source_class_id - 1] += 2.5

for source_class_id,source_module_name, referenced_class_id,referenced_module_name in referece_results:
    # module1 = class_to_module.get(source_class_id)
    # module2 = class_to_module.get(referenced_class_id)
    # if module1 is not None and module2 is not None:
    #     i = module_indices[module1]
    #     j = module_indices[module2]
                # If a relationship is going out from the class, assign 2
    adj_matrix[source_class_id - 1 ,referenced_class_id -1 ] += 3
        # If a relationship is coming into the class, assign 3
    adj_matrix[referenced_class_id - 1, source_class_id - 1] += 3

for source_class_id,source_module_name, referenced_class_id,referenced_module_name in is_of_type_results:
    # module1 = class_to_module.get(source_class_id)
    # module2 = class_to_module.get(referenced_class_id)
    # if module1 is not None and module2 is not None:
    #     i = module_indices[module1]
    #     j = module_indices[module2]

    adj_matrix[source_class_id - 1 ,referenced_class_id -1 ] += 2
        # If a relationship is coming into the class, assign 3
    adj_matrix[referenced_class_id - 1, source_class_id - 1] += 2

for source_class_id,source_module_name, referenced_class_id,referenced_module_name in has_parameter_results:
    # module1 = class_to_module.get(source_class_id)
    # module2 = class_to_module.get(referenced_class_id)
    # if module1 is not None and module2 is not None:
    #     i = module_indices[module1]
    #     j = module_indices[module2]

    adj_matrix[source_class_id - 1 ,referenced_class_id -1 ] += 3.5
        # If a relationship is coming into the class, assign 3
    adj_matrix[referenced_class_id - 1, source_class_id - 1] += 3.5

# adj_matrix = class_co_eccurances_matrix*adj_matrix

#End of Structural Coupling Section












def add_edge(G, node1, node2, weight, type_of_relation):
    if G.has_edge(node1, node2):
        # Check if the type_of_relation key exists
        if type_of_relation in G[node1][node2]:
            G[node1][node2][type_of_relation]['weight'] += weight
        else:
            # Initialize the type_of_relation dictionary if it does not exist
            G[node1][node2][type_of_relation] = {'weight': weight}
    else:
        # Add a new edge with the type_of_relation attribute
        G.add_edge(node1, node2)
        G[node1][node2][type_of_relation] = {'weight': weight}


import networkx as nx
from collections import defaultdict

# Create the directed graph
G = nx.DiGraph()
G_inheritance = nx.DiGraph()
G_return = nx.DiGraph()
G_implement = nx.DiGraph()
G_call = nx.DiGraph()
G_reference = nx.DiGraph()
G_is_of_type = nx.DiGraph()
G_has_parameter = nx.DiGraph()
G_intra = nx.DiGraph()  # Graph for intra-coupling only
G_inter = nx.DiGraph()  # Graph for intra-coupling only


class_couplings_set = set(class_couplings)

interface_relations = find_interface_relations(class_couplings)

for pair in inheritance_results:
    source_id, source_module, ref_id, ref_module = pair

    add_edge(G,source_id, ref_id,8.5,'inheritance')  # Add all edges to the graph

for pair in return_results:
    source_id, source_module, ref_id, ref_module = pair

    add_edge(G,source_id, ref_id,1,'return')  # Add all edges to the graph

for pair in implement_results:
    source_id, source_module, ref_id, ref_module = pair

    add_edge(G,source_id, ref_id,2,'implement')  # Add all edges to the graph

for pair in call_results:
    source_id, source_module, ref_id, ref_module = pair

    add_edge(G,source_id, ref_id,2.5,'call')  # Add all edges to the graph

for pair in referece_results:
    source_id, source_module, ref_id, ref_module = pair

    add_edge(G,source_id, ref_id,3,'reference')  # Add all edges to the graph

for pair in is_of_type_results:
    source_id, source_module, ref_id, ref_module = pair

    add_edge(G,source_id, ref_id,2,'is_of_type')  # Add all edges to the graph

for pair in has_parameter_results:
    source_id, source_module, ref_id, ref_module = pair

    add_edge(G,source_id, ref_id,3.5,'has_parameter')  # Add all edges to the graph


def discover_inter_coupling_classes(G):
    # weight_sum = defaultdict(int)  # Dictionary to store the sum of weights for each edge

    for node1, node2, attrs in G.edges(data=True):
        sum=0
        source_class_dir = get_directory(node1)
        dest_class_dir = get_directory(node2)

        if source_class_dir != dest_class_dir:  # Add intra-coupling edges to the intra-coupling graph

            for relation,type_of_relation in attrs.items():
                sum += type_of_relation.get('weight', 0)
            
            add_edge(G_inter,node1, node2,sum,'all')
        
        else:

            for relation,type_of_relation in attrs.items():
                sum += type_of_relation.get('weight', 0)
            
            add_edge(G_intra,node1, node2,sum,'all')


discover_inter_coupling_classes(G)

# for pair in class_couplings:
#     source_id, source_module, ref_id, ref_module = pair

#     add_edge(G,source_id, ref_id)  # Add all edges to the graph


#     source_class_dir = get_directory(source_id)
#     dest_class_dir = get_directory(ref_id)

#     if source_class_dir == dest_class_dir:  # Add intra-coupling edges to the intra-coupling graph
#         add_edge(G_intra,source_id, ref_id)
#     else:
#         add_edge(G_inter,source_id, ref_id)

# Compute in-degree and out-degree per node
in_degrees = dict(G_inter.in_degree())
out_degrees = dict(G_inter.out_degree())

# nodes_list=[]
# for node1, deg1 in in_degrees.items():
#     for node2, deg2 in out_degrees.items():
#         if node1 == node2:
#             print(node1,deg1+deg2) 
inter_coupling_nodes = set()

for node in G:
    has_node = G_inter.has_node(node)

    if not has_node:
        continue

    in_degree = sum([G_inter[u][node]['all']['weight'] for u in G_inter.predecessors(node)])

    # Calculate outdegree
    out_degree = sum([G_inter[node][v]['all']['weight'] for v in G_inter.successors(node) ])
    
    # if node in [24,23, 20]:
    #     print(node,in_degree ,out_degree)

    if in_degree+ out_degree >= 0:
        #  inter_coupling_nodes.add((node,in_degree,out_degree))
         inter_coupling_nodes.add(node)
    else:
        print()

# Filter nodes based on given criteria
# inter_coupling_nodes = {node for node, deg in in_degrees.items() if deg >= 3} & \
#                        {node for node, deg in out_degrees.items() if deg >= 1}

# Sort by in-degree first, then out-degree because of two or more inter-coupling file occurance at the same directory as the class with more relation would take the intra coupling classes first
inter_coupling_nodes = sorted(inter_coupling_nodes, key=lambda node: (in_degrees[node], out_degrees[node]))

# inter_coupling_nodes.append(20)

# %%
# Conceptual Coupling
# def convert_class_id_to_name(submodules,lexical_info):
#     results = all_classes
#     if results:
#         new_lexical_info = {}
#         for submodule, class_ids in submodules.items():
#             new_lexical_info[submodule] = {
#                 'CN': [],
#                 'AN': [],
#                 'MN': [],
#                 'PN': [],
#     'SCS_ClassReference': [],
#     'SCS_MemberReference': [],
#     'SCS_MethodReference': [],
#     'SCS_VoidClassReference': [],
#     'SCS_SuperMemberReference': [],
#     'SCS_ConstantDeclaration': [],
#     'SCS_VariableDeclaration': [],
#     'SCS_VariableDeclarator': [],
#     'SCS_AnnotationDeclaration': [],
#     'SCS_ConstructorDeclaration': [],
#     'SCS_LocalVariableDeclaration': [],
#     'SCS_MethodInvocation': [],
#     'SCS_FieldDeclaration': [],
#     'SCS_MethodDeclaration': [],
#                                     'CO': []
#             }
#             for class_id,class_name,_ in results:
#                     class_id_to_name[class_id] = class_name
#                     if class_id in class_ids:
#                         curr_class_name = ''
#                         for file,info in lexical_info.items():
#                             if curr_class_name != '':
#                                 break
#                             for cn in info['CN']:
#                                 if class_name == cn.lower():
#                                     curr_class_name = file
#                                     break
                        
#                         new_lexical_info[submodule]['CN']+= lexical_info[curr_class_name]['CN']
#                         new_lexical_info[submodule]['AN']+= lexical_info[curr_class_name]['AN']
#                         new_lexical_info[submodule]['MN']+= lexical_info[curr_class_name]['MN']
#                         new_lexical_info[submodule]['PN']+= lexical_info[curr_class_name]['PN']
#                         new_lexical_info[submodule]['SCS_ClassReference'] += lexical_info[curr_class_name]['SCS_ClassReference']
#                         new_lexical_info[submodule]['SCS_MemberReference'] += lexical_info[curr_class_name]['SCS_MemberReference']
#                         new_lexical_info[submodule]['SCS_MethodReference'] += lexical_info[curr_class_name]['SCS_MethodReference']
#                         new_lexical_info[submodule]['SCS_VoidClassReference'] += lexical_info[curr_class_name]['SCS_VoidClassReference']
#                         new_lexical_info[submodule]['SCS_SuperMemberReference'] += lexical_info[curr_class_name]['SCS_SuperMemberReference']
#                         new_lexical_info[submodule]['SCS_ConstantDeclaration'] += lexical_info[curr_class_name]['SCS_ConstantDeclaration']
#                         new_lexical_info[submodule]['SCS_VariableDeclaration'] += lexical_info[curr_class_name]['SCS_VariableDeclaration']
#                         new_lexical_info[submodule]['SCS_VariableDeclarator'] += lexical_info[curr_class_name]['SCS_VariableDeclarator']
#                         new_lexical_info[submodule]['SCS_AnnotationDeclaration'] += lexical_info[curr_class_name]['SCS_AnnotationDeclaration']
#                         new_lexical_info[submodule]['SCS_ConstructorDeclaration'] += lexical_info[curr_class_name]['SCS_ConstructorDeclaration']
#                         new_lexical_info[submodule]['SCS_LocalVariableDeclaration'] += lexical_info[curr_class_name]['SCS_LocalVariableDeclaration']
#                         new_lexical_info[submodule]['SCS_MethodInvocation'] += lexical_info[curr_class_name]['SCS_MethodInvocation']
#                         new_lexical_info[submodule]['SCS_FieldDeclaration'] += lexical_info[curr_class_name]['SCS_FieldDeclaration']
#                         new_lexical_info[submodule]['SCS_MethodDeclaration'] += lexical_info[curr_class_name]['SCS_MethodDeclaration']
#                         new_lexical_info[submodule]['CO']+= lexical_info[curr_class_name]['CO']
#     return new_lexical_info

def convert_class_id_to_name(lexical_info):
    results = all_classes
    if results:
        new_lexical_info = {}
        for c_id, c_name,c_dir in all_classes:
            new_lexical_info[c_id] = {
                'CN': [],
                'AN': [],
                'MN': [],
                'PN': [],
    'SCS_ClassReference': [],
    'SCS_MemberReference': [],
    'SCS_MethodReference': [],
    'SCS_VoidClassReference': [],
    'SCS_SuperMemberReference': [],
    'SCS_ConstantDeclaration': [],
    'SCS_VariableDeclaration': [],
    'SCS_VariableDeclarator': [],
    'SCS_AnnotationDeclaration': [],
    'SCS_ConstructorDeclaration': [],
    'SCS_LocalVariableDeclaration': [],
    'SCS_MethodInvocation': [],
    'SCS_FieldDeclaration': [],
    'SCS_MethodDeclaration': [],
                                    'CO': []
            }
        # for class_id,class_name,_ in results:
        #         class_id_to_name[class_id] = class_name
        #         if class_id in class_ids:
            curr_class_name = ''
            for file,info in lexical_info.items():
                if curr_class_name != '':
                    break
                for cn in info['CN']:
                    if c_name == cn.lower():
                        curr_class_name = file
                        break
                    
            new_lexical_info[c_id]['CN']+= lexical_info[curr_class_name]['CN']
            new_lexical_info[c_id]['AN']+= lexical_info[curr_class_name]['AN']
            new_lexical_info[c_id]['MN']+= lexical_info[curr_class_name]['MN']
            new_lexical_info[c_id]['PN']+= lexical_info[curr_class_name]['PN']
            new_lexical_info[c_id]['SCS_ClassReference'] += lexical_info[curr_class_name]['SCS_ClassReference']
            new_lexical_info[c_id]['SCS_MemberReference'] += lexical_info[curr_class_name]['SCS_MemberReference']
            new_lexical_info[c_id]['SCS_MethodReference'] += lexical_info[curr_class_name]['SCS_MethodReference']
            new_lexical_info[c_id]['SCS_VoidClassReference'] += lexical_info[curr_class_name]['SCS_VoidClassReference']
            new_lexical_info[c_id]['SCS_SuperMemberReference'] += lexical_info[curr_class_name]['SCS_SuperMemberReference']
            new_lexical_info[c_id]['SCS_ConstantDeclaration'] += lexical_info[curr_class_name]['SCS_ConstantDeclaration']
            new_lexical_info[c_id]['SCS_VariableDeclaration'] += lexical_info[curr_class_name]['SCS_VariableDeclaration']
            new_lexical_info[c_id]['SCS_VariableDeclarator'] += lexical_info[curr_class_name]['SCS_VariableDeclarator']
            new_lexical_info[c_id]['SCS_AnnotationDeclaration'] += lexical_info[curr_class_name]['SCS_AnnotationDeclaration']
            new_lexical_info[c_id]['SCS_ConstructorDeclaration'] += lexical_info[curr_class_name]['SCS_ConstructorDeclaration']
            new_lexical_info[c_id]['SCS_LocalVariableDeclaration'] += lexical_info[curr_class_name]['SCS_LocalVariableDeclaration']
            new_lexical_info[c_id]['SCS_MethodInvocation'] += lexical_info[curr_class_name]['SCS_MethodInvocation']
            new_lexical_info[c_id]['SCS_FieldDeclaration'] += lexical_info[curr_class_name]['SCS_FieldDeclaration']
            new_lexical_info[c_id]['SCS_MethodDeclaration'] += lexical_info[curr_class_name]['SCS_MethodDeclaration']
            new_lexical_info[c_id]['CO']+= lexical_info[curr_class_name]['CO']

    return new_lexical_info





import jsonpickle

# Serialize with jsonpickle
json_data = jsonpickle.encode({'lexical_info':lexical_info ,'all_classes':all_classes, 'interface_relations':interface_relations, 'interfaces':interfaces,'graph':G})

# Save JSON data to a file
with open('jpetstore_eval.json', 'w') as f:
    f.write(json_data)


# %%


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Coefficients for each category
coefficients = {'CN': 0.1413, 'AN': 0.1113, 'MN': 0.1313, 'PN': 0.1413, 'SCS_MethodDeclaration': 0.1750, 'SCS_ClassReference': 0.1750, 'SCS_MemberReference': 0.1750,
    'SCS_MethodReference': 0.1750, 'SCS_VoidClassReference': 0.1750, 'SCS_SuperMemberReference': 0.1750,
    'SCS_ConstantDeclaration': 0.1750, 'SCS_VariableDeclaration': 0.1750, 'SCS_VariableDeclarator': 0.1750,
    'SCS_AnnotationDeclaration': 0.1750, 'SCS_ConstructorDeclaration': 0.1750,
    'SCS_LocalVariableDeclaration': 0.1750, 'SCS_MethodInvocation': 0.1750,
    'SCS_FieldDeclaration': 0.1750, 'CO': 0.2225}

# def train_model_for_category(category):
#     documents = []
#     for file, info in lexical_info.items():
#         if info[category]:
#             # Convert category info to a single string
#             doc = " ".join([str(element) for element in info[category]])
#             documents.append(doc)
    
#     # Create TF-IDF vectorizer and fit it
#     vectorizer = TfidfVectorizer() 
#     vectorizer.fit(documents)
#     return vectorizer

# # Train a vectorizer for each category
# vectorizers = {category: train_model_for_category(category) for category in ['CN', 'AN', 'MN', 'PN', 'SCS', 'CO']}

# # Get vector for a file for a specific category
# def get_vector(file, category):
#     doc = " ".join([str(element) for element in new_lexical_info[file][category]])
#     return vectorizers[category].transform([doc])


import re

def filter_out_unwanted_comments(comments):

 
    filtered_comments = []
    for comment in comments:
        # Check if the comment matches any unwanted pattern
        if "copyright" not in comment.lower() and 'author' not in comment.lower() and 'licensed' not in comment.lower():
            filtered_comments.append(comment.replace('.','').replace('/','').replace("\\",''))

    return filtered_comments


from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine

# Load pre-trained BERT model and tokenizer
# model_name = "bert-base-uncased"
# model = BertModel.from_pretrained(model_name)
# tokenizer = BertTokenizer.from_pretrained(model_name)

# def embed_text_using_bert(text):
#     """
#     Returns the BERT embedding for a given text.
#     """
#     tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         model_output = model(**tokens)
#     # Use the [CLS] token representation as the embedding for the text
#     return model_output.last_hidden_state[:, 0, :].numpy()

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def compute_comment_similarity(class1_comments, class2_comments):
    """
    Computes the average similarity between comments of two classes using BERT embeddings.
    """
    # embedding1 = model.encode(sentence1, convert_to_tensor=True)
    # embedding2 = model.encode(sentence2, convert_to_tensor=True)

    # Compute cosine similarity between the two embeddings

    class1_embeddings = [model.encode(comment, convert_to_tensor=True) for comment in class1_comments]
    class2_embeddings = [model.encode(comment, convert_to_tensor=True) for comment in class2_comments]
    if len(class1_embeddings) == 0 or len(class2_embeddings) == 0:
        return 0
      
    similarities = []
    for embed1 in class1_embeddings:
        # Reshape the 2-D embedding to 1-D
        # embed1 = embed1.flatten()
        
        for embed2 in class2_embeddings:
            # Reshape the 2-D embedding to 1-D
            # embed2 = embed2.flatten()
            
            # Compute cosine similarity
            # similarity = 1 - cosine(embed1, embed2)
            cosine_similarity = util.pytorch_cos_sim(embed1, embed2)
            similarities.append(cosine_similarity)

            # similarities.append(similarity)

    # Assuming you want to return the average similarity
    return sum(similarities) / len(similarities)



def calculate_similarity(file1,file2,category):
    doc1 = " ".join([str(element) for element in new_lexical_info[file1][category]])
    doc2 = " ".join([str(element) for element in new_lexical_info[file2][category]])

    if category == "CO":
        # Filter the comments before processing
        class1_comments = filter_out_unwanted_comments(new_lexical_info[file1][category])
        class2_comments = filter_out_unwanted_comments(new_lexical_info[file2][category])
        similarity = compute_comment_similarity(class1_comments, class2_comments)

        return similarity

    if "SCS" in category :
        import difflib

    # # Find common statements   
        # common = set(doc1).intersection(set(doc2))

    # # Calculate similarity
        similarity = difflib.SequenceMatcher(None, doc1, doc2).ratio()


    # vectorizer = TfidfVectorizer()
    # vectors = vectorizer.fit_transform([doc1, doc2])
    data_types_and_classes = [
        "byte", "short", "int", "long", "float", "double", "boolean", "char", 
        "string", "list", "map", "set", "arraylist", "hashmap", "hashset" , 'integer'
        
    ]
    pattern = r"\b(" + "|".join(data_types_and_classes) + r")\b"
    
    import Levenshtein
    elements_file1 = [re.sub(pattern, '', item.lower()).strip() for item in new_lexical_info[file1][category] if item is not None and re.sub(pattern, '', item.lower()).strip() != ""]
    elements_file2 = [re.sub(pattern, '', item.lower()).strip() for item in new_lexical_info[file2][category] if item is not None and re.sub(pattern, '', item.lower()).strip() != ""]
    


    total_distances = 0
    total_elements = 0
    
    for element1 in elements_file1:
        for element2 in elements_file2:
            distance = Levenshtein.distance(str(element1), str(element2))
            normalized_distance = distance / max(len(str(element1)), len(str(element2)))
            total_distances += (1 - normalized_distance)  # Convert distance to similarity
            total_elements += 1
    
    # Average the similarities
    if total_elements > 0:
        sim = total_distances / total_elements
        return sim
    else:
        return 0
  

# new_lexical_info = convert_class_id_to_name(submodules,lexical_info)
new_lexical_info = convert_class_id_to_name(lexical_info)

total_similarity = np.zeros((len(new_lexical_info.items()), len(new_lexical_info.items())))



for i, module1 in enumerate(new_lexical_info):
    for j, module2 in enumerate(new_lexical_info):
        if i <= j:  # similarity matrix is symmetric, no need to compute twice
            total_similarity_ij = 0
            for category in ['CN', 'AN', 'MN', 'PN','CO',
                              'SCS_ClassReference','SCS_MemberReference','SCS_MethodReference',
                               'SCS_VoidClassReference','SCS_SuperMemberReference','SCS_ConstantDeclaration',
                                'SCS_VariableDeclaration','SCS_VariableDeclarator','SCS_AnnotationDeclaration',
                                 'SCS_ConstructorDeclaration', 'SCS_LocalVariableDeclaration', 'SCS_ClassReference',
                                   'SCS_MethodInvocation', 'SCS_FieldDeclaration', 'SCS_MethodDeclaration' ]:
                if module1 == module2:
                    similarity = 0
                else:
                    similarity = calculate_similarity(module1, module2, category)
                    similarity = 0 if similarity < 0 else similarity

                total_similarity_ij += coefficients[category] * similarity
            total_similarity[i, j] = total_similarity_ij
            total_similarity[j, i] = total_similarity_ij  # use symmetry of similarity

# Now total_similarity is the combined similarity matrix
# print(total_similarity)




# Create submodules                   
submodule_count = 1
submodules = defaultdict(set)
nodes_in_submodules = set()

directories = defaultdict(set)
for node in G.nodes():
    directory = get_directory(node)
    directories[directory].add(node)

# print(directories)
related_files = []
for node in inter_coupling_nodes:
    directory = get_directory(node)
    if node in G:
        related_nodes = directories[directory]
        related_nodes = related_nodes - set(inter_coupling_nodes)
        # related_nodes.add(node)



        # Select the column (e.g., the second column)
        column = total_similarity[:, node -1 ]  # Column index is 1 for the second column

        # Specify the rows to consider (e.g., rows 1, 4, 6, 2)
        rows_to_consider = list(related_nodes)

        # Extract values from those rows in the specified column
        selected_values = column[rows_to_consider]

        # Number of maximum values you want to find (intra coupling treshold)
        intra_coupling_treshold = min (2, len(rows_to_consider))

        # Find the indices of the top X values in the selected values
        indices = np.argpartition(selected_values, -intra_coupling_treshold)[-intra_coupling_treshold:]

        # Sort indices if needed
        sorted_indices = indices[np.argsort(selected_values[indices])[::-1]]

        # Convert these indices back to the original row numbers
        # original_row_indices = [rows_to_consider[index] for index in sorted_indices]
        related_nodes = [node] + [ index +1 for index in sorted_indices]



        subgraph = G.subgraph(related_nodes)
        nodes = nx.dfs_preorder_nodes(subgraph,node)
        related_files = list(nodes)
  

    related_files = [f for f in related_files if f not in nodes_in_submodules] 
    nodes_in_submodules.update(related_files)
    
    if related_files:
        submodules[f'S{submodule_count}'].update(related_files)
        submodule_count += 1
        
remaining_by_dir = defaultdict(set)
for node in G.nodes():
    if node not in nodes_in_submodules:
        # directory = get_directory(node)
        submodules[f'S{submodule_count}'].update([node])
        submodule_count += 1
        # remaining_by_dir[directory].add(node)
        
# for directory, files in remaining_by_dir.items():
#     if files:
#         submodules[f'S{submodule_count}'].update(files)
#         submodule_count += 1

# Print submodules        
# for submodule, files in submodules.items():
    # print(f'{submodule}: {files}')


# %%

# import json

# data = {
#     "directories": {key: list(value) for key, value in directories.items()}
# ,
#     "class_couplings": class_couplings,
# }

# with open("class_dependency_graph_variables.json", "w") as file:
#     json.dump(data, file)


# import matplotlib.pyplot as plt
# import pygraphviz as pgv

# # Create a new graph with ranksep and nodesep adjustments
# NG = pgv.AGraph(directed=True, strict=True, rankdir='LR', splines='curved', ranksep='0.6', nodesep='0.4')

# # Add nodes for each class
# for path, class_ids in directories.items():
#     with NG.subgraph(name="cluster_" + path) as c:
#         x = '/'.join(path.rsplit('/', 6)[1:])
#         c.graph_attr['label'] = x
#         c.graph_attr['penwidth'] = '2'  # Set the border width of the directory box
#         c.graph_attr['rounded'] = 'true'  # Round the corners of the directory box
        
#         for class_id in class_ids:
#             if class_id in inter_coupling_nodes:
#                 c.add_node(class_id, label="C"+str(class_id), color='grey', style='filled', width='1.5', height='1.5', fontsize='26')
#             else:
#                 c.add_node(class_id, label="C"+str(class_id), width='1.5', height='1.5', fontsize='26')

# # Add edges based on class couplings
# for coupling in class_couplings_set:
#     source, _, dest, _ = coupling
#     NG.add_edge(source, dest, len='1.5', weight='1')  # Adjusted the constraint attribute

# Save and visualize the graph
# NG.layout(prog="dot")
# NG.draw("class-level-dependency-graph.png", prog="dot", format='png')


# # %%



# submodule_names = {'S1':'catalog_service','S2':'actions','S3':'domain','S4':'mapper','S5':'other_services','S6':'test'}


# import json

# data = {
#     "submodule_names": submodule_names
# ,
#     "class_couplings": class_couplings,
#     "submodules": {key: list(value) for key, value in submodules.items()},
# }

# with open("submodule_dependency_graph_variables.json", "w") as file:
#     json.dump(data, file)



# Create a new graph with settings
# NG = pgv.AGraph(strict=True, directed=True, rankdir='LR', splines='curved', nodesep=1.0, ranksep=2.0)
# NG.node_attr['shape'] = 'box'

# # Add nodes for each submodule
# for sm_id, sm_name in submodule_names.items():
#     NG.add_node(sm_id, label=f"{sm_id}:{sm_name}")

# Determine submodule dependencies and add edges
# added_edges = set()
# for coupling in class_couplings:
#     src_class, _, dest_class, _ = coupling
#     src_module = [sm for sm, classes in submodules.items() if src_class in classes][0]
#     dest_module = [sm for sm, classes in submodules.items() if dest_class in classes][0]
    
#     if src_module != dest_module and (src_module, dest_module) not in added_edges:
#         NG.add_edge(src_module, dest_module)
#         added_edges.add((src_module, dest_module))

# # Save and visualize the graph
# NG.layout(prog="dot")
# NG.draw("submodule-dependency-graph.png", prog="dot", format='png')

# %%
# import matplotlib.pyplot as plt
# import networkx as nx

# # This function will draw a graph where the node color depends on the submodule
# def draw_graph(G, submodules):
#     plt.figure(figsize=(10, 10))  # Large figure size for more complex graphs

#     # Create a color map where each node gets a color based on its submodule
#     color_map = []
#     for node in G:
#         for i, submodule in enumerate(submodules.values()):
#             if node in submodule:
#                 color_map.append(i)  # Use index for color
#                 break
#         else:
#             color_map.append(len(submodules))  # If node not in any submodule, give it a different color

#     # Generate node sizes based on degrees
#     degrees = [G.degree(n) * 100 for n in G.nodes]

#     # Create a layout for the nodes
#     layout = nx.spring_layout(G, k=3)  # Increase this value to increase node distance (e.g., 0.3)

#     # Draw the graph using the color map, node sizes, and layout
#     nx.draw(G, with_labels=True, node_color=color_map, node_size=degrees, pos=layout)

#     plt.show()

# Call the function with your graph and submodules
# draw_graph(G, submodules)






# %%


from sklearn import preprocessing
import numpy as np


# Create a scaler object
scaler = preprocessing.MinMaxScaler()

# Normalize the matrix
reshaped_array = total_similarity.reshape(-1, 1)

normalized_conseptual_matrix = scaler.fit_transform(reshaped_array)

normalized_conseptual_matrix = normalized_conseptual_matrix.reshape(total_similarity.shape)

# Cosine similarity boundaries (-1,1) and we make it normalize to (0,1). 
# The point is that some similarities are negative.
# So in normalization the relation of each submodule to itself can be nonzero and we should change it to zero in this step.
for index, value in np.ndenumerate(normalized_conseptual_matrix):
    if index[0] == index[1]:
        normalized_conseptual_matrix[index] = 0





reshaped_array = adj_matrix.reshape(-1, 1)


normalized_structural_matrix = scaler.fit_transform(reshaped_array)

normalized_structural_matrix = normalized_structural_matrix.reshape(adj_matrix.shape)


#Coupling mi, mj=w*WSCmi, mj+1-w*WCC(mi, mj)
coupling  = normalized_conseptual_matrix * 0.2 + normalized_structural_matrix * 0.8


# import git


# repo = git.Repo(repo_path)

# commit_history = {}

# # specific_commit_hash = '6fa36aad5fe55571a13b76a85802d31e6e355ed4'
# test_count=0
# # Iterate through each commit
# for commit in repo.iter_commits():
#     changed_files = set()
#     test_count +=1
#     # if commit.hexsha == specific_commit_hash:
#     #     break  # Stop processing further

#     # For each changed file in the commit
#     for item in commit.stats.files.keys():
#         # Assuming Java files; adjust the condition for other languages
#         if item.endswith('.java'):
#             # Extract class name from file name
#             class_name = item.split('/')[-1].replace('.java', '').lower()
#             if class_name in module_indices.values():
#                 changed_files.add(class_name)
#     if len(changed_files) > 1 and len(changed_files) <= len(module_indices)/2: # If one class in a commit changes, doesn't it mean that it is independent?
#         commit_history[commit.hexsha] = changed_files

# # Now commit_history dictionary is populated
# # print(commit_history)


# from itertools import combinations

# # For each pair of classes, count how many times they changed together
# def count_co_changes(commit_history):
#     co_change_count = {}

#     for classes in commit_history.values():
#         for class1, class2 in combinations(classes, 2):
#             if (class1.lower(), class2.lower()) not in co_change_count:
#                 co_change_count[(class1.lower(), class2.lower())] = 0
#             co_change_count[(class1.lower(), class2.lower())] += 1
            
#             if (class2.lower(), class1.lower()) not in co_change_count:
#                 co_change_count[(class2.lower(), class1.lower())] = 0
#             co_change_count[(class2.lower(), class1.lower())] += 1

#     return co_change_count



# co_change_count = count_co_changes(commit_history)

# # Create a matrix initialized with zeros
# matrix_size = len(module_indices)
# matrix = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]

# # Map classes to indices
# # class_to_index = {class_name: index for class_name,index,  in enumerate(module_indices)}

# # Populate the matrix
# for (class1, class2), value in co_change_count.items():
#     row = [index-1 for index,class_name in module_indices.items() if class_name == class1][0]
#     col = [index-1 for index,class_name in module_indices.items() if class_name == class2][0]
#     matrix[row][col] = value

# with open('roller_co_commited_classes.json','w') as file:
#     json.dump({'co_commited_classes':matrix},file, indent=4)

with open('xwiki_coupling.json','w') as file:
    json.dump({'conceptual_coupling_matrix':total_similarity.tolist(),'normalized_conceptual_coupling_matrix':normalized_conseptual_matrix.tolist(),'normalized_structural_coupling_matrix':normalized_structural_matrix.tolist(),'structural_coupling_matrix':adj_matrix.tolist(),'class_indices':module_indices,'class_co_eccurances_in_execution_traces':class_co_eccurances_matrix.tolist() },file, indent=4)

