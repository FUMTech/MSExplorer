#%%
# init constants for database
import mysql.connector
from collections import defaultdict
import networkx as nx


host='127.0.0.1'
user='root'
password='root'
# database = 'java-uuid-generator'
database = 'jpetstore-6.0.2'
# database = 'ftgo-monolith'
port = '3308'
directory_path = "/home/amir/Desktop/PJ/MonoMicro/jpetstore-6-jpetstore-6.0.2"  # Replace with your directory
# directory_path = "/home/amir/Desktop/MetricTool/java-uuid-generator-java-uuid-generator-3.1.5"  # Replace with your directory
# directory_path = "/home/amir/Desktop/PJ/MonoMicro/ftgo-monolith"  # Replace with your directory

# Path to your local git project
repo_path = '/home/amir/Desktop/PJ/MonoMicro/jpetstore-6'
# repo_path = '/home/amir/Desktop/java-uuid-generator'
# %%
# define database functions for connecting,executing queries and printing results
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


# %%
# Extract conseptual coupling features from code + enums feature of structural coupling
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


# %%
import os
import javalang
from collections import defaultdict

import re

# Extract conseptual coupling features from code + enums feature of structural coupling
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
            class_info['SCS_MethodReference'].append(node.member + ":" + ",".join(arg.type.name for arg in node.arguments))  # Source Code Statement

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

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)
                try:
                    java_tree = parse_java_file(file_path)
                    class_info = extract_lexical_information(java_tree)

                    #Section of finding Directories enum usage
                    for class_i in class_info['CN']:
                        new_class_info = [(a_class[0],a_class[1],root)for a_class in all_classes if a_class[1] == class_i.lower()]
                        class_list += new_class_info

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



import numpy as np
from collections import defaultdict

module_indices = {class_name: index for index, class_name,_ in all_classes}

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

# print(adj_matrix)

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

    if in_degree+ out_degree > 5:
        #  inter_coupling_nodes.add((node,in_degree,out_degree))
         inter_coupling_nodes.add(node)


# Filter nodes based on given criteria
# inter_coupling_nodes = {node for node, deg in in_degrees.items() if deg >= 3} & \
#                        {node for node, deg in out_degrees.items() if deg >= 1}

# Sort by in-degree first, then out-degree because of two or more inter-coupling file occurance at the same directory as the class with more relation would take the intra coupling classes first
inter_coupling_nodes = sorted(inter_coupling_nodes, key=lambda node: (in_degrees[node], out_degrees[node]))


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

    # Pattern for license comments
    license_pattern = re.compile(r'/\*\*.*?Licensed under the Apache License.*?limitations under the License\.\s\*/', re.DOTALL)
    copy_right_pattern = re.compile(r'/\*\*.*?Copyright 2010-2016 the original author or authors.\n\.\s\*/', re.DOTALL)

    # Pattern for author signatures
    author_pattern = re.compile(r'/\*\*.*?@author.*?\*/', re.DOTALL)

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

import json

data = {
    "directories": {key: list(value) for key, value in directories.items()}
,
    "class_couplings": class_couplings,
}

with open("class_dependency_graph_variables.json", "w") as file:
    json.dump(data, file)


import matplotlib.pyplot as plt
import pygraphviz as pgv

# Create a new graph with ranksep and nodesep adjustments
NG = pgv.AGraph(directed=True, strict=True, rankdir='LR', splines='curved', ranksep='0.6', nodesep='0.4')

# Add nodes for each class
for path, class_ids in directories.items():
    with NG.subgraph(name="cluster_" + path) as c:
        x = '/'.join(path.rsplit('/', 6)[1:])
        c.graph_attr['label'] = x
        c.graph_attr['penwidth'] = '2'  # Set the border width of the directory box
        c.graph_attr['rounded'] = 'true'  # Round the corners of the directory box
        
        for class_id in class_ids:
            if class_id in inter_coupling_nodes:
                c.add_node(class_id, label="C"+str(class_id), color='grey', style='filled', width='1.5', height='1.5', fontsize='26')
            else:
                c.add_node(class_id, label="C"+str(class_id), width='1.5', height='1.5', fontsize='26')

# Add edges based on class couplings
for coupling in class_couplings_set:
    source, _, dest, _ = coupling
    NG.add_edge(source, dest, len='1.5', weight='1')  # Adjusted the constraint attribute

# Save and visualize the graph
NG.layout(prog="dot")
NG.draw("class-level-dependency-graph.png", prog="dot", format='png')


# %%



submodule_names = {'S1':'catalog_service','S2':'actions','S3':'domain','S4':'mapper','S5':'other_services','S6':'test'}


import json

data = {
    "submodule_names": submodule_names
,
    "class_couplings": class_couplings,
    "submodules": {key: list(value) for key, value in submodules.items()},
}

with open("submodule_dependency_graph_variables.json", "w") as file:
    json.dump(data, file)



# Create a new graph with settings
NG = pgv.AGraph(strict=True, directed=True, rankdir='LR', splines='curved', nodesep=1.0, ranksep=2.0)
NG.node_attr['shape'] = 'box'

# Add nodes for each submodule
for sm_id, sm_name in submodule_names.items():
    NG.add_node(sm_id, label=f"{sm_id}:{sm_name}")

# Determine submodule dependencies and add edges
added_edges = set()
for coupling in class_couplings:
    src_class, _, dest_class, _ = coupling
    src_module = [sm for sm, classes in submodules.items() if src_class in classes][0]
    dest_module = [sm for sm, classes in submodules.items() if dest_class in classes][0]
    
    if src_module != dest_module and (src_module, dest_module) not in added_edges:
        NG.add_edge(src_module, dest_module)
        added_edges.add((src_module, dest_module))

# Save and visualize the graph
NG.layout(prog="dot")
NG.draw("submodule-dependency-graph.png", prog="dot", format='png')

# %%
import matplotlib.pyplot as plt
import networkx as nx

# This function will draw a graph where the node color depends on the submodule
def draw_graph(G, submodules):
    plt.figure(figsize=(10, 10))  # Large figure size for more complex graphs

    # Create a color map where each node gets a color based on its submodule
    color_map = []
    for node in G:
        for i, submodule in enumerate(submodules.values()):
            if node in submodule:
                color_map.append(i)  # Use index for color
                break
        else:
            color_map.append(len(submodules))  # If node not in any submodule, give it a different color

    # Generate node sizes based on degrees
    degrees = [G.degree(n) * 100 for n in G.nodes]

    # Create a layout for the nodes
    layout = nx.spring_layout(G, k=3)  # Increase this value to increase node distance (e.g., 0.3)

    # Draw the graph using the color map, node sizes, and layout
    nx.draw(G, with_labels=True, node_color=color_map, node_size=degrees, pos=layout)

    plt.show()

# Call the function with your graph and submodules
draw_graph(G, submodules)






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

with open('coupling.json','w') as file:
    json.dump({'conceptual':total_similarity.tolist(),'normalized_conseptual_coupling_matrix':normalized_conseptual_matrix.tolist(),'normalized_structural_coupling_matrix':normalized_structural_matrix.tolist(),'submodules': {key: list(value) for key, value in submodules.items()},'class_id_to_name':class_id_to_name},file, indent=4)

# %% 

# %%

def find_IFN(set_dictionary, microservices, interface_relationships):
    counter = 0
    set_of_microservices =[]
    # Create a mapping from class to submodule
    class_to_submodule = {cls: submodule for submodule, classes in set_dictionary.items() for cls in classes}

    for class1, interface1, class2, interface2 in interface_relationships:
        # Find the microservices that contain each class
        class1_microservice = [ms for ms in microservices if class_to_submodule[class1] in ms][0]
        class2_microservice = [ms for ms in microservices if class_to_submodule[class2] in ms][0]

        # If there is no overlap between the microservices of the two classes, increment the counter
        if not set(class1_microservice) & set(class2_microservice):
            if (class1_microservice,class2_microservice) not in set_of_microservices:
                set_of_microservices.append((class1_microservice,class2_microservice))
            counter += 1
    len_of_unique_microservice_published_interface = len(set_of_microservices)

    if len_of_unique_microservice_published_interface == 0:
        return 0
    else:
        return counter/ len_of_unique_microservice_published_interface



# IFN = find_IFN(set_dictionary=submodules,microservices=candidate_state,interface_relationships=find_interface_relations(class_couplings))
# print("IFN = ",IFN)



# %%
import javalang

# Helper function to compute intersection over union
def iou(set1, set2):
    if not set1 and not set2:
        return 1
    return len(set1.intersection(set2)) / len(set1.union(set2))

# Extract method signatures from Java source code
def extract_method_signatures(java_code):
    tree = javalang.parse.parse(java_code)
    signatures = []

    for _, type_decl in tree.filter(javalang.tree.TypeDeclaration):
        for _, method in type_decl.filter(javalang.tree.MethodDeclaration):
            input_params = [param.type.name for param in method.parameters]
            return_type = method.return_type.name if method.return_type else None
            signatures.append((set(input_params), return_type))

    return signatures

# Compute the f_msg value for a pair of interfaces
def compute_fmsg(signatures1, signatures2):
    total_param_similarity = 0
    total_return_value_similarity = 0

    for (input_params1, return_type1) in signatures1:
        for (input_params2, return_type2) in signatures2:
            total_param_similarity += iou(input_params1, input_params2)
            total_return_value_similarity += iou(set([return_type1]), set([return_type2]))

    param_similarity = total_param_similarity / (len(signatures1) * len(signatures2))
    return_value_similarity = total_return_value_similarity / (len(signatures1) * len(signatures2))
    
    return (param_similarity + return_value_similarity) / 2

def find_CHM(candidate_microservices, interface_relationships):
    microservice_chms = []

    def get_interface_code(interface_id):
        for file,info in lexical_info.items():
            for class_name in info['CN']:
                if  class_name.lower() == class_id_to_name[interface_id]:
                    return info['CODE']

    for microservice in candidate_microservices:
        fmsg_values = []

        # Initialize a set to store all class IDs for the current microservice
        class_ids_for_microservice = set()
        for submodule in microservice:
            # Update the set with class IDs for each submodule in the microservice
            class_ids_for_microservice.update(submodules.get(submodule, set()))

        # Filter out interfaces related to the current microservice
        microservice_interfaces = [interface for interface in interfaces if interface in class_ids_for_microservice]

        for i in range(len(microservice_interfaces)):
            first_interface_code = get_interface_code(microservice_interfaces[i])
            if first_interface_code == '':
                continue
            signatures1 = extract_method_signatures(first_interface_code)
            for j in range(i+1, len(microservice_interfaces)):
                second_interface_code = get_interface_code(microservice_interfaces[j])
                if second_interface_code == '':
                    continue
                signatures2 = extract_method_signatures(second_interface_code)
                fmsg_values.append(compute_fmsg(signatures1, signatures2))
        
        if fmsg_values:
            microservice_chms.append(sum(fmsg_values) / len(fmsg_values)) # What should I do for Oj in formula?

    # Calculate the average CHM across all microservices
    CHM = sum(microservice_chms) / len(microservice_chms) if microservice_chms else 0 # What does N mean in here?
    return CHM

# Assuming you have a dictionary mapping class IDs to their respective Java source codes




# CHM = find_CHM(candidate_microservices=candidate_state, interface_relationships=find_interface_relations(class_couplings) )
# print("CHM = ",CHM)


# %%

def extract_domain_terms_from_interface(java_code):
    #Extract domain terms (method names, parameter names, return types) from a Java interface
    tree = javalang.parse.parse(java_code)
    domain_terms = set()

    for _, type_decl in tree.filter(javalang.tree.TypeDeclaration):
        for _, method in type_decl.filter(javalang.tree.MethodDeclaration):
            # Add method name
            domain_terms.add(method.name)
            # Add parameter names
            domain_terms.update([param.name for param in method.parameters])
            # Add return type if exists
            if method.return_type:
                domain_terms.add(method.return_type.name)

    return domain_terms

def compute_fdom(terms1, terms2):
    #Compute fdom value for a pair of interfaces based on their domain terms
    return iou(terms1, terms2)

def find_CHD(candidate_microservices, interface_relationships):
    microservice_chds = []

    def get_interface_code(interface_id):
        for file, info in lexical_info.items():
            for class_name in info['CN']:
                if class_name.lower() == class_id_to_name[interface_id]:
                    return info['CODE']

    for microservice in candidate_microservices:
        fdom_values = []

        class_ids_for_microservice = set()
        for submodule in microservice:
            class_ids_for_microservice.update(submodules.get(submodule, set()))

        microservice_interfaces = [interface for interface in interfaces if interface in class_ids_for_microservice]

        for i in range(len(microservice_interfaces)):
            first_interface_code = get_interface_code(microservice_interfaces[i])
            if first_interface_code == '':
                continue
            terms1 = extract_domain_terms_from_interface(first_interface_code)
            for j in range(i+1, len(microservice_interfaces)):
                second_interface_code = get_interface_code(microservice_interfaces[j])
                if second_interface_code == '':
                    continue
                terms2 = extract_domain_terms_from_interface(second_interface_code)
                fdom_values.append(compute_fdom(terms1, terms2))
        
        num_interfaces = len(interfaces)
        if num_interfaces > 1 and fdom_values:
            avg_fdom = sum(fdom_values) / (num_interfaces * (num_interfaces - 1) / 2)
            microservice_chds.append(avg_fdom)

    # Calculate the average CHD across all microservices
    CHD = sum(microservice_chds) / len(microservice_chds) if microservice_chds else 0
    return CHD

# CHD = find_CHD(candidate_microservices=candidate_state, interface_relationships=find_interface_relations(class_couplings))
# print("CHD = ",CHD)


# %%
from itertools import combinations

def smq(microservices, graph):

    mq = 0

    for ms in microservices:

        # Initialize a set to store all class IDs for the current microservice
        class_ids_for_microservice = set()
        for submodule in ms:
            # Update the set with class IDs for each submodule in the microservice
            class_ids_for_microservice.update(submodules.get(submodule, set()))

        nodes = [n for n in graph if n in class_ids_for_microservice]
        subgraph = graph.subgraph(nodes)
        edges = subgraph.edges()
        
        intra_edges = [e for e in edges if e[0] in class_ids_for_microservice and e[1] in class_ids_for_microservice] #Includes self calling

        mq += len(intra_edges) / len(nodes)**2

    mq /= len(microservices)

    for a, b in combinations(microservices, 2):

        # Initialize a set to store all class IDs for the current microservice
        class_ids_for_microservice_1 = set()
        for submodule in a:
            # Update the set with class IDs for each submodule in the microservice
            class_ids_for_microservice_1.update(submodules.get(submodule, set()))

        # Initialize a set to store all class IDs for the current microservice
        class_ids_for_microservice_2 = set()
        for submodule in b:
            # Update the set with class IDs for each submodule in the microservice
            class_ids_for_microservice_2.update(submodules.get(submodule, set()))

        inter_edges = [e for e in graph.edges() 
                    if (e[0] in class_ids_for_microservice_1 and e[1] in class_ids_for_microservice_2) or 
                        (e[0] in class_ids_for_microservice_2 and e[1] in class_ids_for_microservice_1)]
        
        mq -= len(inter_edges) / (len(class_ids_for_microservice_1) * len(class_ids_for_microservice_2))

    if len(microservices) == 1:
        return 1

    mq /= len(microservices) * (len(microservices) - 1) / 2
    
    return mq

# SMQ = smq(candidate_state,G)
# print("SMQ = ",SMQ)


# %%

def extract_domain_terms_from_class(java_code): #Better to be caclulated by weights
    """Extract domain terms from a Java class."""
    tree = javalang.parse.parse(java_code)
    domain_terms = set()

    for _, type_decl in tree.filter(javalang.tree.TypeDeclaration):
        # Add class name
        domain_terms.add(type_decl.name)
        
        # Add annotations
        if type_decl.annotations:
            domain_terms.update([annotation.name for annotation in type_decl.annotations])


            # 0:
            # 'modifiers'
            # 1:
            # 'annotations'
            # 2:
            # 'documentation'
            # 3:
            # 'name'
            # 4:
            # 'body'
            # 5:
            # 'type_parameters'
            # 6:
            # 'extends'
            # 7:
            # 'implements'

        for _, method in type_decl.filter(javalang.tree.MethodDeclaration):
            # Add method name
            domain_terms.add(method.name)
            
            # Add thrown exceptions
            if method.throws:
                domain_terms.update(method.throws)

            # Add parameter names and types
            domain_terms.update([param.name for param in method.parameters])
            domain_terms.update([param.type.name for param in method.parameters if param.type])
            
            # Add return type if exists
            if method.return_type:
                domain_terms.add(method.return_type.name)

            # Add local variables from the method
            for _, local_var in method.filter(javalang.tree.LocalVariableDeclaration):
                domain_terms.update([decl.name for decl in local_var.declarators])

            # 0:
            # 'documentation'
            # 1:
            # 'modifiers'
            # 2:
            # 'annotations'
            # 3:
            # 'type_parameters'
            # 4:
            # 'return_type'
            # 5:
            # 'name'
            # 6:
            # 'parameters'
            # 7:
            # 'throws'
            # 8:
            # 'body'

        # Add field names and types
        for _, field in type_decl.filter(javalang.tree.FieldDeclaration):
            domain_terms.update([field_decl.name for field_decl in field.declarators])
            if field.type:
                domain_terms.add(field.type.name)


            # 0:
            # 'documentation'
            # 1:
            # 'modifiers'
            # 2:
            # 'annotations'
            # 3:
            # 'type'
            # 4:
            # 'declarators'

        # Extract comments tooks long time(most spending in comparison to other parts)
        for _, comment in type_decl.filter(javalang.tree.Documented):
            if comment.documentation is not None:
                domain_terms.update(comment.documentation.split('\n'))

    return domain_terms


def compute_cohesion(classes, get_class_code):
    total_links = 0
    for i in range(len(classes)):
        class_code_i = get_class_code(classes[i])
        terms_i = extract_domain_terms_from_class(class_code_i)
        for j in range(i + 1, len(classes)):
            class_code_j = get_class_code(classes[j])
            terms_j = extract_domain_terms_from_class(class_code_j)
            total_links += iou(terms_i, terms_j)
    num_classes = len(classes)
    return total_links / (num_classes * (num_classes - 1) / 2) if num_classes > 1 else 0

def compute_coupling(classes_m, classes_n, get_class_code):
    total_links = 0
    for class_id_m in classes_m:
        class_code_m = get_class_code(class_id_m)
        terms_m = extract_domain_terms_from_class(class_code_m)
        for class_id_n in classes_n:
            class_code_n = get_class_code(class_id_n)
            terms_n = extract_domain_terms_from_class(class_code_n)
            total_links += iou(terms_m, terms_n)
    return total_links / (len(classes_m) * len(classes_n))

def find_CMQ(candidate_microservices):
    N = len(candidate_microservices)
    cohesion_values = []
    coupling_values = []

    def get_class_code(class_id):
        # This function retrieves the code for a given class_id
        for file, info in lexical_info.items():
            for class_name in info['CN']:
                if class_name.lower() == class_id_to_name[class_id]:
                    return info['CODE']

    for m in candidate_microservices:
        classes_m = set()
        for submodule in m:
            classes_m.update(submodules.get(submodule, set()))
        cohesion_values.append(compute_cohesion(list(classes_m), get_class_code))

        for n in candidate_microservices:
            if m != n:
                classes_n = set()
                for submodule in n:
                    classes_n.update(submodules.get(submodule, set()))
                coupling_values.append(compute_coupling(list(classes_m), list(classes_n), get_class_code))

    CMQ = sum(cohesion_values)/N - sum(coupling_values)/(N*(N-1)/2)
    return CMQ

# CMQ = find_CMQ(candidate_microservices=candidate_state)
# print("CMQ = ",CMQ)


# %%
import git


repo = git.Repo(repo_path)

commit_history = {}

# Iterate through each commit
for commit in repo.iter_commits():
    changed_files = set()

    # For each changed file in the commit
    for item in commit.stats.files.keys():
        # Assuming Java files; adjust the condition for other languages
        if item.endswith('.java'):
            # Extract class name from file name
            class_name = item.split('/')[-1].replace('.java', '')
            changed_files.add(class_name)
    if len(changed_files) > 1 : # If one class in a commit changes, doesn't it mean that it is independent?
        commit_history[commit.hexsha] = changed_files

# Now commit_history dictionary is populated
# print(commit_history)


from itertools import combinations

# For each pair of classes, count how many times they changed together
def count_co_changes(commit_history):
    co_change_count = {}

    for classes in commit_history.values():
        for class1, class2 in combinations(classes, 2):
            if (class1.lower(), class2.lower()) not in co_change_count:
                co_change_count[(class1.lower(), class2.lower())] = 0
            co_change_count[(class1.lower(), class2.lower())] += 1
            
            if (class2.lower(), class1.lower()) not in co_change_count:
                co_change_count[(class2.lower(), class1.lower())] = 0
            co_change_count[(class2.lower(), class1.lower())] += 1

    return co_change_count



def calculate_ICF(microservices, commit_history):
    co_change_count = count_co_changes(commit_history)
    total_icfm = 0

    for microservice in microservices:

        # Initialize a set to store all class IDs for the current microservice
        classes = set()
        for submodule in microservice:
            # Update the set with class IDs for each submodule in the microservice
            classes.update(submodules.get(submodule, set()))

        icfm = 0

        if len(microservice) > 1:
            for class1, class2 in combinations(classes, 2):
                icfm += co_change_count.get((class_id_to_name[class1], class_id_to_name[class2]), 0)
        else:
            icfm = 1

        icfm /= len(classes) ** 2
        total_icfm += icfm

    ICF = total_icfm / len(microservices)
    return ICF



# ICF = calculate_ICF(candidate_state, commit_history)
# print("ICF = ",ICF)


# %%

# Assuming commit_history is already populated



def compute_ecf(microservices, co_changes):

    total_ecfm = 0

    all_classes = set()
    for microservice in microservices:
        for submodule in microservice:
            # Update the set with class IDs for each submodule in the microservice
            all_classes.update(submodules.get(submodule, set()))

    for microservice in microservices:
        Cm = set()
        for submodule in microservice:
            # Update the set with class IDs for each submodule in the microservice
            Cm.update(submodules.get(submodule, set()))

        Cm_prime = all_classes - Cm
        sum_f_cmt = 0

        for ci in Cm:
            for cj in Cm_prime:
                pair = tuple(sorted([class_id_to_name[ci],class_id_to_name[cj]]))
                sum_f_cmt += co_changes.get(pair, 0)

        ecfm = (1 / len(Cm)) * (1 / len(Cm_prime)) * sum_f_cmt
        total_ecfm += ecfm

    ECF = total_ecfm / len(microservices)
    return ECF



co_change_count = count_co_changes(commit_history)
# ECF = compute_ecf(candidate_state, co_change_count)

# print("ECF = ",ECF)



# REI = ECF / ICF

# print("REI = ",REI)



# %%

import numpy as np

def get_coupling(submodule1, submodule2):
    # Assuming the submodules are represented as strings like "S1", "S2", etc.
    # Extract the indices from the submodule names
    index1 = int(submodule1[1:]) - 1
    index2 = int(submodule2[1:]) - 1

    # Retrieve the coupling from the coupling matrix
    output = coupling[index1][index2]

    return output

def calculate_bmc_imc(coupling, state):
    n = len(state) 
    bmc = 0
    imc = 0
    between_max = 0
    within_max = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                modules_i = state[i] 
                modules_j = state[j]
                
                # Calculate between-module coupling
                between_sum = 0
                for m1 in modules_i:
                    for m2 in modules_j:
                        c = get_coupling(m1, m2)
                        between_sum += c
                        between_max = max(between_max, c)
                bmc += between_sum
                
                # Calculate within-module coupling                
                within_sum = 0
                for m1 in modules_i:
                    for m2 in modules_i:
                        if m1 != m2:
                            c = get_coupling(m1, m2)
                            within_sum += c
                            within_max = max(within_max, c)
                imc += within_sum 
                
    if len(state) == 1:
        # Special case - single module
        bmc = 0 
        imc = 1
    else:
        # Normal case        
        bmc =  bmc / (2 * between_max) if between_max != 0 else 0
        imc = imc / (2 * within_max) if within_max != 0 else 0
    
    return bmc, imc

initial_state = [{submodule} for submodule in submodules.keys()]

bmc, imc = calculate_bmc_imc(coupling, initial_state)  

# print(bmc,imc)

# %%
initial_temperature = 100  # for example
max_iterations = 200000
import random
import math
def simulated_annealing(initial_state, energy_function, neighbourhood_function, annealing_schedule):
    current_state = initial_state
    current_energy = energy_function(current_state)
    current_temperature = initial_temperature
    iterations = 0

    while current_temperature > 1e-100 and iterations < max_iterations:
        neighbour = neighbourhood_function(current_state)
        neighbour_energy = energy_function(neighbour)
        iterations += 1

        # If the neighbouring state is better, accept it
        # If it's worse, accept it with a probability dependent on the temperature and the energy difference
        if (neighbour_energy > current_energy) :
            current_state = neighbour
            current_energy = neighbour_energy

        # Decrease the temperature
        current_temperature = annealing_schedule(current_temperature,iterations)

    return current_state


# %%
def find_number_of_classes_within_microservice(microservice):
    sum =0
    for submodule in microservice:
        classes = submodules.get(submodule)
        sum += len(classes)
    return sum


# %%
import math

def get_classes_count():
    cnx = connect_to_database(host, user, password, database,port)
        
    query = '''
SELECT COUNT(*)
from list_class lc

        '''
    results = execute_query(cnx, query)
    close_database_connection(cnx)
    return results[0][0]


all_classes = get_classes_count()



# %%
from itertools import combinations

cost_array = []
def energy_function(state):
    # Initialize total cohesion and size (for MSI calculation)
    total_size = 0

    # Calculate s_avg sigma
    for microservice in state:
        classes_count = find_number_of_classes_within_microservice(microservice)
        total_size += classes_count **2 # comment later 

    # Calculate average module size and MSI
    s_avg = total_size / all_classes #sigma len(microservice)**2(means number of classes in it) /n
    s_star = all_classes / (0.1 * all_classes)  # assuming m* is 10% of total classes
    w = 0.05  # penalty factor
    MSI = math.exp(-0.5 * ((math.log(s_avg) - math.log(s_star)) / (w * math.log(all_classes))) ** 2)

    BMCI, IMCI = calculate_bmc_imc(coupling, state)  


    # Define how to combine cohesion and MSI into a single cost
    # This could be a simple sum, a weighted sum, a product, etc.
    cost = (IMCI/(IMCI+BMCI))**0.5 *   MSI**0.5  # Assumed alfa and beta coefficients 1.

    #eval metrics
    # curr_smq=smq(state,G)
    # smq_list.append(curr_smq)

    IFN = find_IFN(set_dictionary=submodules,microservices=state,interface_relationships=find_interface_relations(class_couplings))

    CHM = find_CHM(candidate_microservices=state, interface_relationships=find_interface_relations(class_couplings) )

    CHD = find_CHD(candidate_microservices=state, interface_relationships=find_interface_relations(class_couplings))

    
    SMQ = smq(state,G)

    ICF = calculate_ICF(state, commit_history)

    ECF = compute_ecf(state, co_change_count)

    cost_array.append({str(state) :(cost,IFN,CHM,CHD,SMQ,ICF,ECF)  })
    # print("State=",state)
    # print("Cost=",cost)

    return cost


# %%
initial_state = [{submodule} for submodule in submodules.keys()]


# %%
import random

def neighbourhood_function(state):
    # Copy the state to not modify the original one
    neighbour = [set(microservice) for microservice in state]

    if len(neighbour) < 2:
        return neighbour

    # Select two different microservices
    microservice1, microservice2 = random.sample(neighbour, 2)

    # Randomly decide whether to move a submodule or swap two submodules
    if random.uniform(0, 1) <= 1:  # 50% chance to move a submodule
        if microservice1 and microservice2:  # Ensure neither microservice is empty
            # Select a random submodule from the first microservice to move to the second one
            submodule_to_move = random.sample(microservice1, 1)[0]
            microservice1.remove(submodule_to_move)
            microservice2.add(submodule_to_move)
    else:  # 50% chance to swap two submodules
        if len(microservice1) > 1 and len(microservice2) > 1:  # Ensure both microservices have at least two submodules
            # Select a random submodule from each microservice
            submodule1 = random.sample(microservice1, 1)[0]
            submodule2 = random.sample(microservice2, 1)[0]
            # Swap the submodules
            microservice1.remove(submodule1)
            microservice2.remove(submodule2)
            microservice1.add(submodule2)
            microservice2.add(submodule1)

    # Remove empty microservices
    neighbour = [microservice for microservice in neighbour if microservice]

    return neighbour



# %%

def annealing_schedule(temperature,iteration):
    return temperature * 0.99**iteration


# %%
candidate_state = simulated_annealing(initial_state=initial_state,energy_function=energy_function,neighbourhood_function=neighbourhood_function,annealing_schedule=annealing_schedule)
print("Candidate State = ",candidate_state)


# Find the state with the maximum cost

max_cost_state = None
max_cost = float('-inf')  # Initialize with negative infinity
max_IFN = float('+inf')  # Initialize with negative infinity
max_CHM = float('-inf')  # Initialize with negative infinity
max_CHD = float('-inf')  # Initialize with negative infinity
max_SMQ = float('-inf')  # Initialize with negative infinity
max_ICF = float('-inf')  # Initialize with negative infinity
max_ECF = float('+inf')  # Initialize with negative infinity
max_IFN_state = None  # Initialize with negative infinity
max_CHM_state = None  # Initialize with negative infinity
max_CHD_state = None  # Initialize with negative infinity
max_SMQ_state = None  # Initialize with negative infinity
max_ICF_state = None  # Initialize with negative infinity
max_ECF_state = None  # Initialize with negative infinity



for entry in cost_array:
    for state, values in entry.items():
        cost = values[0]
        IFN = values[1]
        CHM = values[2]
        CHD = values[3]
        SMQ = values[4]
        ICF = values[5]
        ECF = values[6]
        
        if cost > max_cost:
            max_cost = cost
            max_cost_state = state
        if IFN < max_IFN:
            max_IFN = cost
            max_IFN_state = state
        if CHM > max_CHM:
            max_CHM = cost
            max_CHM_state = state
        if CHD > max_CHD:
            max_CHD = cost
            max_CHD_state = state
        if SMQ > max_SMQ:
            max_SMQ = cost
            max_SMQ_state = state
        if ICF > max_ICF:
            max_ICF = cost
            max_ICF_state = state
        if ECF < max_ECF:
            max_ECF = cost
            max_ECF_state = state

# Output the result

if max_cost_state is not None:
    print(f"The state with the maximum cost is {max_cost_state} with cost {max_cost}.")

if max_IFN is not None:
    print(f"The state with the min IFN is {max_IFN} with cost {max_IFN_state}.")

if max_CHM is not None:
    print(f"The state with the maximum max_CHM is {max_CHM} with cost {max_CHM_state}.")

if max_CHD is not None:
    print(f"The state with the maximum max_CHD is {max_CHD} with cost {max_CHD_state}.")

if max_SMQ is not None:
    print(f"The state with the maximum max_SMQ is {max_SMQ} with cost {max_SMQ_state}.")

if max_ICF is not None:
    print(f"The state with the maximum max_ICF is {max_ICF} with cost {max_ICF_state}.")

if max_ECF is not None:
    print(f"The state with the min ECF is {max_ECF} with cost {max_ECF_state}.")
# from multiprocessing import Pool

# def parallel_simulated_annealing(n_processes, initial_states, energy_function, neighbourhood_function, annealing_schedule):
#     # Create a pool of worker processes
#     with Pool(n_processes) as pool:
#         # Run simulated annealing in parallel with different initial statessolutions = parallel_simulated_annealing(4, initial_states, energy_function, neighbourhood_function, annealing_schedule)

#         solutions = pool.starmap(simulated_annealing, [([initial_state], energy_function, neighbourhood_function, annealing_schedule) for initial_state in initial_states])

#     return solutions

# solutions = parallel_simulated_annealing(4, initial_state, energy_function, neighbourhood_function, annealing_schedule)


