import mysql.connector
from collections import defaultdict
import networkx as nx


host='127.0.0.1'
user='root'
password='root'
database = 'java-uuid-generator'
port = '3308'

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


if __name__ == '__main__':

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
    '''
    has_parameter_results = execute_query(cnx, has_parameter_query)

    print(has_parameter_results)
    print('------------------------------has_parameter---------------------------------')
    
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

    '''
    is_of_type_results = execute_query(cnx, is_of_type_query)
    #removing records of system data type have to be added

    print(is_of_type_results)
    print('-------------------------------is_of_type--------------------------------')
    
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

    where at.class_id != lm.method_class_id
    '''
    referece_results = execute_query(cnx, referece_query)

    print(referece_results)
    print('------------------------------referece---------------------------------')

    
    call_query = '''
    SELECT  
        mcr.class_id as source_class_id,
        lc1.package_name,
    #    lc1.class_name AS source_class_name,
        lm.method_class_id  AS referenced_class_id,
        lc2.package_name
    #    lc2.class_name  AS referenced_class_name
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
    call_results = execute_query(cnx, call_query)


    print(call_results)
    print('-----------------------------call----------------------------------')

    
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
    implement_results = execute_query(cnx, implement_query)

    print(implement_results)
    print('----------------------------implement-----------------------------------')

    
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
    return_results = execute_query(cnx, return_query)

    print(return_results)
    print('--------------------------------return-------------------------------')

    
    inheritance_query = '''
SELECT 
    ls.class_id,
    ls.package_name,
#    ls.class_name,
    lc.class_id,
    lc.package_name
 #   replace(jsontable.class_parents,'>','') as class_parents
FROM 
(SELECT ls.class_id, ls.class_name,ls.package_name, jsontable.class_parents
FROM list_class AS ls
CROSS JOIN JSON_TABLE(CONCAT('["', REPLACE(class_parents, ',', '","'), '"]'),
                      '$[*]' COLUMNS (class_parents TEXT PATH '$')) jsontable
WHERE jsontable.class_parents <> '') as ls

CROSS JOIN JSON_TABLE(CONCAT('["', REPLACE(class_parents, '<', '","'), '"]'),
                      '$[*]' COLUMNS (class_parents TEXT PATH '$')) jsontable
join list_class lc on replace(jsontable.class_parents,'>','') = lc.class_name
        '''
    inheritance_results = execute_query(cnx, inheritance_query)

    print(inheritance_results)
    print('-------------------------------inheritance--------------------------------')


    close_database_connection(cnx)


class_couplings = inheritance_results + return_results + implement_results + call_results + referece_results + is_of_type_results + has_parameter_results 

# indegree = {}
# outdegree = {}
# file_connections = defaultdict(set)

# for pair in class_couplings:
#     source_id, source_module, ref_id, ref_module = pair
    
#     if source_module != ref_module:

#         outdegree[source_id] = outdegree.get(source_id, 0) + 1
        
#         indegree[ref_id] = indegree.get(ref_id, 0) + 1

#   # Keep track of connections within a module
#     if source_module == ref_module:
#         file_connections[source_module].add(source_id)
#         file_connections[source_module].add(ref_id)



# inter_coupling_files = {k for k, v in indegree.items() if v >= 3}
# inter_coupling_files.update({k for k, v in outdegree.items() if v >= 1})

# print('Inter-coupling files:', inter_coupling_files)


# # Initialize dictionaries for submodules
# submodules = defaultdict(list)


# # Iterate over each pair in the list again to create submodules
# for pair in class_couplings:
#     source_id, source_module, ref_id, ref_module = pair
    
#     # Check if the pair is inter-coupling or intra-coupling
#     if source_id in inter_coupling_files or ref_id in inter_coupling_files:
#         # Create a new submodule if not exists
#         if source_module not in submodules or source_id not in submodules[source_module]:
#             submodules[source_module].append({source_id, ref_id})

#     else:
#         # Add to the remaining files submodule
#         if source_module not in submodules or source_id not in submodules[source_module]:
#             submodules[source_module].append({source_id})

# # Print submodules
# for module, submods in submodules.items():
#     print(f'Module: {module}')
#     for i, submod in enumerate(submods, 1):
#         print(f'Submodule S{i}: {submod}')

import networkx as nx
from collections import defaultdict

# Create the directed graph
G = nx.DiGraph()
G_intra = nx.DiGraph()  # Graph for intra-coupling only

for pair in class_couplings:
    source_id, source_module, ref_id, ref_module = pair
    G.add_edge(source_id, ref_id)  # Add all edges to the graph
    if source_module == ref_module:  # Add intra-coupling edges to the intra-coupling graph
        G_intra.add_edge(source_id, ref_id)

# Compute in-degree and out-degree per node
in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())

# Filter nodes based on given criteria
inter_coupling_nodes = {node for node, deg in in_degrees.items() if deg > 3} | \
                       {node for node, deg in out_degrees.items() if deg > 1}

# Create submodules
submodule_count = 1
submodules = defaultdict(set)  # use set to avoid duplicates
nodes_in_submodules = set()  # Keep track of nodes that are already in submodules

for node in inter_coupling_nodes:
    if node in G_intra:  # Only process the node if it exists in G_intra
        # Perform a depth-first search starting from the node on the intra-coupling graph
        related_files = list(nx.dfs_preorder_nodes(G_intra, node))

        # Include only nodes that are not already in other submodules
        related_files = [file for file in related_files if file not in nodes_in_submodules]
        nodes_in_submodules.update(related_files)

        if related_files:  # Only create a new submodule if there are nodes to add
            submodules[f'S{submodule_count}'].update(related_files)
            submodule_count += 1

# Create separate submodules for remaining files
for node in G.nodes():
    if node not in nodes_in_submodules:
        if node in G_intra:  # Only process the node if it exists in G_intra
            related_files = list(nx.dfs_preorder_nodes(G_intra, node))
            
            # Include only nodes that are not already in other submodules
            related_files = [file for file in related_files if file not in nodes_in_submodules]
            nodes_in_submodules.update(related_files)
            
            if related_files:  # Only create a new submodule if there are nodes to add
                submodules[f'S{submodule_count}'].update(related_files)
                submodule_count += 1
        else:
            submodules[f'S{submodule_count}'].add(node)
            submodule_count += 1

for submodule, files in submodules.items():
    print(f'{submodule}: {files}')







import numpy as np
from collections import defaultdict

module_indices = {module: index for index, module in enumerate(submodules)}

n = len(submodules)
adj_matrix = np.zeros((n, n), dtype=int)

class_to_module = {class_: module for module, classes in submodules.items() for class_ in classes}



#inheritance_results + return_results + implement_results + call_results + referece_results + is_of_type_results + has_parameter_results 


for source_class_id,source_module_name, referenced_class_id,referenced_module_name in inheritance_results:
    module1 = class_to_module.get(source_class_id)
    module2 = class_to_module.get(referenced_class_id)
    if module1 is not None and module2 is not None:
        i = module_indices[module1]
        j = module_indices[module2]
                # If a relationship is going out from the class, assign 2
        adj_matrix[i, j] += 2
        # If a relationship is coming into the class, assign 3
        adj_matrix[j, i] += 3

for source_class_id,source_module_name, referenced_class_id,referenced_module_name in return_results:
    module1 = class_to_module.get(source_class_id)
    module2 = class_to_module.get(referenced_class_id)
    if module1 is not None and module2 is not None:
        i = module_indices[module1]
        j = module_indices[module2]
                # If a relationship is going out from the class, assign 2
        adj_matrix[i, j] += 5
        # If a relationship is coming into the class, assign 3
        adj_matrix[j, i] += 7


for source_class_id,source_module_name, referenced_class_id,referenced_module_name in implement_results:
    module1 = class_to_module.get(source_class_id)
    module2 = class_to_module.get(referenced_class_id)
    if module1 is not None and module2 is not None:
        i = module_indices[module1]
        j = module_indices[module2]
                # If a relationship is going out from the class, assign 2
        adj_matrix[i, j] += 11
        # If a relationship is coming into the class, assign 3
        adj_matrix[j, i] += 13


for source_class_id,source_module_name, referenced_class_id,referenced_module_name in call_results:
    module1 = class_to_module.get(source_class_id)
    module2 = class_to_module.get(referenced_class_id)
    if module1 is not None and module2 is not None:
        i = module_indices[module1]
        j = module_indices[module2]
                # If a relationship is going out from the class, assign 2
        adj_matrix[i, j] += 17
        # If a relationship is coming into the class, assign 3
        adj_matrix[j, i] += 19

for source_class_id,source_module_name, referenced_class_id,referenced_module_name in referece_results:
    module1 = class_to_module.get(source_class_id)
    module2 = class_to_module.get(referenced_class_id)
    if module1 is not None and module2 is not None:
        i = module_indices[module1]
        j = module_indices[module2]
                # If a relationship is going out from the class, assign 2
        adj_matrix[i, j] += 23
        # If a relationship is coming into the class, assign 3
        adj_matrix[j, i] += 29

for source_class_id,source_module_name, referenced_class_id,referenced_module_name in is_of_type_results:
    module1 = class_to_module.get(source_class_id)
    module2 = class_to_module.get(referenced_class_id)
    if module1 is not None and module2 is not None:
        i = module_indices[module1]
        j = module_indices[module2]

        # If a relationship is going out from the class, assign 2
        adj_matrix[i, j] += 31
        # If a relationship is coming into the class, assign 3
        adj_matrix[j, i] += 37

for source_class_id,source_module_name, referenced_class_id,referenced_module_name in has_parameter_results:
    module1 = class_to_module.get(source_class_id)
    module2 = class_to_module.get(referenced_class_id)
    if module1 is not None and module2 is not None:
        i = module_indices[module1]
        j = module_indices[module2]

        # If a relationship is going out from the class, assign 2
        adj_matrix[i, j] += 41
        # If a relationship is coming into the class, assign 3
        adj_matrix[j, i] += 43

print(adj_matrix)