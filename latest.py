# ! pip install --upgrade mysql-connector-python javalang
# ! pip install transformers torch torchvision torchaudio sentence-transformers python-Levenshtein

import mysql.connector
from collections import defaultdict
import networkx as nx
import pandas as pd
import numpy as np


# directory_path = "/home/amir/Desktop/PJ/MonoMicro/agilefant-3.5.4"  # Replace with your directory
# directory_path = "/home/amir/Desktop/PJ/MonoMicro/projects/SpringBlog"  # Replace with your directory
# directory_path = "/home/amir/Desktop/PJ/MonoMicro/jpetstore-6-jpetstore-6.0.2"  # Replace with your directory
# directory_path = "/home/ec2-user/SageMaker/mono2micro/xwiki-platform"  # Replace with your directory
directory_path = "/home/amir/Desktop/PJ/MonoMicro/xwiki-platform-xwiki-platform-10.8"  # Replace with your directory
# traces_file_path = '/home/amir/Desktop/PJ/MonoMicro/FoSCI-master/traces/jpetstore.txt'
# traces_file_path = '/home/amir/Desktop/PJ/MonoMicro/FoSCI-master/traces/springblog.txt'
traces_file_path = '/home/amir/Desktop/PJ/MonoMicro/FoSCI-master/traces/xwiki-platform/xwiki-platform_trace.txt'

# # Path to your local git project
# repo_path = '/home/ec2-user/SageMaker/mono2micro/xwiki-platform'


class_id_to_name = {}

import os
import javalang
from collections import defaultdict
import re

def parse_java_file(file_path):
    with open(file_path, 'r') as file:
        java_code = file.read()
    return javalang.parse.parse(java_code)

has_parameter_results=[]
is_of_type_results = []
referece_results = []
call_results =[]
implement_results= []
return_results = []
inheritance_results = []

def extract_comments(file_path):
    with open(file_path, 'r') as file:
        java_code = file.read()

    pattern = r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"|\w+'
    matches = re.findall(pattern, java_code, re.DOTALL | re.MULTILINE)
    comments = [match for match in matches if match.startswith('//') or match.startswith('/*')]

    return comments



def resolve_type(tree,type_node, current_class):
    if isinstance(type_node, javalang.parser.tree.Type):
        type_name = type_node.name
        if '.' in type_name:
            return type_name
        elif is_nested_class(type_name, current_class):
            return f"{current_class.name}.{type_name}"
        else:
            return resolve_imported_type(tree,type_name)
    elif isinstance(type_node, javalang.parser.tree.TypeArgument):
        return resolve_type(tree,type_node.type, current_class)
    elif isinstance(type_node, javalang.parser.tree.TypeParameter):
        return resolve_type(tree,type_node.name, current_class)
    elif isinstance(type_node, str):
        return type_node
    else:
        raise ValueError(f"Unsupported type node: {type_node}")

def is_nested_class(type_name, current_class):
    for nested_class in current_class.body:
        if isinstance(nested_class, javalang.parser.tree.ClassDeclaration) and nested_class.name == type_name:
            return True
    return False

def resolve_imported_type(tree,type_name):
    import_declarations = [node for node in tree.imports if isinstance(node, javalang.parser.tree.Import)]
    for import_decl in import_declarations:
        if type_name == import_decl.path :
            return import_decl.path
        elif '.' in import_decl.path:
            package_name, imported_type = import_decl.path.rsplit('.', 1)
            if imported_type == type_name:
                return import_decl.path
    return type_name

def find_structural_dependencies(tree, all_classes,variable_types):
    class_package = tree.package.name if tree.package else 'default'

    for _, a_class in tree.filter(javalang.parser.tree.ClassDeclaration):
        
        #  Has Parameter (HP)
        for _, method in a_class.filter(javalang.parser.tree.MethodDeclaration):
            for param in method.parameters:
                param_type = param.type.name

                matches = [each_class for each_class in all_classes if each_class[1] == param_type]
                if len(matches) > 0:
                    for match in matches:
                        has_parameter_results.append(([c[0] for c in all_classes if c[1] == a_class.name][0], class_package, match[0], match[2]))
        
        # Reference (RE)
        for _, ref in a_class.filter(javalang.parser.tree.MemberReference):
            matches = [each_class for each_class in all_classes if each_class[1] == ref.qualifier]
            if len(matches) > 0:
                for match in matches:
                    referece_results.append((match[0], match[2], [c[0] for c in all_classes if c[1] == a_class.name][0], class_package))
        
        #  Calls (CA)
        for _, method_invocation in a_class.filter(javalang.parser.tree.MethodInvocation):
            qualifier = method_invocation.qualifier
            if isinstance(qualifier, str) and qualifier in variable_types:
                class_name = resolve_type(tree,variable_types[qualifier], a_class)
                matches = [each_class for each_class in all_classes if each_class[1] == class_name]
                if len(matches) > 0:
                    for match in matches:
                        call_results.append(([c[0] for c in all_classes if c[1] == a_class.name][0], class_package, match[0], match[2]))
            elif isinstance(qualifier, javalang.parser.tree.Literal ) :
                class_name = resolve_type(method_invocation.member.split('.')[0], a_class)
                matches = [each_class for each_class in all_classes if each_class[1] == class_name]
                if len(matches) > 0:
                    for match in matches:
                        call_results.append(([c[0] for c in all_classes if c[1] == a_class.name][0], class_package, match[0], match[2]))

        # Is-of-Type (IT)
        for _, field in a_class.filter(javalang.parser.tree.FieldDeclaration):
            if field.type:
                field_type = field.type.name
                matches = [each_class for each_class in all_classes if each_class[1] == field_type]
                if len(matches) > 0:
                    for match in matches:
                        is_of_type_results.append(([c[0] for c in all_classes if c[1] == a_class.name][0], class_package,match[0], match[2]))
        
        # Return (RT)
        for _, method in a_class.filter(javalang.parser.tree.MethodDeclaration):
            if method.return_type is not None:
                return_type = method.return_type.name
                matches = [each_class for each_class in all_classes if each_class[1] == return_type]
                if len(matches) > 0:
                    for match in matches:
                        return_results.append(([c[0] for c in all_classes if c[1] == a_class.name][0], class_package, match[0], match[2]))
        
        # Extends (EX)
        if a_class.extends:
            class_name = resolve_type(tree,a_class.extends.name, a_class)
            matches = [each_class for each_class in all_classes if each_class[1] == class_name]
            if len(matches) > 0:
                for match in matches:
                    inheritance_results.append((match[0], match[2], [c[0] for c in all_classes if c[1] == a_class.name][0], class_package))
        
        # Implement (IM)
        if a_class.implements:
            for implemented_interface in a_class.implements:
                interface_name = resolve_type(tree,implemented_interface, a_class)
                matches = [each_class for each_class in all_classes if each_class[2]+'.'+each_class[1] == interface_name]
                if len(matches) > 0:
                    for match in matches:
                        implement_results.append((match[0], match[2], [c[0] for c in all_classes if c[1] == a_class.name][0], class_package))



def extract_lexical_information(java_tree):
    class_info = defaultdict(list)
    for path, node in java_tree:
        try:
            if isinstance(node, javalang.parser.tree.ClassDeclaration):
                class_info['CN'].append(node.name)  # Class Name
            elif isinstance(node, javalang.parser.tree.FieldDeclaration):
                class_info['AN'].extend([field.name for field in node.declarators])  # Attribute Name
            elif isinstance(node, javalang.parser.tree.MethodDeclaration):
                class_info['MN'].append(node.name)  # Method Name
                class_info['PN'].extend([param.name for param in node.parameters])  # Parameter Name

            elif isinstance(node, javalang.parser.tree.ClassReference):
                class_info['SCS_ClassReference'].append(node.type.name)  # Source Code Statement
            elif isinstance(node, javalang.parser.tree.MemberReference):
                class_info['SCS_MemberReference'].append(node.member)  # Source Code Statement
            elif isinstance(node, javalang.parser.tree.MethodReference):
                class_info['SCS_MethodReference'].append(node.method.member)# + ":" + ",".join(arg.member for arg in node.children))  # Source Code Statement
            elif isinstance(node, javalang.parser.tree.VoidClassReference):
                class_info['SCS_VoidClassReference'].append(node.name)  # Source Code Statement
            elif isinstance(node, javalang.parser.tree.SuperMemberReference):
                class_info['SCS_SuperMemberReference'].append(node.member)  # Source Code Statement
            elif isinstance(node, javalang.parser.tree.ConstantDeclaration):
                class_info['SCS_ConstantDeclaration'].append(node.name)  # Source Code Statement
            elif isinstance(node, javalang.parser.tree.VariableDeclaration):
                class_info['SCS_VariableDeclaration'].append(node.type.name)  # Source Code Statement
            elif isinstance(node, javalang.parser.tree.VariableDeclarator):
                class_info['SCS_VariableDeclarator'].append(node.name)  # Source Code Statement
            elif isinstance(node, javalang.parser.tree.AnnotationDeclaration):
                class_info['SCS_AnnotationDeclaration'].append(node.name)  # Source Code Statement
            elif isinstance(node, javalang.parser.tree.ConstructorDeclaration):
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
        except Exception as e:
            print(f"Failed to parse {node} due to {str(e)}")            
    return class_info


all_classes= []
interfaces = []

def analyze_directory(directory):
    all_class_info = {}
    i=0
    variable_types = {}
    global all_classes
    for root, dirs, files in os.walk(directory):
        if 'test' not in root.lower():
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    try:
                        java_tree = parse_java_file(file_path)
                        class_package = java_tree.package.name if java_tree.package else "default"

                        for _, node in java_tree:

                            if isinstance(node, javalang.parser.tree.ClassDeclaration):
                                i=i+1
                                all_classes.append((i,node.name,class_package))

                            elif isinstance(node, javalang.parser.tree.InterfaceDeclaration):
                                i=i+1
                                all_classes.append((i,node.name,class_package))
                                interfaces.append(i)

                            elif isinstance(node, javalang.parser.tree.EnumDeclaration):
                                i=i+1
                                all_classes.append((i,node.name,class_package))

                            if isinstance(node, javalang.parser.tree.FieldDeclaration):
                                for declarator in node.declarators:
                                    variable_name = declarator.name
                                    variable_type = node.type.name
                                    variable_types[variable_name] = variable_type
                            elif isinstance(node, javalang.parser.tree.VariableDeclaration):
                                variable_name = node.declarators[0].name
                                variable_type = node.type.name
                                variable_types[variable_name] = variable_type

                            elif isinstance(node, javalang.parser.tree.MethodDeclaration):
                                for param in node.parameters:
                                    variable_name = param.name
                                    variable_type = param.type.name
                                    variable_types[variable_name] = variable_type

                            elif isinstance(node, javalang.parser.tree.ConstructorDeclaration):
                                for param in node.parameters:
                                    variable_name = param.name
                                    variable_type = param.type.name
                                    variable_types[variable_name] = variable_type

                            elif isinstance(node, javalang.parser.tree.TryStatement):
                                if node.catches is not None:
                                    for catch in node.catches:
                                        variable_name = catch.parameter.name
                                        variable_type = catch.parameter.types[0]
                                        variable_types[variable_name] = variable_type
                            
                            elif isinstance(node, javalang.parser.tree.ForStatement):
                                if node.control and isinstance(node.control, javalang.parser.tree.ForControl):
                                    if node.control.init is not None:
                                        for initializer in node.control.init:
                                            if isinstance(initializer, javalang.parser.tree.VariableDeclaration):
                                                variable_name = initializer.declarators[0].name
                                                variable_type = initializer.type.name
                                                variable_types[variable_name] = variable_type

                            elif isinstance(node, javalang.parser.tree.LambdaExpression):
                                for param in node.parameters:
                                    if 'qualifier' in param.attrs and param.qualifier !='':
                                        variable_name = param.member
                                        variable_type = param.qualifier
                                        variable_types[variable_name] = variable_type

                            elif isinstance(node, javalang.parser.tree.LocalVariableDeclaration):
                                for declarator in node.declarators:
                                    variable_name = declarator.name
                                    variable_type = node.type.name
                                    variable_types[variable_name] = variable_type
                    except Exception as e:
                        print(f"Failed to parse {file_path} due to {str(e)}")

    for root, dirs, files in os.walk(directory):
        if 'test' not in root.lower():
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    java_tree = parse_java_file(file_path)
                    class_info = extract_lexical_information(java_tree)

                    find_structural_dependencies(java_tree, all_classes,variable_types)
                    comments = extract_comments(file_path)
                    class_info['CO'].extend(comments)  # Comments

                    with open(file_path, 'r') as file:
                        java_code = file.read()
                    class_info['CODE'] = java_code
                    all_class_info[file] = class_info

    return all_class_info

lexical_info = analyze_directory(directory_path)




class_couplings = inheritance_results + return_results + implement_results + call_results + referece_results + is_of_type_results + has_parameter_results

interface_relations = []

def find_interface_relations(class_couplings):
    if interfaces:
        for pair in class_couplings:
            source_id, source_module, ref_id, ref_module = pair
            for interface in interfaces:
                if interface == source_id or interface == ref_id:
                    interface_relations.append(pair)

    return interface_relations

def get_directory(class_id):
    for a_class in all_classes:
        if a_class[0] == class_id:
            return a_class[2]

def get_value_from_indices(x, y, df, indices_dict):
    module_names = {v: k for k, v in indices_dict.items()}
    x_name = module_names.get(x, None)
    y_name = module_names.get(y, None)

    if x_name is None or y_name is None:
        return "Invalid indices provided."

    return df.loc[df['Unnamed: 0'] == x_name, y_name].values[0]

import numpy as np
from collections import defaultdict

module_indices = {index: class_name for index, class_name, _ in all_classes}

# class_names = all_classes

# name_to_id = {}
# for item in class_names:
#     id, name, path = item
#     if name not in name_to_id:
#         name_to_id[name] = id

import re


# unique_ids = sorted([item[0] for item in class_names])

# co_occurrence_matrix = {item[0]: {other_item[0]: 0 for other_item in all_classes} for item in all_classes}

# traces = {}

# with open(traces_file_path, 'r') as file:
#     next(file)  # Skip the header line
#     for line in file:
#         parts = line.strip().split(',')
#         if len(parts) < 9:
#             continue
#         for element in parts:
#             # if 'fi' in element:
#             #     class_name = element.split('.')[-1]
#             trace_id = parts[0]
#             class1_name = re.search(r'[\w\.]+\.(\w+)', element )
#             class1_name = class1_name.group(1) if class1_name else None
#             if trace_id not in traces:
#                 traces[trace_id] = []
#             class1_id =  next((a_class[0] for a_class in all_classes if a_class[1] == class1_name), None) #  name_to_id.get(class1_name)
#             if class1_id is not None:
#                 traces[trace_id].append(class1_id)

# for trace in traces.values():
#     for i in range(len(trace) - 1):
#         current_item = trace[i]
#         next_item = trace[i + 1]
#         if current_item != next_item:
#             co_occurrence_matrix[current_item][next_item] += 1

import re
import multiprocessing

def process_chunk(chunk):
    chunk_traces = {}
    for line in chunk:
        parts = line.strip().split(',')
        if len(parts) < 9:
            continue
        for element in parts:
            trace_id = parts[0]
            class1_name = re.search(r'[\w\.]+\.(\w+)', element)
            class1_name = class1_name.group(1) if class1_name else None
            if trace_id not in chunk_traces:
                chunk_traces[trace_id] = []
            class1_id = next((a_class[0] for a_class in all_classes if a_class[1] == class1_name), None)
            if class1_id is not None:
                chunk_traces[trace_id].append(class1_id)
    return chunk_traces


def update_co_occurrence_matrix(trace, co_occurrence_matrix):
    for i in range(len(trace) - 1):
        current_item = trace[i]
        next_item = trace[i + 1]
        if current_item != next_item:
            co_occurrence_matrix[current_item][next_item].value += 1

# if __name__ == '__main__':
manager = multiprocessing.Manager()
co_occurrence_matrix = {
    item[0]: {
        other_item[0]: manager.Value('i', 0) for other_item in all_classes
    } for item in all_classes
}

# co_occurrence_matrix = {item[0]: {other_item[0]: 0 for other_item in all_classes} for item in all_classes}
matrix_lock = multiprocessing.Lock()

pool = multiprocessing.Pool()

with open(traces_file_path, 'r') as file:
    next(file)  # Skip the header line
    chunks = []
    chunk_size = 100000  # Adjust the chunk size based on your system's memory
    chunk = []
    for line in file:
        chunk.append(line)
        if len(chunk) == chunk_size:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)

chunk_results = pool.map(process_chunk, chunks)

traces = {}
for chunk_result in chunk_results:
    traces.update(chunk_result)

# Use starmap to pass multiple arguments to update_co_occurrence_matrix
pool.starmap(update_co_occurrence_matrix, [(trace, co_occurrence_matrix) for trace in traces.values()])

pool.close()
pool.join()

# Convert the managed values back to regular integers
co_occurrence_matrix = {
    k: {inner_k: inner_v.value for inner_k, inner_v in v.items()}
    for k, v in co_occurrence_matrix.items()
}


transformed_data = []
for class_name, co_occurrences in co_occurrence_matrix.items():
    row = {'class': class_name}
    row.update(co_occurrences)
    transformed_data.append(row)

class_co_eccurances_df = pd.DataFrame(transformed_data)
class_co_eccurances_df.set_index('class', inplace=True)

class_co_eccurances_matrix = np.where(class_co_eccurances_df == 0, 1, class_co_eccurances_df)

n = len(all_classes)
adj_matrix = np.zeros((n, n), dtype=int)

for source_class_id, source_module_name, referenced_class_id, referenced_module_name in inheritance_results:
    adj_matrix[source_class_id - 1, referenced_class_id - 1] += 8.5
    adj_matrix[referenced_class_id - 1, source_class_id - 1] += 8.5

for source_class_id, source_module_name, referenced_class_id, referenced_module_name in return_results:
    adj_matrix[source_class_id - 1, referenced_class_id - 1] += 1
    adj_matrix[referenced_class_id - 1, source_class_id - 1] += 1

for source_class_id, source_module_name, referenced_class_id, referenced_module_name in implement_results:
    adj_matrix[source_class_id - 1, referenced_class_id - 1] += 2
    adj_matrix[referenced_class_id - 1, source_class_id - 1] += 2

for source_class_id, source_module_name, referenced_class_id, referenced_module_name in call_results:
    adj_matrix[source_class_id - 1, referenced_class_id - 1] += 2.5
    adj_matrix[referenced_class_id - 1, source_class_id - 1] += 2.5

for source_class_id, source_module_name, referenced_class_id, referenced_module_name in referece_results:
    adj_matrix[source_class_id - 1, referenced_class_id - 1] += 3
    adj_matrix[referenced_class_id - 1, source_class_id - 1] += 3

for source_class_id, source_module_name, referenced_class_id, referenced_module_name in is_of_type_results:
    adj_matrix[source_class_id - 1, referenced_class_id - 1] += 2
    adj_matrix[referenced_class_id - 1, source_class_id - 1] += 2

for source_class_id, source_module_name, referenced_class_id, referenced_module_name in has_parameter_results:
    adj_matrix[source_class_id - 1, referenced_class_id - 1] += 3.5
    adj_matrix[referenced_class_id - 1, source_class_id - 1] += 3.5

def add_edge(G, node1, node2, weight, type_of_relation):
    if G.has_edge(node1, node2):
        if type_of_relation in G[node1][node2]:
            G[node1][node2][type_of_relation]['weight'] += weight
        else:
            G[node1][node2][type_of_relation] = {'weight': weight}
    else:
        G.add_edge(node1, node2)
        G[node1][node2][type_of_relation] = {'weight': weight}

import networkx as nx

G = nx.DiGraph()
G_inheritance = nx.DiGraph()
G_return = nx.DiGraph()
G_implement = nx.DiGraph()
G_call = nx.DiGraph()
G_reference = nx.DiGraph()
G_is_of_type = nx.DiGraph()
G_has_parameter = nx.DiGraph()
G_intra = nx.DiGraph()
G_inter = nx.DiGraph()

class_couplings_set = set(class_couplings)

for pair in inheritance_results:
    source_id, source_module, ref_id, ref_module = pair
    add_edge(G, source_id, ref_id, 8.5, 'inheritance')

for pair in return_results:
    source_id, source_module, ref_id, ref_module = pair
    add_edge(G, source_id, ref_id, 1, 'return')

for pair in implement_results:
    source_id, source_module, ref_id, ref_module = pair
    add_edge(G, source_id, ref_id, 2, 'implement')

for pair in call_results:
    source_id, source_module, ref_id, ref_module = pair
    add_edge(G, source_id, ref_id, 2.5, 'call')

for pair in referece_results:
    source_id, source_module, ref_id, ref_module = pair
    add_edge(G, source_id, ref_id, 3, 'reference')

for pair in is_of_type_results:
    source_id, source_module, ref_id, ref_module = pair
    add_edge(G, source_id, ref_id, 2, 'is_of_type')

for pair in has_parameter_results:
    source_id, source_module, ref_id, ref_module = pair
    add_edge(G, source_id, ref_id, 3.5, 'has_parameter')

def discover_inter_coupling_classes(G):
    for node1, node2, attrs in G.edges(data=True):
        sum = 0
        source_class_dir = get_directory(node1)
        dest_class_dir = get_directory(node2)

        if source_class_dir != dest_class_dir:
            for relation, type_of_relation in attrs.items():
                sum += type_of_relation.get('weight', 0)
            add_edge(G_inter, node1, node2, sum, 'all')
        else:
            for relation, type_of_relation in attrs.items():
                sum += type_of_relation.get('weight', 0)
            add_edge(G_intra, node1, node2, sum, 'all')

discover_inter_coupling_classes(G)

import pandas as pd

in_degrees = dict(G_inter.in_degree())
out_degrees = dict(G_inter.out_degree())

inter_coupling_nodes = set()

for node in G:
    has_node = G_inter.has_node(node)
    if not has_node:
        continue
    in_degree = sum([G_inter[u][node]['all']['weight'] for u in G_inter.predecessors(node)])
    out_degree = sum([G_inter[node][v]['all']['weight'] for v in G_inter.successors(node)])
    if in_degree + out_degree >= 0:
        inter_coupling_nodes.add(node)
    else:
        continue

inter_coupling_nodes = sorted(inter_coupling_nodes, key=lambda node: (in_degrees[node], out_degrees[node]))


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
                    if c_name == cn:
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



import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

coefficients = {'CN': 0.1413, 'AN': 0.1113, 'MN': 0.1313, 'PN': 0.1413, 'SCS_MethodDeclaration': 0.1750, 'SCS_ClassReference': 0.1750, 'SCS_MemberReference': 0.1750,
    'SCS_MethodReference': 0.1750, 'SCS_VoidClassReference': 0.1750, 'SCS_SuperMemberReference': 0.1750,
    'SCS_ConstantDeclaration': 0.1750, 'SCS_VariableDeclaration': 0.1750, 'SCS_VariableDeclarator': 0.1750,
    'SCS_AnnotationDeclaration': 0.1750, 'SCS_ConstructorDeclaration': 0.1750,
    'SCS_LocalVariableDeclaration': 0.1750, 'SCS_MethodInvocation': 0.1750,
    'SCS_FieldDeclaration': 0.1750, 'CO': 0.2225}

import re

def filter_out_unwanted_comments(comments):
    filtered_comments = []
    for comment in comments:
        if "copyright" not in comment.lower() and 'author' not in comment.lower() and 'licensed' not in comment.lower():
            filtered_comments.append(comment.replace('.','').replace('/','').replace("\\",''))
    return filtered_comments


from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def compute_comment_similarity(class1_comments, class2_comments):
    class1_embeddings = [model.encode(comment, convert_to_tensor=True) for comment in class1_comments]
    class2_embeddings = [model.encode(comment, convert_to_tensor=True) for comment in class2_comments]
    if len(class1_embeddings) == 0 or len(class2_embeddings) == 0:
        return 0
    similarities = []
    for embed1 in class1_embeddings:
        for embed2 in class2_embeddings:
            cosine_similarity = util.pytorch_cos_sim(embed1, embed2)
            similarities.append(cosine_similarity)
    return sum(similarities) / len(similarities)

def calculate_similarity(file1,file2,category):
    doc1 = " ".join([str(element) for element in new_lexical_info[file1][category]])
    doc2 = " ".join([str(element) for element in new_lexical_info[file2][category]])

    if category == "CO":
        class1_comments = filter_out_unwanted_comments(new_lexical_info[file1][category])
        class2_comments = filter_out_unwanted_comments(new_lexical_info[file2][category])
        similarity = compute_comment_similarity(class1_comments, class2_comments)
        return similarity

    if "SCS" in category:
        import difflib
        similarity = difflib.SequenceMatcher(None, doc1, doc2).ratio()

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
            total_distances += (1 - normalized_distance)
            total_elements += 1
    
    if total_elements > 0:
        sim = total_distances / total_elements
        return sim
    else:
        return 0

new_lexical_info = convert_class_id_to_name(lexical_info)

total_similarity = np.zeros((len(new_lexical_info.items()), len(new_lexical_info.items())))

for i, module1 in enumerate(new_lexical_info):
    for j, module2 in enumerate(new_lexical_info):
        if i <= j:
            total_similarity_ij = 0
            for category in ['CN', 'AN', 'MN', 'PN', 'CO',
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
            total_similarity[j, i] = total_similarity_ij

submodule_count = 1
submodules = defaultdict(set)
nodes_in_submodules = set()

directories = defaultdict(set)
for node in G.nodes():
    directory = get_directory(node)
    directories[directory].add(node)

related_files = []
for node in inter_coupling_nodes:
    directory = get_directory(node)
    if node in G:
        related_nodes = directories[directory]
        related_nodes = related_nodes - set(inter_coupling_nodes)
        column = total_similarity[:, node -1 ]
        rows_to_consider = list(related_nodes)
        intra_coupling_treshold = min (2, len(rows_to_consider))
        indices = np.argpartition(column, -intra_coupling_treshold)[-intra_coupling_treshold:]
        sorted_indices = indices[np.argsort(column[indices])[::-1]]
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
        submodules[f'S{submodule_count}'].update([node])
        submodule_count += 1




from sklearn import preprocessing
import numpy as np
import git
import json
from itertools import combinations

# Create a scaler object
scaler = preprocessing.MinMaxScaler()

# Normalize the conceptual matrix
reshaped_array = total_similarity.reshape(-1, 1)
normalized_conceptual_matrix = scaler.fit_transform(reshaped_array)
normalized_conceptual_matrix = normalized_conceptual_matrix.reshape(total_similarity.shape)

# Set the diagonal elements to zero
np.fill_diagonal(normalized_conceptual_matrix, 0)

# Normalize the structural matrix
reshaped_array = adj_matrix.reshape(-1, 1)
normalized_structural_matrix = scaler.fit_transform(reshaped_array)
normalized_structural_matrix = normalized_structural_matrix.reshape(adj_matrix.shape)

# Calculate coupling using weighted combination of conceptual and structural matrices
coupling  = normalized_conceptual_matrix * 0.2 + normalized_structural_matrix * 0.8

# repo = git.Repo(repo_path)
# commit_history = {}

# # Iterate through each commit
# for commit in repo.iter_commits():
#     changed_files = set()

#     # For each changed file in the commit
#     for item in commit.stats.files.keys():
#         # Assuming Java files; adjust the condition for other languages
#         if item.endswith('.java'):
#             # Extract class name from file name
#             class_name = item.split('/')[-1].replace('.java', '').lower()
#             if class_name in module_indices.values():
#                 changed_files.add(class_name)
#     if len(changed_files) > 1 and len(changed_files) <= len(module_indices)/2:
#         commit_history[commit.hexsha] = changed_files

# # Count co-changes for each pair of classes
# def count_co_changes(commit_history):
#     co_change_count = {}
#     for classes in commit_history.values():
#         for class1, class2 in combinations(classes, 2):
#             co_change_count[(class1.lower(), class2.lower())] = co_change_count.get((class1.lower(), class2.lower()), 0) + 1
#             co_change_count[(class2.lower(), class1.lower())] = co_change_count.get((class2.lower(), class1.lower()), 0) + 1
#     return co_change_count

# co_change_count = count_co_changes(commit_history)

# # Create a matrix initialized with zeros
# matrix_size = len(module_indices)
# matrix = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]

# # Populate the matrix
# for (class1, class2), value in co_change_count.items():
#     row = [index-1 for index, class_name in module_indices.items() if class_name == class1][0]
#     col = [index-1 for index, class_name in module_indices.items() if class_name == class2][0]
#     matrix[row][col] = value

# # Dump the matrices and other data into JSON files
# with open('roller_co_commited_classes.json', 'w') as file:
#     json.dump({'co_commited_classes': matrix}, file, indent=4)

def convert_keys_to_string(d):
    return {str(k): {str(inner_k): inner_v for inner_k, inner_v in v.items()} for k, v in d.items()}

co_occurrence_matrix_serializable = convert_keys_to_string(co_occurrence_matrix)



with open('coupling.json', 'w') as file:
    json.dump({
        'conceptual_coupling_matrix': total_similarity.tolist(),
        'normalized_conceptual_coupling_matrix': normalized_conceptual_matrix.tolist(),
        'normalized_structural_coupling_matrix': normalized_structural_matrix.tolist(),
        'structural_coupling_matrix': adj_matrix.tolist(),
        'submodules': {key: list(value) for key, value in submodules.items()},
        'class_indices': module_indices,
        'class_co_eccurances_in_execution_traces': co_occurrence_matrix_serializable
    }, file, indent=4)



import jsonpickle

# Serialize with jsonpickle
json_data = jsonpickle.encode({'lexical_info':lexical_info ,'all_classes':all_classes, 'interface_relations':interface_relations, 'interfaces':interfaces, 'submodules':submodules,'graph':G})

# Save JSON data to a file
with open('eval.json', 'w') as f:
    f.write(json_data)


# # Read JSON data from the file
# with open('lexical_info.json', 'r') as f:
#     json_data = f.read()

# # Deserialize the JSON string back into Python objects with jsonpickle
# restored_objects = jsonpickle.decode(json_data)

# # Access your objects
# lexical_info = restored_objects['lexical_info']
# all_classes = restored_objects['all_classes']
# interface_relations = restored_objects['interface_relations']
# interfaces = restored_objects['interfaces']
# graph_info = restored_objects['graph']
# submodules = restored_objects['submodules']
