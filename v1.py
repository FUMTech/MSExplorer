# %%
# Initialize constants for database and other configurations
import mysql.connector
import os
import javalang
import re
import random
import math
from itertools import combinations
from collections import defaultdict
import networkx as nx
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util
import torch
from scipy.spatial.distance import cosine
import json
import git

# Database connection parameters
DB_PARAMS = {
    'host': '3.67.227.198',
    'user': 'root',
    'password': 'root',
    'database': 'xwiki',
    'port': '3355'
}

# Directory paths
directory_path = "/home/amir/Desktop/PJ/MonoMicro/jpetstore-6-jpetstore-6.0.2"
repo_path = '/home/amir/Desktop/PJ/MonoMicro/jpetstore-6-jpetstore-6.0.2'

# Coefficients for similarity calculation
coefficients = {
    'CN': 0.1413, 'AN': 0.1113, 'MN': 0.1313, 'PN': 0.1413,
    'SCS_MethodDeclaration': 0.1750, 'SCS_ClassReference': 0.1750, 'SCS_MemberReference': 0.1750,
    'SCS_MethodReference': 0.1750, 'SCS_VoidClassReference': 0.1750, 'SCS_SuperMemberReference': 0.1750,
    'SCS_ConstantDeclaration': 0.1750, 'SCS_VariableDeclaration': 0.1750, 'SCS_VariableDeclarator': 0.1750,
    'SCS_AnnotationDeclaration': 0.1750, 'SCS_ConstructorDeclaration': 0.1750,
    'SCS_LocalVariableDeclaration': 0.1750, 'SCS_MethodInvocation': 0.1750,
    'SCS_FieldDeclaration': 0.1750, 'CO': 0.2225
}

# Global variables for storing extracted data
lexical_info = {}
all_classes = []
interfaces = []
class_id_to_name = {}
interface_relations = []
has_parameter_results = []
is_of_type_results = []
referece_results = []
call_results = []
implement_results = []
return_results = []
inheritance_results = []
cost_array = []

# Helper function to connect to the database
def connect_to_database(host, user, password, database, port):
    cnx = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port
    )
    return cnx

# Helper function to execute a SQL query
def execute_query(cnx, query):
    cursor = cnx.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    return results

# Helper function to close the database connection
def close_database_connection(cnx):
    cnx.close()

# Helper function to parse Java files
def parse_java_file(file_path):
    with open(file_path, 'r') as file:
        java_code = file.read()
    return javalang.parse.parse(java_code)

# Helper function to extract comments from Java files
def extract_comments(file_path):
    with open(file_path, 'r') as file:
        java_code = file.read()
    pattern = r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"|\w+'
    matches = re.findall(pattern, java_code, re.DOTALL | re.MULTILINE)
    comments = [match for match in matches if match.startswith('//') or match.startswith('/*')]
    return comments

# Helper function to filter out unwanted comments
def filter_out_unwanted_comments(comments):
    filtered_comments = [comment.replace('.', '').replace('/', '').replace("\\", '') for comment in comments
                         if "copyright" not in comment.lower() and 'author' not in comment.lower() and 'licensed' not in comment.lower()]
    return filtered_comments

# Helper function to embed text using BERT
def embed_text_using_bert(text):
    model_name = "bert-base-uncased"
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        model_output = model(**tokens)
    return model_output.last_hidden_state[:, 0, :].numpy()

# Helper function to compute similarity between comments using BERT embeddings
def compute_comment_similarity(class1_comments, class2_comments):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    class1_embeddings = [model.encode(comment, convert_to_tensor=True) for comment in class1_comments]
    class2_embeddings = [model.encode(comment, convert_to_tensor=True) for comment in class2_comments]
    if not class1_embeddings or not class2_embeddings:
        return 0
    similarities = [util.pytorch_cos_sim(embed1, embed2).item() for embed1 in class1_embeddings for embed2 in class2_embeddings]
    return sum(similarities) / len(similarities)

# Helper function to calculate similarity between two classes for a specific category
def calculate_similarity(file1, file2, category):
    doc1 = " ".join([str(element) for element in new_lexical_info[file1][category]])
    doc2 = " ".join([str(element) for element in new_lexical_info[file2][category]])
    if category == "CO":
        class1_comments = filter_out_unwanted_comments(new_lexical_info[file1][category])
        class2_comments = filter_out_unwanted_comments(new_lexical_info[file2][category])
        return compute_comment_similarity(class1_comments, class2_comments)
    if "SCS" in category:
        import difflib
        similarity = difflib.SequenceMatcher(None, doc1, doc2).ratio()
        return similarity
    elements_file1 = [re.sub(r'\b(byte|short|int|long|float|double|boolean|char|string|list|map|set|arraylist|hashmap|hashset|integer)\b', '', item.lower()).strip()
                      for item in new_lexical_info[file1][category] if item is not None and re.sub(r'\b(byte|short|int|long|float|double|boolean|char|string|list|map|set|arraylist|hashmap|hashset|integer)\b', '', item.lower()).strip() != ""]
    elements_file2 = [re.sub(r'\b(byte|short|int|long|float|double|boolean|char|string|list|map|set|arraylist|hashmap|hashset|integer)\b', '', item.lower()).strip()
                      for item in new_lexical_info[file2][category] if item is not None and re.sub(r'\b(byte|short|int|long|float|double|boolean|char|string|list|map|set|arraylist|hashmap|hashset|integer)\b', '', item.lower()).strip() != ""]
    total_distances = 0
    total_elements = 0
    for element1 in elements_file1:
        for element2 in elements_file2:
            distance = Levenshtein.distance(str(element1), str(element2))
            normalized_distance = distance / max(len(str(element1)), len(str(element2)))
            total_distances += (1 - normalized_distance)
            total_elements += 1
    return total_distances / total_elements if total_elements > 0 else 0

# Function to extract lexical information from Java files
def extract_lexical_information(java_tree):
    class_info = defaultdict(list)
    for path, node in java_tree:
        try:
            if isinstance(node, javalang.tree.ClassDeclaration):
                class_info['CN'].append(node.name)
            elif isinstance(node, javalang.tree.FieldDeclaration):
                class_info['AN'].extend([field.name for field in node.declarators])
            elif isinstance(node, javalang.tree.MethodDeclaration):
                class_info['MN'].append(node.name)
                class_info['PN'].extend([param.name for param in node.parameters])
            elif isinstance(node, javalang.tree.ClassReference):
                class_info['SCS_ClassReference'].append(node.type.name)
            elif isinstance(node, javalang.tree.MemberReference):
                class_info['SCS_MemberReference'].append(node.member)
            elif isinstance(node, javalang.tree.MethodReference):
                class_info['SCS_MethodReference'].append(node.method.member)
            elif isinstance(node, javalang.tree.VoidClassReference):
                class_info['SCS_VoidClassReference'].append(node.name)
            elif isinstance(node, javalang.tree.SuperMemberReference):
                class_info['SCS_SuperMemberReference'].append(node.member)
            elif isinstance(node, javalang.tree.ConstantDeclaration):
                class_info['SCS_ConstantDeclaration'].append(node.name)
            elif isinstance(node, javalang.tree.VariableDeclaration):
                class_info['SCS_VariableDeclaration'].append(node.type.name)
            elif isinstance(node, javalang.tree.VariableDeclarator):
                class_info['SCS_VariableDeclarator'].append(node.name)
            elif isinstance(node, javalang.tree.AnnotationDeclaration):
                class_info['SCS_AnnotationDeclaration'].append(node.name)
            elif isinstance(node, javalang.tree.ConstructorDeclaration):
                class_info['SCS_ConstructorDeclaration'].append(node.name)
            elif isinstance(node, javalang.tree.LocalVariableDeclaration):
                class_info['SCS_LocalVariableDeclaration'].append(node.name)
            elif isinstance(node, javalang.tree.MethodInvocation):
                class_info['SCS_ClassReference'].append(node.qualifier)
                class_info['SCS_MethodInvocation'].append(node.member)
            elif isinstance(node, javalang.tree.FieldDeclaration):
                class_info['SCS_FieldDeclaration'].append(node.type.name)
            elif isinstance(node, javalang.tree.MethodDeclaration):
                class_info['SCS_MethodDeclaration'].append(node.return_type.name)
            elif isinstance(node, javalang.tree.EnumDeclaration):
                class_info['CN'].append(node.name)
            elif isinstance(node, javalang.tree.InterfaceDeclaration):
                class_info['CN'].append(node.name)
        except Exception as e:
            print(f"Failed to parse {node} due to {str(e)}")
    return class_info

# Function to analyze the directory and extract lexical information
def analyze_directory(directory):
    all_class_info = {}
    class_list = []
    global all_classes
    i = 0
    variable_types = {}
    for root, dirs, files in os.walk(directory):
        if 'test' not in root.lower():
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    try:
                        java_tree = parse_java_file(file_path)
                        class_package = java_tree.package.name if java_tree.package else "default"
                        for path, node in java_tree:
                            if isinstance(node, javalang.tree.ClassDeclaration):
                                i += 1
                                all_classes.append((i, node.name, class_package))
                            elif isinstance(node, javalang.tree.InterfaceDeclaration):
                                i += 1
                                all_classes.append((i, node.name, class_package))
                                interfaces.append(i)
                            elif isinstance(node, javalang.tree.EnumDeclaration):
                                i += 1
                                all_classes.append((i, node.name, class_package))
                            if isinstance(node, javalang.tree.FieldDeclaration):
                                for declarator in node.declarators:
                                    variable_name = declarator.name
                                    variable_type = node.type.name
                                    variable_types[variable_name] = variable_type
                            elif isinstance(node, javalang.tree.VariableDeclaration):
                                variable_name = node.declarators[0].name
                                variable_type = node.type.name
                                variable_types[variable_name] = variable_type
                            elif isinstance(node, javalang.tree.MethodDeclaration):
                                for param in node.parameters:
                                    variable_name = param.name
                                    variable_type = param.type.name
                                    variable_types[variable_name] = variable_type
                            elif isinstance(node, javalang.tree.ConstructorDeclaration):
                                for param in node.parameters:
                                    variable_name = param.name
                                    variable_type = param.type.name
                                    variable_types[variable_name] = variable_type
                            elif isinstance(node, javalang.tree.TryStatement):
                                if node.catches is not None:
                                    for catch in node.catches:
                                        variable_name = catch.parameter.name
                                        variable_type = catch.parameter.types[0]
                                        variable_types[variable_name] = variable_type
                            elif isinstance(node, javalang.tree.ForStatement):
                                if node.control and isinstance(node.control, javalang.tree.ForControl):
                                    if node.control.init is not None:
                                        for initializer in node.control.init:
                                            if isinstance(initializer, javalang.tree.VariableDeclaration):
                                                variable_name = initializer.declarators[0].name
                                                variable_type = initializer.type.name
                                                variable_types[variable_name] = variable_type
                            elif isinstance(node, javalang.tree.LambdaExpression):
                                for param in node.parameters:
                                    if 'qualifier' in param.attrs and param.qualifier != '':
                                        variable_name = param.member
                                        variable_type = param.qualifier
                                        variable_types[variable_name] = variable_type
                            elif isinstance(node, javalang.tree.LocalVariableDeclaration):
                                for declarator in node.declarators:
                                    variable_name = declarator.name
                                    variable_type = node.type.name
                                    variable_types[variable_name] = variable_type
                        class_info = extract_lexical_information(java_tree)
                        comments = extract_comments(file_path)
                        class_info['CO'].extend(comments)
                        with open(file_path, 'r') as file:
                            java_code = file.read()
                        class_info['CODE'] = java_code
                        all_class_info[file] = class_info
                    except Exception as e:
                        print(f"Failed to parse {file_path} due to {str(e)}")
    return all_class_info

lexical_info = analyze_directory(directory_path)

# Function to convert class IDs to names and create new lexical info
def convert_class_id_to_name(lexical_info):
    results = all_classes
    if results:
        new_lexical_info = {}
        for c_id, c_name, c_dir in all_classes:
            new_lexical_info[c_id] = {
                'CN': [], 'AN': [], 'MN': [], 'PN': [],
                'SCS_ClassReference': [], 'SCS_MemberReference': [], 'SCS_MethodReference': [],
                'SCS_VoidClassReference': [], 'SCS_SuperMemberReference': [], 'SCS_ConstantDeclaration': [],
                'SCS_VariableDeclaration': [], 'SCS_VariableDeclarator': [], 'SCS_AnnotationDeclaration': [],
                'SCS_ConstructorDeclaration': [], 'SCS_LocalVariableDeclaration': [], 'SCS_MethodInvocation': [],
                'SCS_FieldDeclaration': [], 'SCS_MethodDeclaration': [], 'CO': []
            }
            curr_class_name = ''
            for file, info in lexical_info.items():
                if curr_class_name != '':
                    break
                for cn in info['CN']:
                    if c_name == cn.lower():
                        curr_class_name = file
                        break
            new_lexical_info[c_id]['CN'] += lexical_info[curr_class_name]['CN']
            new_lexical_info[c_id]['AN'] += lexical_info[curr_class_name]['AN']
            new_lexical_info[c_id]['MN'] += lexical_info[curr_class_name]['MN']
            new_lexical_info[c_id]['PN'] += lexical_info[curr_class_name]['PN']
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
            new_lexical_info[c_id]['CO'] += lexical_info[curr_class_name]['CO']
    return new_lexical_info

new_lexical_info = convert_class_id_to_name(lexical_info)

# Function to analyze the directory and extract lexical information
def calculate_total_similarity(new_lexical_info, coefficients):
    total_similarity = np.zeros((len(new_lexical_info.items()), len(new_lexical_info.items())))
    for i, module1 in enumerate(new_lexical_info):
        for j, module2 in enumerate(new_lexical_info):
            if i <= j:
                total_similarity_ij = 0
                for category in ['CN', 'AN', 'MN', 'PN', 'CO', 'SCS_ClassReference', 'SCS_MemberReference', 'SCS_MethodReference',
                                 'SCS_VoidClassReference', 'SCS_SuperMemberReference', 'SCS_ConstantDeclaration', 'SCS_VariableDeclaration',
                                 'SCS_VariableDeclarator', 'SCS_AnnotationDeclaration', 'SCS_ConstructorDeclaration', 'SCS_LocalVariableDeclaration',
                                 'SCS_MethodInvocation', 'SCS_FieldDeclaration', 'SCS_MethodDeclaration']:
                    if module1 == module2:
                        similarity = 0
                    else:
                        similarity = calculate_similarity(module1, module2, category)
                        similarity = 0 if similarity < 0 else similarity
                    total_similarity_ij += coefficients[category] * similarity
                total_similarity[i, j] = total_similarity_ij
                total_similarity[j, i] = total_similarity_ij
    return total_similarity

total_similarity = calculate_total_similarity(new_lexical_info, coefficients)

# Create submodules based on the similarity matrix
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

        column = total_similarity[:, node - 1]

        rows_to_consider = list(related_nodes)

        selected_values = column[rows_to_consider]

        intra_coupling_treshold = min(2, len(rows_to_consider))

        indices = np.argpartition(selected_values, -intra_coupling_treshold)[-intra_coupling_treshold:]

        sorted_indices = indices[np.argsort(selected_values[indices])[::-1]]

        related_nodes = [node] + [index + 1 for index in sorted_indices]

        subgraph = G.subgraph(related_nodes)
        nodes = nx.dfs_preorder_nodes(subgraph, node)
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

# Function to compute conceptual coupling using BERT embeddings
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

# Function to calculate total similarity
def calculate_similarity(file1, file2, category):
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
        return similarity

    data_types_and_classes = ["byte", "short", "int", "long", "float", "double", "boolean", "char",
                              "string", "list", "map", "set", "arraylist", "hashmap", "hashset", 'integer']
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

# Normalize the coupling matrices
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()

reshaped_array = total_similarity.reshape(-1, 1)
normalized_conseptual_matrix = scaler.fit_transform(reshaped_array)
normalized_conseptual_matrix = normalized_conseptual_matrix.reshape(total_similarity.shape)

for index, value in np.ndenumerate(normalized_conseptual_matrix):
    if index[0] == index[1]:
        normalized_conseptual_matrix[index] = 0

reshaped_array = adj_matrix.reshape(-1, 1)
normalized_structural_matrix = scaler.fit_transform(reshaped_array)
normalized_structural_matrix = normalized_structural_matrix.reshape(adj_matrix.shape)

# Calculate coupling
coupling = normalized_conseptual_matrix * 0.2 + normalized_structural_matrix * 0.8

# Calculate energy function
def energy_function(state):
    total_size = 0

    for microservice in state:
        classes_count = find_number_of_classes_within_microservice(microservice)
        total_size += classes_count ** 2

    s_avg = total_size / all_classes
    s_star = all_classes / (0.1 * all_classes)
    w = 0.05
    MSI = math.exp(-0.5 * ((math.log(s_avg) - math.log(s_star)) / (w * math.log(all_classes))) ** 2)

    BMCI, IMCI = calculate_bmc_imc(coupling, state)

    cost = (IMCI / (IMCI + BMCI)) ** 0.5 * MSI ** 0.5

    IFN = find_IFN(set_dictionary=submodules, microservices=state, interface_relationships=find_interface_relations(class_couplings))

    CHM = find_CHM(candidate_microservices=state, interface_relationships=find_interface_relations(class_couplings))

    CHD = find_CHD(candidate_microservices=state, interface_relationships=find_interface_relations(class_couplings))

    SMQ = smq(state, G)

    ICF = calculate_ICF(state, commit_history)

    ECF = compute_ecf(state, co_change_count)

    cost_array.append({str(state): (cost, IFN, CHM, CHD, SMQ, ICF, ECF)})

    return cost

# Simulated annealing function
def simulated_annealing(initial_state, energy_function, neighbourhood_function, annealing_schedule):
    current_state = initial_state
    current_energy = energy_function(current_state)
    current_temperature = initial_temperature
    iterations = 0

    while current_temperature > 1e-100 and iterations < max_iterations:
        neighbour = neighbourhood_function(current_state)
        neighbour_energy = energy_function(neighbour)
        iterations += 1

        if (neighbour_energy > current_energy):
            current_state = neighbour
            current_energy = neighbour_energy

        current_temperature = annealing_schedule(current_temperature, iterations)

    return current_state

# Neighbourhood function for simulated annealing
def neighbourhood_function(state):
    neighbour = [set(microservice) for microservice in state]

    if len(neighbour) < 2:
        return neighbour

    microservice1, microservice2 = random.sample(neighbour, 2)

    if random.uniform(0, 1) <= 1:
        if microservice1 and microservice2:
            submodule_to_move = random.sample(microservice1, 1)[0]
            microservice1.remove(submodule_to_move)
            microservice2.add(submodule_to_move)
    else:
        if len(microservice1) > 1 and len(microservice2) > 1:
            submodule1 = random.sample(microservice1, 1)[0]
            submodule2 = random.sample(microservice2, 1)[0]
            microservice1.remove(submodule1)
            microservice2.remove(submodule2)
            microservice1.add(submodule2)
            microservice2.add(submodule1)

    neighbour = [microservice for microservice in neighbour if microservice]

    return neighbour

# Annealing schedule function
def annealing_schedule(temperature, iteration):
    return temperature * 0.99 ** iteration

# Simulated annealing process
initial_state = [{submodule} for submodule in submodules.keys()]
candidate_state = simulated_annealing(initial_state=initial_state, energy_function=energy_function, neighbourhood_function=neighbourhood_function, annealing_schedule=annealing_schedule)
print("Candidate State = ", candidate_state)

# Find the state with the maximum cost
max_cost_state = None
max_cost = float('-inf')
max_IFN = float('+inf')
max_CHM = float('-inf')
max_CHD = float('-inf')
max_SMQ = float('-inf')
max_ICF = float('-inf')
max_ECF = float('+inf')
max_IFN_state = None
max_CHM_state = None
max_CHD_state = None
max_SMQ_state = None
max_ICF_state = None
max_ECF_state = None

for entry in cost_array:
    for state, values in entry.items():
        if values[0] > max_cost:
            max_cost = values[0]
            max_cost_state = state
        if values[1] < max_IFN:
            max_IFN = values[1]
            max_IFN_state = state
        if values[2] > max_CHM:
            max_CHM = values[2]
            max_CHM_state = state
        if values[3] > max_CHD:
            max_CHD = values[3]
            max_CHD_state = state
        if values[4] > max_SMQ:
            max_SMQ = values[4]
            max_SMQ_state = state
        if values[5] > max_ICF:
            max_ICF = values[5]
            max_ICF_state = state
        if values[6] < max_ECF:
            max_ECF = values[6]
            max_ECF_state = state

print("Max cost state: ", max_cost_state)
print("Max IFN state: ", max_IFN_state)
print("Max CHM state: ", max_CHM_state)
print("Max CHD state: ", max_CHD_state)
print("Max SMQ state: ", max_SMQ_state)
print("Max ICF state: ", max_ICF_state)
print("Max ECF state: ", max_ECF_state)

# Export the dictionary to a JSON file
json_file_path = '/path/to/submodules.json'
with open(json_file_path, 'w') as json_file:
    json.dump(submodules, json_file, indent=4)

# Print the path to the saved JSON file
print("Submodules saved to:", json_file_path)

import ace_tools as tools; tools.display_dataframe_to_user(name="Submodules", dataframe=submodules)
