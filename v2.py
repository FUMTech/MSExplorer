# Required Installations
# ! pip install --upgrade mysql-connector-python javalang
# ! pip install transformers torch torchvision torchaudio sentence-transformers python-Levenshtein

import os
import re
import json
import numpy as np
import pandas as pd
import javalang
import networkx as nx
import multiprocessing
from collections import defaultdict
from itertools import combinations
from sentence_transformers import SentenceTransformer, util
from sklearn import preprocessing
import jsonpickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
directory_path = "/home/amir/Desktop/PJ/MonoMicro/xwiki-platform-xwiki-platform-10.8"
traces_file_path = '/home/amir/Desktop/PJ/MonoMicro/FoSCI-master/traces/xwiki-platform/xwiki-platform_trace.txt'
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Global Variables
class_id_to_name = {}
all_classes = []
interfaces = []
has_parameter_results = []
is_of_type_results = []
referece_results = []
call_results = []
implement_results = []
return_results = []
inheritance_results = []

# Functions
def parse_java_file(file_path):
    with open(file_path, 'r') as file:
        java_code = file.read()
    return javalang.parse.parse(java_code)

def extract_comments(file_path):
    with open(file_path, 'r') as file:
        java_code = file.read()
    pattern = r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"|\w+'
    matches = re.findall(pattern, java_code, re.DOTALL | re.MULTILINE)
    return [match for match in matches if match.startswith('//') or match.startswith('/*')]

def resolve_type(tree, type_node, current_class):
    if isinstance(type_node, javalang.parser.tree.Type):
        type_name = type_node.name
        if '.' in type_name:
            return type_name
        elif is_nested_class(type_name, current_class):
            return f"{current_class.name}.{type_name}"
        else:
            return resolve_imported_type(tree, type_name)
    elif isinstance(type_node, javalang.parser.tree.TypeArgument):
        return resolve_type(tree, type_node.type, current_class)
    elif isinstance(type_node, javalang.parser.tree.TypeParameter):
        return resolve_type(tree, type_node.name, current_class)
    elif isinstance(type_node, str):
        return type_node
    else:
        raise ValueError(f"Unsupported type node: {type_node}")

def is_nested_class(type_name, current_class):
    for nested_class in current_class.body:
        if isinstance(nested_class, javalang.parser.tree.ClassDeclaration) and nested_class.name == type_name:
            return True
    return False

def resolve_imported_type(tree, type_name):
    import_declarations = [node for node in tree.imports if isinstance(node, javalang.parser.tree.Import)]
    for import_decl in import_declarations:
        if type_name == import_decl.path:
            return import_decl.path
        elif '.' in import_decl.path:
            package_name, imported_type = import_decl.path.rsplit('.', 1)
            if imported_type == type_name:
                return import_decl.path
    return type_name

def find_structural_dependencies(tree, all_classes, variable_types):
    class_package = tree.package.name if tree.package else 'default'
    for _, a_class in tree.filter(javalang.parser.tree.ClassDeclaration):
        analyze_methods(tree, a_class, class_package, all_classes, variable_types)
        analyze_references(tree, a_class, class_package, all_classes, variable_types)
        analyze_fields(tree, a_class, class_package, all_classes, variable_types)
        analyze_inheritance(tree, a_class, class_package, all_classes)

def analyze_methods(tree, a_class, class_package, all_classes, variable_types):
    for _, method in a_class.filter(javalang.parser.tree.MethodDeclaration):
        for param in method.parameters:
            param_type = param.type.name
            matches = [each_class for each_class in all_classes if each_class[1] == param_type]
            if matches:
                for match in matches:
                    has_parameter_results.append(([c[0] for c in all_classes if c[1] == a_class.name][0], class_package, match[0], match[2]))
        if method.return_type:
            return_type = method.return_type.name
            matches = [each_class for each_class in all_classes if each_class[1] == return_type]
            if matches:
                for match in matches:
                    return_results.append(([c[0] for c in all_classes if c[1] == a_class.name][0], class_package, match[0], match[2]))

def analyze_references(tree, a_class, class_package, all_classes, variable_types):
    for _, ref in a_class.filter(javalang.parser.tree.MemberReference):
        matches = [each_class for each_class in all_classes if each_class[1] == ref.qualifier]
        if matches:
            for match in matches:
                referece_results.append((match[0], match[2], [c[0] for c in all_classes if c[1] == a_class.name][0], class_package))
    for _, method_invocation in a_class.filter(javalang.parser.tree.MethodInvocation):
        analyze_method_invocation(tree, method_invocation, a_class, class_package, all_classes, variable_types)

def analyze_method_invocation(tree, method_invocation, a_class, class_package, all_classes, variable_types):
    qualifier = method_invocation.qualifier
    if isinstance(qualifier, str) and qualifier in variable_types:
        class_name = resolve_type(tree, variable_types[qualifier], a_class)
        matches = [each_class for each_class in all_classes if each_class[1] == class_name]
        if matches:
            for match in matches:
                call_results.append(([c[0] for c in all_classes if c[1] == a_class.name][0], class_package, match[0], match[2]))
    elif isinstance(qualifier, javalang.parser.tree.Literal):
        class_name = resolve_type(method_invocation.member.split('.')[0], a_class)
        matches = [each_class for each_class in all_classes if each_class[1] == class_name]
        if matches:
            for match in matches:
                call_results.append(([c[0] for c in all_classes if c[1] == a_class.name][0], class_package, match[0], match[2]))

def analyze_fields(tree, a_class, class_package, all_classes, variable_types):
    for _, field in a_class.filter(javalang.parser.tree.FieldDeclaration):
        if field.type:
            field_type = field.type.name
            matches = [each_class for each_class in all_classes if each_class[1] == field_type]
            if matches:
                for match in matches:
                    is_of_type_results.append(([c[0] for c in all_classes if c[1] == a_class.name][0], class_package, match[0], match[2]))

def analyze_inheritance(tree, a_class, class_package, all_classes):
    if a_class.extends:
        class_name = resolve_type(tree, a_class.extends.name, a_class)
        matches = [each_class for each_class in all_classes if each_class[1] == class_name]
        if matches:
            for match in matches:
                inheritance_results.append((match[0], match[2], [c[0] for c in all_classes if c[1] == a_class.name][0], class_package))
    if a_class.implements:
        for implemented_interface in a_class.implements:
            interface_name = resolve_type(tree, implemented_interface, a_class)
            matches = [each_class for each_class in all_classes if each_class[2] + '.' + each_class[1] == interface_name]
            if matches:
                for match in matches:
                    implement_results.append((match[0], match[2], [c[0] for c in all_classes if c[1] == a_class.name][0], class_package))

def extract_lexical_information(java_tree):
    class_info = defaultdict(list)
    for path, node in java_tree:
        try:
            if isinstance(node, javalang.parser.tree.ClassDeclaration):
                class_info['CN'].append(node.name)
            elif isinstance(node, javalang.parser.tree.FieldDeclaration):
                class_info['AN'].extend([field.name for field in node.declarators])
            elif isinstance(node, javalang.parser.tree.MethodDeclaration):
                class_info['MN'].append(node.name)
                class_info['PN'].extend([param.name for param in node.parameters])
            extract_source_code_statements(node, class_info)
        except Exception as e:
            print(f"Failed to parse {node} due to {str(e)}")
    return class_info

def extract_source_code_statements(node, class_info):
    if isinstance(node, javalang.parser.tree.ClassReference):
        class_info['SCS_ClassReference'].append(node.type.name)
    elif isinstance(node, javalang.parser.tree.MemberReference):
        class_info['SCS_MemberReference'].append(node.member)
    elif isinstance(node, javalang.parser.tree.MethodReference):
        class_info['SCS_MethodReference'].append(node.method.member)
    elif isinstance(node, javalang.parser.tree.VoidClassReference):
        class_info['SCS_VoidClassReference'].append(node.name)
    elif isinstance(node, javalang.parser.tree.SuperMemberReference):
        class_info['SCS_SuperMemberReference'].append(node.member)
    elif isinstance(node, javalang.parser.tree.ConstantDeclaration):
        class_info['SCS_ConstantDeclaration'].append(node.name)
    elif isinstance(node, javalang.parser.tree.VariableDeclaration):
        class_info['SCS_VariableDeclaration'].append(node.type.name)
    elif isinstance(node, javalang.parser.tree.VariableDeclarator):
        class_info['SCS_VariableDeclarator'].append(node.name)
    elif isinstance(node, javalang.parser.tree.AnnotationDeclaration):
        class_info['SCS_AnnotationDeclaration'].append(node.name)
    elif isinstance(node, javalang.parser.tree.ConstructorDeclaration):
        class_info['SCS_ConstructorDeclaration'].append(node.name)
    elif isinstance(node, javalang.parser.tree.LocalVariableDeclaration):
        class_info['SCS_LocalVariableDeclaration'].append(node.name)
    elif isinstance(node, javalang.parser.tree.MethodInvocation):
        class_info['SCS_ClassReference'].append(node.qualifier)
        class_info['SCS_MethodInvocation'].append(node.member)
    elif isinstance(node, javalang.parser.tree.FieldDeclaration):
        class_info['SCS_FieldDeclaration'].append(node.type.name)
    elif isinstance(node, javalang.parser.tree.MethodDeclaration):
        class_info['SCS_MethodDeclaration'].append(node.return_type.name)
    elif isinstance(node, javalang.parser.tree.EnumDeclaration):
        class_info['CN'].append(node.name)
    elif isinstance(node, javalang.parser.tree.InterfaceDeclaration):
        class_info['CN'].append(node.name)

def analyze_directory(directory):
    all_class_info = {}
    i = 0
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
                        extract_class_info(java_tree, class_package, variable_types, i)
                    except Exception as e:
                        print(f"Failed to parse {file_path} due to {str(e)}")
    analyze_files_in_directory(directory, all_class_info, variable_types)
    return all_class_info

def extract_class_info(java_tree, class_package, variable_types, i):
    for _, node in java_tree:
        if isinstance(node, javalang.parser.tree.ClassDeclaration):
            i += 1
            all_classes.append((i, node.name, class_package))
        elif isinstance(node, javalang.parser.tree.InterfaceDeclaration):
            i += 1
            all_classes.append((i, node.name, class_package))
            interfaces.append(i)
        elif isinstance(node, javalang.parser.tree.EnumDeclaration):
            i += 1
            all_classes.append((i, node.name, class_package))
        extract_variable_types(node, variable_types)

def extract_variable_types(node, variable_types):
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
            if 'qualifier' in param.attrs and param.qualifier != '':
                variable_name = param.member
                variable_type = param.qualifier
                variable_types[variable_name] = variable_type
    elif isinstance(node, javalang.parser.tree.LocalVariableDeclaration):
        for declarator in node.declarators:
            variable_name = declarator.name
            variable_type = node.type.name
            variable_types[variable_name] = variable_type

def analyze_files_in_directory(directory, all_class_info, variable_types):
    for root, dirs, files in os.walk(directory):
        if 'test' not in root.lower():
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    java_tree = parse_java_file(file_path)
                    class_info = extract_lexical_information(java_tree)
                    find_structural_dependencies(java_tree, all_classes, variable_types)
                    comments = extract_comments(file_path)
                    class_info['CO'].extend(comments)
                    with open(file_path, 'r') as file:
                        java_code = file.read()
                    class_info['CODE'] = java_code
                    all_class_info[file] = class_info

def find_interface_relations(class_couplings):
    interface_relations = []
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

def convert_class_id_to_name(lexical_info):
    new_lexical_info = {}
    for c_id, c_name, c_dir in all_classes:
        new_lexical_info[c_id] = defaultdict(list)
        curr_class_name = ''
        for file, info in lexical_info.items():
            if curr_class_name != '':
                break
            for cn in info['CN']:
                if c_name == cn:
                    curr_class_name = file
                    break
        if curr_class_name:
            new_lexical_info[c_id].update(lexical_info[curr_class_name])
    return new_lexical_info

def filter_out_unwanted_comments(comments):
    return [comment.replace('.', '').replace('/', '').replace("\\", '') for comment in comments if "copyright" not in comment.lower() and 'author' not in comment.lower() and 'licensed' not in comment.lower()]

def compute_comment_similarity(class1_comments, class2_comments):
    class1_embeddings = [model.encode(comment, convert_to_tensor=True) for comment in class1_comments]
    class2_embeddings = [model.encode(comment, convert_to_tensor=True) for comment in class2_comments]
    if not class1_embeddings or not class2_embeddings:
        return 0
    similarities = [util.pytorch_cos_sim(embed1, embed2).item() for embed1 in class1_embeddings for embed2 in class2_embeddings]
    return sum(similarities) / len(similarities)

def calculate_similarity(file1, file2, category):
    doc1 = " ".join([str(element) for element in new_lexical_info[file1][category]])
    doc2 = " ".join([str(element) for element in new_lexical_info[file2][category]])
    if category == "CO":
        class1_comments = filter_out_unwanted_comments(new_lexical_info[file1][category])
        class2_comments = filter_out_unwanted_comments(new_lexical_info[file2][category])
        return compute_comment_similarity(class1_comments, class2_comments)
    if "SCS" in category:
        return difflib.SequenceMatcher(None, doc1, doc2).ratio()
    data_types_and_classes = ["byte", "short", "int", "long", "float", "double", "boolean", "char", "string", "list", "map", "set", "arraylist", "hashmap", "hashset", 'integer']
    pattern = r"\b(" + "|".join(data_types_and_classes) + r")\b"
    elements_file1 = [re.sub(pattern, '', item.lower()).strip() for item in new_lexical_info[file1][category] if item]
    elements_file2 = [re.sub(pattern, '', item.lower()).strip() for item in new_lexical_info[file2][category] if item]
    total_distances = sum([Levenshtein.distance(str(element1), str(element2)) / max(len(str(element1)), len(str(element2))) for element1 in elements_file1 for element2 in elements_file2])
    return (1 - total_distances / (len(elements_file1) * len(elements_file2))) if elements_file1 and elements_file2 else 0

def add_edge(G, node1, node2, weight, type_of_relation):
    if G.has_edge(node1, node2):
        if type_of_relation in G[node1][node2]:
            G[node1][node2][type_of_relation]['weight'] += weight
        else:
            G[node1][node2][type_of_relation] = {'weight': weight}
    else:
        G.add_edge(node1, node2)
        G[node1][node2][type_of_relation] = {'weight': weight}

def discover_inter_coupling_classes(G):
    for node1, node2, attrs in G.edges(data=True):
        sum_weights = sum([type_of_relation.get('weight', 0) for type_of_relation in attrs.values()])
        if get_directory(node1) != get_directory(node2):
            add_edge(G_inter, node1, node2, sum_weights, 'all')
        else:
            add_edge(G_intra, node1, node2, sum_weights, 'all')

# Main Execution
if __name__ == '__main__':
    lexical_info = analyze_directory(directory_path)

    class_couplings = (inheritance_results + return_results + implement_results + call_results +
                       referece_results + is_of_type_results + has_parameter_results)

    interface_relations = find_interface_relations(class_couplings)

    manager = multiprocessing.Manager()
    co_occurrence_matrix = {item[0]: {other_item[0]: manager.Value('i', 0) for other_item in all_classes} for item in all_classes}
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
    traces = {trace_id: trace for chunk_result in chunk_results for trace_id, trace in chunk_result.items()}
    pool.starmap(update_co_occurrence_matrix, [(trace, co_occurrence_matrix) for trace in traces.values()])
    pool.close()
    pool.join()

    co_occurrence_matrix = {k: {inner_k: inner_v.value for inner_k, inner_v in v.items()} for k, v in co_occurrence_matrix.items()}

    transformed_data = []
    for class_name, co_occurrences in co_occurrence_matrix.items():
        row = {'class': class_name}
        row.update(co_occurrences)
        transformed_data.append(row)

    class_co_occurrences_df = pd.DataFrame(transformed_data).set_index('class')
    class_co_occurrences_matrix = np.where(class_co_occurrences_df == 0, 1, class_co_occurrences_df)

    n = len(all_classes)
    adj_matrix = np.zeros((n, n), dtype=int)

    def populate_adj_matrix(couplings, weight):
        for source_class_id, source_module_name, referenced_class_id, referenced_module_name in couplings:
            adj_matrix[source_class_id - 1, referenced_class_id - 1] += weight
            adj_matrix[referenced_class_id - 1, source_class_id - 1] += weight

    populate_adj_matrix(inheritance_results, 8.5)
    populate_adj_matrix(return_results, 1)
    populate_adj_matrix(implement_results, 2)
    populate_adj_matrix(call_results, 2.5)
    populate_adj_matrix(referece_results, 3)
    populate_adj_matrix(is_of_type_results, 2)
    populate_adj_matrix(has_parameter_results, 3.5)

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

    def add_couplings_to_graph(graph, couplings, weight, relation_type):
        for source_id, source_module, ref_id, ref_module in couplings:
            add_edge(graph, source_id, ref_id, weight, relation_type)

    add_couplings_to_graph(G, inheritance_results, 8.5, 'inheritance')
    add_couplings_to_graph(G, return_results, 1, 'return')
    add_couplings_to_graph(G, implement_results, 2, 'implement')
    add_couplings_to_graph(G, call_results, 2.5, 'call')
    add_couplings_to_graph(G, referece_results, 3, 'reference')
    add_couplings_to_graph(G, is_of_type_results, 2, 'is_of_type')
    add_couplings_to_graph(G, has_parameter_results, 3.5, 'has_parameter')

    discover_inter_coupling_classes(G)

    in_degrees = dict(G_inter.in_degree())
    out_degrees = dict(G_inter.out_degree())
    inter_coupling_nodes = sorted([node for node in G if G_inter.has_node(node) and sum([G_inter[u][node]['all']['weight'] for u in G_inter.predecessors(node)]) + sum([G_inter[node][v]['all']['weight'] for v in G_inter.successors(node)]) >= 0], key=lambda node: (in_degrees[node], out_degrees[node]))

    new_lexical_info = convert_class_id_to_name(lexical_info)

    coefficients = {'CN': 0.1413, 'AN': 0.1113, 'MN': 0.1313, 'PN': 0.1413, 'SCS_MethodDeclaration': 0.1750, 'SCS_ClassReference': 0.1750, 'SCS_MemberReference': 0.1750,
                    'SCS_MethodReference': 0.1750, 'SCS_VoidClassReference': 0.1750, 'SCS_SuperMemberReference': 0.1750,
                    'SCS_ConstantDeclaration': 0.1750, 'SCS_VariableDeclaration': 0.1750, 'SCS_VariableDeclarator': 0.1750,
                    'SCS_AnnotationDeclaration': 0.1750, 'SCS_ConstructorDeclaration': 0.1750,
                    'SCS_LocalVariableDeclaration': 0.1750, 'SCS_MethodInvocation': 0.1750,
                    'SCS_FieldDeclaration': 0.1750, 'CO': 0.2225}

    total_similarity = np.zeros((len(new_lexical_info.items()), len(new_lexical_info.items())))

    for i, module1 in enumerate(new_lexical_info):
        for j, module2 in enumerate(new_lexical_info):
            if i <= j:
                total_similarity_ij = sum([coefficients[category] * calculate_similarity(module1, module2, category) for category in coefficients])
                total_similarity[i, j] = total_similarity_ij
                total_similarity[j, i] = total_similarity_ij

    submodule_count = 1
    submodules = defaultdict(set)
    nodes_in_submodules = set()
    directories = defaultdict(set)

    for node in G.nodes():
        directories[get_directory(node)].add(node)

    for node in inter_coupling_nodes:
        directory = get_directory(node)
        if node in G:
            related_nodes = directories[directory] - set(inter_coupling_nodes)
            column = total_similarity[:, node - 1]
            rows_to_consider = list(related_nodes)
            intra_coupling_threshold = min(2, len(rows_to_consider))
            indices = np.argpartition(column, -intra_coupling_threshold)[-intra_coupling_threshold:]
            sorted_indices = indices[np.argsort(column[indices])[::-1]]
            related_nodes = [node] + [index + 1 for index in sorted_indices]
            subgraph = G.subgraph(related_nodes)
            nodes = nx.dfs_preorder_nodes(subgraph, node)
            related_files = list(nodes)

        related_files = [f for f in related_files if f not in nodes_in_submodules]
        nodes_in_submodules.update(related_files)

        if related_files:
            submodules[f'S{submodule_count}'].update(related_files)
            submodule_count += 1

    for node in G.nodes():
        if node not in nodes_in_submodules:
            submodules[f'S{submodule_count}'].update([node])
            submodule_count += 1

    scaler = preprocessing.MinMaxScaler()
    normalized_conceptual_matrix = scaler.fit_transform(total_similarity.reshape(-1, 1)).reshape(total_similarity.shape)
    np.fill_diagonal(normalized_conceptual_matrix, 0)
    normalized_structural_matrix = scaler.fit_transform(adj_matrix.reshape(-1, 1)).reshape(adj_matrix.shape)
    coupling = normalized_conceptual_matrix * 0.2 + normalized_structural_matrix * 0.8

    co_occurrence_matrix_serializable = {str(k): {str(inner_k): inner_v for inner_k, inner_v in v.items()} for k, v in co_occurrence_matrix.items()}

    with open('coupling.json', 'w') as file:
        json.dump({
            'conceptual_coupling_matrix': total_similarity.tolist(),
            'normalized_conceptual_coupling_matrix': normalized_conceptual_matrix.tolist(),
            'normalized_structural_coupling_matrix': normalized_structural_matrix.tolist(),
            'structural_coupling_matrix': adj_matrix.tolist(),
            'submodules': {key: list(value) for key, value in submodules.items()},
            'class_indices': {index: class_name for index, class_name, _ in all_classes},
            'class_co_occurrences_in_execution_traces': co_occurrence_matrix_serializable
        }, file, indent=4)

    json_data = jsonpickle.encode({'lexical_info': lexical_info, 'all_classes': all_classes, 'interface_relations': interface_relations, 'interfaces': interfaces, 'submodules': submodules, 'graph': G})
    with open('eval.json', 'w') as f:
        f.write(json_data)
