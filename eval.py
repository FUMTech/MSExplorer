import scipy.io
import numpy as np
import os
import json
import jsonpickle
import networkx as nx
import javalang
from functools import lru_cache

java_system_classes = {
        'String', 'Object', 'Math', 'System', 'Thread', 'Exception', 'Error',
        'List', 'Map', 'Set', 'Integer', 'Long', 'Double', 'Float', 'Boolean',
        'Byte', 'Character', 'Short', 'Class', 'ClassLoader', 'Throwable',
        'InputStream', 'OutputStream', 'File', 'Runnable', 'Thread', 'Arrays',
        'Collections', 'HashMap', 'ArrayList', 'LinkedList', 'HashSet', 'TreeMap',
        'TreeSet', 'Optional', 'Stream', 'Date', 'Locale', 'Calendar', 'TimeZone',
        'RuntimeException', 'SQLException', 'IOException', 'InterruptedException',
        'NoSuchElementException', 'IndexOutOfBoundsException', 'ConcurrentModificationException',
        'NumberFormatException', 'IllegalArgumentException', 'IllegalStateException', 'UnsupportedOperationException','int','temp'
    }



def load_mat_data(file_path):
    mat = scipy.io.loadmat(file_path)
    unprocessed_best_sol = mat['BestSol'][0][0][0][0]
    class_names = mat['ClassName'][0][0]
    return unprocessed_best_sol, class_names

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_jsonpickle_data(file_path):
    with open(file_path, 'r') as file:
        data = jsonpickle.decode(file.read())
    return data


def custom_round(x):
    return np.floor(x) if x - np.floor(x) < 0.5 else np.ceil(x)

def group_classes_into_submodules(best_sol):
    submodules = {}

    # Iterate over each element in the array
    for class_index, submodule_index in enumerate(best_sol, start=1):
        # Create the submodule key as 'S' followed by the submodule index
        submodule_key = f"S{int(class_index)}"
        # Add the class index to the correct set in the dictionary
        # if submodule_key not in submodules:
        submodules[submodule_key] = set()
        submodules[submodule_key].add(class_index)
    return submodules

def prepare_submodules(best_sol):
    
    # Get the unique submodules sorted
    unique_submodules = np.unique(best_sol)

    # Initialize a list to hold the sets
    candidate_state = []

    # Iterate over the unique submodules
    for submodule in unique_submodules:
        # Find the indices where the current submodule occurs
        class_indices = np.where(best_sol == submodule)[0] + 1
        # Create a set with the formatted strings
        submodule_set = {"S{}".format(class_index) for class_index in class_indices}
        # Append the set to the output list
        candidate_state.append(submodule_set)
    return candidate_state

def find_IFN(set_dictionary, microservices, interface_relationships):
    counter = 0
    set_of_microservices = []
    class_to_submodule = {cls: submodule for submodule, classes in set_dictionary.items() for cls in classes}
    for class1, interface1, class2, interface2 in interface_relationships:
        class1_microservice = [ms for ms in microservices if class_to_submodule[class1] in ms][0]
        class2_microservice = [ms for ms in microservices if class_to_submodule[class2] in ms][0]
        if not set(class1_microservice) & set(class2_microservice):
            if (class1_microservice, class2_microservice) not in set_of_microservices:
                set_of_microservices.append((class1_microservice, class2_microservice))
            counter += 1
    len_unique = len(set_of_microservices)
    return 0 if len_unique == 0 else counter / len_unique


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

    num_comparisons = len(signatures1) * len(signatures2)
    if num_comparisons == 0:
        param_similarity = 0
        return_value_similarity = 0
    else: 
        param_similarity = total_param_similarity / (len(signatures1) * len(signatures2))
        return_value_similarity = total_return_value_similarity / (len(signatures1) * len(signatures2))
    
    return (param_similarity + return_value_similarity) / 2

def find_CHM(candidate_microservices,interface_relations  ,submodules, class_id_to_name, interfaces, lexical_info):
    microservice_chms = []

    def get_interface_code(interface_id):
        for file,info in lexical_info.items():
            for class_name in info['CN']:
                if  class_name.lower() == class_id_to_name[str(interface_id)]:
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

@lru_cache(maxsize=2024)
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

def find_CHD(candidate_microservices, interface_relationships, submodules, class_id_to_name, interfaces, lexical_info):
    microservice_chds = []

    def get_interface_code(interface_id):
        for file, info in lexical_info.items():
            for class_name in info['CN']:
                if class_name.lower() == class_id_to_name[str(interface_id)]:
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



# %%
from itertools import combinations

def smq(microservices, graph, submodules):

    mq = 0

    for ms in microservices:

        # Initialize a set to store all class IDs for the current microservice
        class_ids_for_microservice = set()
        for submodule in ms:
            # Update the set with class IDs for each submodule in the microservice
            class_ids_for_microservice.update(submodules.get(submodule, set()))
        # networkx.MultiDiGraph().nodes.
        # for node in graph.nodes():
        #     if int(node) in class_ids_for_microservice:
        #         print()
        nodes = [n for n in graph.nodes() if int(n) in class_ids_for_microservice]
        subgraph = graph.subgraph(nodes)
        edges = subgraph.edges()
        
        intra_edges = [e for e in edges if int(e[0]) in class_ids_for_microservice and int(e[1]) in class_ids_for_microservice] #Includes self calling

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


# %%
@lru_cache(maxsize=2024)
def extract_domain_terms_from_class(java_code): #Better to be caclulated by weights
    if java_code is None:
        domain_terms = set()
        return domain_terms
    """Extract domain terms from a Java class."""
    tree = javalang.parse.parse(java_code)
    domain_terms = set()

    for _, type_decl in tree.filter(javalang.parser.tree.TypeDeclaration):
        # Add class name
        domain_terms.add(type_decl.name)
        
        # Add annotations
        # if type_decl.annotations:
        #     domain_terms.update([annotation.name for annotation in type_decl.annotations])


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

        for _, method in type_decl.filter(javalang.parser.tree.MethodDeclaration):
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
            for _, local_var in method.filter(javalang.parser.tree.LocalVariableDeclaration):
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
        for _, field in type_decl.filter(javalang.parser.tree.FieldDeclaration):
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
        # for _, comment in type_decl.filter(javalang.parser.tree.Documented):
        #     if comment.documentation is not None:
        #         domain_terms.update(comment.documentation.split('\n'))
    filtered_terms = {term for term in domain_terms if term not in java_system_classes}
    return filtered_terms
    # return domain_terms


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

def find_CMQ(candidate_microservices, submodules, class_id_to_name, lexical_info):
    N = len(candidate_microservices)
    cohesion_values = []
    coupling_values = []

    def get_class_code(class_id):
        # This function retrieves the code for a given class_id
        for file, info in lexical_info.items():
            for class_name in info['CN']:
                if class_name.lower() == class_id_to_name[str(class_id)]:
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




def process_data(mat_data, json_data, jsonpickle_data):
    # Extracting necessary data
    unprocessed_best_sol = mat_data[0]
    class_id_to_name = json_data['class_indices']
    interface_relations = jsonpickle_data['interface_relations']
    interfaces = jsonpickle_data['interfaces']
    lexical_info = jsonpickle_data['lexical_info']
    G = jsonpickle_data['graph']

    # Custom rounding
    best_sol = np.array([custom_round(xi) for xi in unprocessed_best_sol])

    # Group classes into submodules
    submodules = group_classes_into_submodules(best_sol)

    # Find unique submodules and prepare them for further analysis
    candidate_state = prepare_submodules(best_sol)

    # Calculate IFN Metric
    IFN = find_IFN(submodules, candidate_state, interface_relations)
    CHM = find_CHM(candidate_state, interface_relations, submodules, class_id_to_name, interfaces, lexical_info)
    CHD = find_CHD(candidate_state, interface_relations, submodules, class_id_to_name, interfaces, lexical_info)
    SMQ = smq(candidate_state, G, submodules)
    CMQ = find_CMQ(candidate_state, submodules, class_id_to_name, lexical_info)


    print("IFN =", IFN)
    print("CHM =", CHM)
    print("CHD =", CHD)
    print("SMQ =", SMQ)
    print("CMQ =", CMQ)
    
    # Additional metric calculations can be added here

    return (IFN,CHM,CHD,SMQ,CMQ)


json_file_path = "C:\\Users\\Amir\\Desktop\\PJ\\MonoMicroPJ\\MonoMicro\\coheision-extractor\\jpetstore_coupling.json"
jsonpickle_file_path = "C:\\Users\\Amir\\Desktop\\PJ\\MonoMicroPJ\\MonoMicro\\coheision-extractor\\jpetstore_eval.json"

json_data = load_json_data(json_file_path)
jsonpickle_data = load_jsonpickle_data(jsonpickle_file_path)

def process_directory(directory_path):

    for root, dirs, files in os.walk(directory_path):
        collected_metrics = {
                                'IFN': [],
                                'CHM': [],
                                'CHD': [],
                                'SMQ': [],
                                'CMQ': []
                            }
        for file in files:
            if file.endswith('.mat'):
                mat_file_path = os.path.join(root, file)

                mat_data = load_mat_data(mat_file_path)


                metrics = process_data(mat_data, json_data, jsonpickle_data)

                # Append each metric to its respective list in the dictionary
                collected_metrics['IFN'].append(metrics[0])
                collected_metrics['CHM'].append(metrics[1])
                collected_metrics['CHD'].append(metrics[2])
                collected_metrics['SMQ'].append(metrics[3])
                collected_metrics['CMQ'].append(metrics[4])

        # Check if the directory contained any .mat files to process
        if collected_metrics['IFN']:
            # Calculate the average for each metric in this directory
            averages = {metric: np.mean(values) for metric, values in collected_metrics.items() if values}

            # Construct the path for the results file
            results_file_path = os.path.join(root, 'results.txt')
            
            # Write the results to a file in the current directory
            with open(results_file_path, 'w') as result_file:
                for metric, average in averages.items():
                    result_file.write(f'Average {metric}: {average}\n')


if __name__ == "__main__":
    root_directory = 'C:\\Users\\Amir\\Desktop\\PJ\\MonoMicroPJ\\MonoMicro\\JPetstore_k=4'
    process_directory(root_directory)