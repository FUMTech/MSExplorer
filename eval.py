import scipy.io
import numpy as np
import networkx
import json

# Replace 'your_file.mat' with the path to your .mat file
mat = scipy.io.loadmat('/home/amir/Desktop/PJ/MonoMicro/JPetstore_k=4/workspace_60_JpetStore_1_run_.mat')
unprocessed_best_sol = mat['BestSol'][0][0][0][0]
class_names = mat['ClassName'][0][0]
# print(unprocessed_best_sol)
# mat is a dict with variable names as keys, and loaded matrices as values

# Read JSON data from the file
with open('/home/amir/Desktop/PJ/MonoMicro/coheision-extractor/jpetsore_coupling.json', 'r') as file:
    data = json.load(file)

# Restore the objects
conceptual_coupling_matrix = np.array(data['conceptual_coupling_matrix'])
normalized_conceptual_coupling_matrix = np.array(data['normalized_conceptual_coupling_matrix'])
normalized_structural_coupling_matrix = np.array(data['normalized_structural_coupling_matrix'])
structural_coupling_matrix = np.array(data['structural_coupling_matrix'])
# submodules = {key: set(value) for key, value in data['submodules'].items()}
class_id_to_name = dict(data['class_indices'])
class_co_occurrences_in_execution_traces = data['class_co_eccurances_in_execution_traces']

# Now you have the objects restored and you can use them as before
coupling = 0.5 * normalized_conceptual_coupling_matrix + 0.5 *  normalized_structural_coupling_matrix


import jsonpickle

# Load JSON data from a file
with open('jpetsore_eval.json', 'r') as f:
    json_data = f.read()

# Deserialize with jsonpickle
restored_objects = jsonpickle.decode(json_data)

# Access the restored objects
lexical_info = restored_objects['lexical_info']
all_classes = restored_objects['all_classes']
interface_relations = restored_objects['interface_relations']
interfaces = restored_objects['interfaces']
submodules = restored_objects['submodules']
G = restored_objects['graph']

# Now you can use lexical_info, all_classes, interface_relations, interfaces, submodules, and G as before

def custom_round(x):
    if x - np.floor(x) < 0.5:
        return np.floor(x)
    else:
        return np.ceil(x)

# Apply the custom rounding rule to each element
best_sol = np.array([custom_round(xi) for xi in unprocessed_best_sol])
# print(best_sol)



# Initialize an empty dictionary to hold the submodules and class indices
submodules = {}

# Iterate over each element in the array
for class_index, submodule_index in enumerate(best_sol, start=1):
    # Create the submodule key as 'S' followed by the submodule index
    submodule_key = f"S{int(class_index)}"
    # Add the class index to the correct set in the dictionary
    # if submodule_key not in submodules:
    submodules[submodule_key] = set()
    submodules[submodule_key].add(class_index)




import numpy as np

# Input array

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

# Display the resulting list of sets
print(candidate_state)


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



IFN = find_IFN(set_dictionary=submodules,microservices=candidate_state,interface_relationships=interface_relations)
print("IFN = ",IFN)



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

# Assuming you have a dictionary mapping class IDs to their respective Java source codes




CHM = find_CHM(candidate_microservices=candidate_state, interface_relationships=interface_relations )
print("CHM = ",CHM)


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

CHD = find_CHD(candidate_microservices=candidate_state, interface_relationships=interface_relations)
print("CHD = ",CHD)


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

SMQ = smq(candidate_state,G)
print("SMQ = ",SMQ)


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

CMQ = find_CMQ(candidate_microservices=candidate_state)
print("CMQ = ",CMQ)


# %%
# import git

# repo_path = '/home/amir/Desktop/PJ/MonoMicro/jpetstore-6-jpetstore-6.0.2'

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
#             class_name = item.split('/')[-1].replace('.java', '')
#             changed_files.add(class_name)
#     if len(changed_files) > 1 : # If one class in a commit changes, doesn't it mean that it is independent?
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



# def calculate_ICF(microservices, commit_history):
#     co_change_count = count_co_changes(commit_history)
#     total_icfm = 0

#     for microservice in microservices:

#         # Initialize a set to store all class IDs for the current microservice
#         classes = set()
#         for submodule in microservice:
#             # Update the set with class IDs for each submodule in the microservice
#             classes.update(submodules.get(submodule, set()))

#         icfm = 0

#         if len(microservice) > 1:
#             for class1, class2 in combinations(classes, 2):
#                 icfm += co_change_count.get((class_id_to_name[class1], class_id_to_name[class2]), 0)
#         else:
#             icfm = 1

#         icfm /= len(classes) ** 2
#         total_icfm += icfm

#     ICF = total_icfm / len(microservices)
#     return ICF



# ICF = calculate_ICF(candidate_state, commit_history)
# print("ICF = ",ICF)


# # %%

# # Assuming commit_history is already populated



# def compute_ecf(microservices, co_changes):

#     total_ecfm = 0

#     all_classes = set()
#     for microservice in microservices:
#         for submodule in microservice:
#             # Update the set with class IDs for each submodule in the microservice
#             all_classes.update(submodules.get(submodule, set()))

#     for microservice in microservices:
#         Cm = set()
#         for submodule in microservice:
#             # Update the set with class IDs for each submodule in the microservice
#             Cm.update(submodules.get(submodule, set()))

#         Cm_prime = all_classes - Cm
#         sum_f_cmt = 0

#         for ci in Cm:
#             for cj in Cm_prime:
#                 pair = tuple(sorted([class_id_to_name[ci],class_id_to_name[cj]]))
#                 sum_f_cmt += co_changes.get(pair, 0)

#         ecfm = (1 / len(Cm)) * (1 / len(Cm_prime)) * sum_f_cmt
#         total_ecfm += ecfm

#     ECF = total_ecfm / len(microservices)
#     return ECF



# co_change_count = count_co_changes(commit_history)
# ECF = compute_ecf(candidate_state, co_change_count)

# print("ECF = ",ECF)



# REI = ECF / ICF

# print("REI = ",REI)

