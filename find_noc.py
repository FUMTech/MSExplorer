# import os

# def count_lines(path):
#     """
#     Recursively counts the total number of lines in all files within a directory.
#     """
#     total_lines = 0
#     for root, dirs, files in os.walk(path):
#         for file in files:

#             if not file.endswith('.java'):
#                 continue

#             file_path = os.path.join(root, file)
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     lines = f.readlines()
#                     total_lines += len(lines)
#             except (UnicodeDecodeError, IsADirectoryError):
#                 # Ignore files that cannot be decoded or are directories
#                 pass
#     return total_lines


# folder_path = 'C:\\Users\\Amir\\Desktop\\PJ\\MonoMicroPJ\\MonoMicro\\roller'
# total_lines = count_lines(folder_path)
# print(f"Total lines of code in '{folder_path}': {total_lines}")


import os

def count_java_classes(path):
    """
    Recursively counts the total number of Java classes in all files within a directory.
    """
    total_classes = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        contents = f.read()
                        class_count = contents.count('class ')
                        total_classes += class_count
                except UnicodeDecodeError:
                    # Ignore files that cannot be decoded
                    pass
    return total_classes


# import os
# import re

# def count_java_classes(path):
#     """
#     Recursively counts the total number of Java classes in all files within a directory.
#     """
#     total_classes = 0
#     class_pattern = r'(?:^|\n)(?:public|private|protected)?\s*class\s+\w+\b'
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             if file.endswith('.java'):
#                 file_path = os.path.join(root, file)
#                 try:
#                     with open(file_path, 'r', encoding='utf-8') as f:
#                         contents = f.read()
#                         class_count = len(re.findall(class_pattern, contents))
#                         total_classes += class_count
#                 except UnicodeDecodeError:
#                     # Ignore files that cannot be decoded
#                     pass
#     return total_classes


# Example usage
# folder_path = 'C:\\Users\\Amir\\Desktop\\PJ\\MonoMicroPJ\\MonoMicro\\roller'
folder_path = 'C:\\Users\\Amir\\Downloads\\jpetstore-6-jpetstore-6.0.2'
total_classes = count_java_classes(folder_path)
print(f"Total Java classes in '{folder_path}': {total_classes}")


