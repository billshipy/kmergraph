# version: 1.1 
# changes; 1.0 -> 1.1: Added ability to "hide" edges below a certain threshold

def save_graph_to_dot(nodes_arg, edges_arg, color_dictionary_arg, filename_arg, edge_visibility_arg):
    dot_output_file = open(filename_arg, "w")
    dot_output_file.write("graph G {\n")
    dot_output_file.write("graph [overlap=false,splines=true,size=\"4,4\"];\n")
    dot_output_file.write("node [shape=circle,fixedsize=true,width=0.35];\n");
    for node in nodes_arg.keys():
        node_color = "#000000"
        if node in color_dictionary_arg:
            node_color = color_dictionary_arg[node]
        dot_output_file.write(
            node + " [label = \"\" style=filled color=\"" + node_color + "\" fillcolor=\"" + node_color + "\"];\n")
    for edge_index in range(0, len(edges_arg)):
        edge_data = edges_arg[edge_index]
        first_node = edge_data[0]
        second_node = edge_data[1]
        edge_weight = float(edge_data[2])
        if (edge_weight >= edge_visibility_arg):
            dot_output_file.write(first_node + " -- " + second_node + " [len=" + str(2.0 - edge_weight) + "];\n")
        else:
            dot_output_file.write(first_node + " -- " + second_node + " [len=" + str(2.0 - edge_weight) + " style=\"invis\"];\n")
    dot_output_file.write("}")
    dot_output_file.close()
    return


def read_graph_from_edge_list(filename_arg):
    node_dict = {}
    edge_list = []
    graph_input_file = open(filename_arg, "r")
    for line in graph_input_file:
        data = tuple(line.split())
        print(data)
        node = data[0]
        if node not in node_dict:
            node_dict[node] = ""
        node = data[1]
        if node not in node_dict:
            node_dict[node] = ""
        edge_list.append(data)
    return node_dict, edge_list


def read_color_dictionary(filename_arg):
    color_dict = {}
    color_input_file = open(filename_arg, "r")
    for line in color_input_file:
        line = line.strip()
        if len(line) > 0:
            data = tuple(line.split(","))
            color_dict[data[0]] = data[1]
    return color_dict


# main starts here

### UPDATE THESE FIVE VARIABLES ###
# The folder and name of your color file (mine is in the same folder as this python file)
color_file_name = "colors.txt"   
# The number of samples
number_of_samples = 100
# The folder where the Prx1 examples are
prx1_folder_name = "Prx1TrimWith10Nodes"
# The folder where the PrxQ examples are
prxq_folder_name = "PrxQTrimWith10Nodes"
# The edge weight threshold for visibility (>= this value, appears; <, invisible)
edge_visibility_threshold = 0.4


colors = read_color_dictionary(color_file_name)
neato_script_output_file = open("batchneato.sh", "w")
for index in range(0, number_of_samples):
    input_graph_filename = prx1_folder_name +  "/sample" + str(index) + "JaccSorted.txt"
    output_dot_filename = prx1_folder_name + "/sample" + str(index) + ".dot"
    output_png_filename = prx1_folder_name + "/Prx1_sample" + str(index) + ".png"
    nodes, edges = read_graph_from_edge_list(input_graph_filename)
    save_graph_to_dot(nodes, edges, colors, output_dot_filename, edge_visibility_threshold)
    neato_script_output_file.write("neato -Tpng " + output_dot_filename + " -o " + output_png_filename + "\n")
for index in range(0, number_of_samples):
    input_graph_filename = prxq_folder_name + "/sample" + str(index) + "JaccSorted.txt"
    output_dot_filename = prxq_folder_name + "/sample" + str(index) + ".dot"
    output_png_filename = prxq_folder_name + "/PrxQ_sample" + str(index) + ".png"
    nodes, edges = read_graph_from_edge_list(input_graph_filename)
    save_graph_to_dot(nodes, edges, colors, output_dot_filename, edge_visibility_threshold)
    neato_script_output_file.write("neato -Tpng " + output_dot_filename + " -o " + output_png_filename + "\n")
neato_script_output_file.close()
