# Sapir Savariego 209534379
# Ofir Azriel 209532688

import random
import string
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left
import time
import tkinter as tk

SPECIAL_CHARS = (' ', '.', ',', ';', '\n')
ALL_CHARS = tuple(string.ascii_lowercase)
population_size = 100
percentage_to_keep = 5
percentage_of_mutations = 25
number_of_optimization_steps = 10
num_of_generation_to_fix_convergence = 10
definition_of_plateau = 40
plateau_of_maximum = 20
activate_fitness_func_count = 0
steps_of_graph = 1
gap_of_plateau = 10
method_of_running ="random"


def compare_results(max_fitness_scores ,average_fitness_scores, activate_fitness, technique, com_values, type_of_compression, name):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs = axs.flatten()

    colors = ['C9', 'C4', 'C3', 'C2', 'C0', 'C1', 'C5', 'C6', 'C7', 'C8']
    handles = []



    # plot max fitness
    ax = axs[0]
    for i, scores in enumerate(max_fitness_scores):
        line, = ax.plot(scores, label=f'%s {com_values[i]}' %type_of_compression, color=colors[i])
        handles.append(line)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Max fitness score')
    ax.set_title('Max fitness scores - %s' % technique)
    ax.legend(loc='lower right')

    # plot max fitness
    ax = axs[1]
    for i, scores in enumerate(average_fitness_scores):
        line, = ax.plot(scores, label=f'%s {com_values[i]}' % type_of_compression, color=colors[i])
        handles.append(line)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average fitness score')
    ax.set_title('Average fitness scores - %s' % technique)
    ax.legend(loc='lower right')

    # plot fitness calls
    ax = axs[2]
    x_pos = range(1, len(com_values) + 1)
    width = 0.4
    bars = ax.bar(x_pos, activate_fitness, width, color=colors[:len(x_pos)])
    ax.set_xticks(x_pos)  # Set the x-axis ticks to match the values in x_pos
    ax.set_xticklabels(com_values)  # Set the x-axis tick labels to the names of the bars
    ax.set_xlabel('%s values' %type_of_compression)
    ax.set_ylabel('Number of fitness calls')
    ax.set_title('Fitness calls - %s' % technique)
    # ax.legend(bars, ['Fitness Calls'])

    # Add labels to the bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                str(height), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
    fig.savefig('%s.png' % name)


class VariableInputGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry(f"{450}x{250}")
        self.root.title("Genetic Algorithm")
        self.root.config(bg="black")

        # Create labels and entry boxes for each variable
        self.t_label = tk.Label(self.root, text="Welcome to the\n Genetic Algorithm!\n\n")
        self.t_label.config(bg="black", fg="slateBlue2", font=("song ti", 30, "bold"))
        self.p_label = tk.Label(self.root, text="Population size:\n")
        self.p_label.config(bg="black", fg="seashell2", font=("Arial", 12))
        self.p_entry = tk.Entry(self.root)
        self.p_entry.insert(0, "100")
        self.p_entry.config(bg="lavender blush", fg="black")
        self.keep_label = tk.Label(self.root, text="Percentage to keep:\n")
        self.keep_label.config(bg="black", fg="seashell2", font=("Arial", 12))
        self.keep_entry = tk.Entry(self.root)
        self.keep_entry.insert(0, "5")
        self.keep_entry.config(bg="lavender blush", fg="black")
        self.mutations_label = tk.Label(self.root, text="Percentage of mutations:\n")
        self.mutations_label.config(bg="black", fg="seashell2", font=("Arial", 12))
        self.mutations_entry = tk.Entry(self.root)
        self.mutations_entry.insert(0, "20")
        self.mutations_entry.config(bg="lavender blush", fg="black")


        # Create a button to start the code
        self.start_button_Regular = tk.Button(self.root, text="Regular", command=self.start_Regular, width=10, height=1)
        self.start_button_Regular.config(bg="lavender blush", fg="black", font=("Arial", 11))
        self.start_button_Darwin = tk.Button(self.root, text="Darwin", command=self.start_Darwin, width=10, height=1)
        self.start_button_Darwin.config(bg="lavender blush", fg="black", font=("Arial", 11))
        self.start_button_Lamarck = tk.Button(self.root, text="Lamarck", command=self.start_Lamarckn, width=10, height=1)
        self.start_button_Lamarck.config(bg="lavender blush", fg="black", font=("Arial", 11))

        # Place the labels and entry boxes on the screen
        self.p_label.place(relx=0.15, rely=0.2550)
        self.p_entry.place(relx=0.7, rely=0.2950, anchor=tk.CENTER)
        self.keep_label.place(relx=0.15, rely=0.3700)
        self.keep_entry.place(relx=0.7, rely=0.4100, anchor=tk.CENTER)
        self.mutations_label.place(relx=0.15, rely=0.4850)
        self.mutations_entry.place(relx=0.7, rely=0.5300, anchor=tk.CENTER)


        # Place the start, title and legend button on the screen
        self.t_label.place(relx=0.5, rely=0.200, anchor=tk.CENTER)
        self.start_button_Regular.place(relx=0.5, rely=0.76, anchor=tk.CENTER)
        self.start_button_Darwin.place(relx=0.25, rely=0.76, anchor=tk.CENTER)
        self.start_button_Lamarck.place(relx=0.75, rely=0.76, anchor=tk.CENTER)


    def mainloop(self):
        self.root.mainloop()



    def start_Regular(self):
        global method_of_running
        method_of_running = "random"
        self.start_code()

    def start_Darwin(self):
        global method_of_running
        method_of_running = "Darwin"
        self.start_code()

    def start_Lamarckn(self):
        global method_of_running
        method_of_running = "Lamarck"
        self.start_code()

    def start_code(self):
        # Retrieve the values from the entry boxes
        p = self.p_entry.get()
        keep = self.keep_entry.get()
        mutations = self.mutations_entry.get()

        # Validate the input values
        error_message = ""

        try:
            p = int(p)
            if not (1 < p): # there must be at least one person
                raise ValueError("P must be more than 1.")
        except ValueError as e:
            error_message += "* Invalid input for Population \n"

        try:
            keep= int(keep)
            if not 0 <= keep <= 100:
                raise ValueError("keep must be between 0 to 100.")
        except ValueError as e:
            error_message += "* Invalid input for % keep \n "

        try:
            mutations= int(mutations)
            if not 0 <= mutations <= 100:
                raise ValueError("mutations must be between 0 to 100.")
        except ValueError as e:
            error_message += "* Invalid input for % mutations \n "


        # If all input values are valid, proceed to the grid screen
        if not error_message:
            self.root.destroy()
            global population_size, percentage_to_keep, percentage_of_mutations, number_of_optimization_steps
            population_size = p
            percentage_to_keep = keep
            percentage_of_mutations = mutations
            # number_of_optimization_steps = optimization

        else:
            # Display error message in GUI
            if not hasattr(self, "error_label"):
                self.error_label = tk.Label(self.root, text=error_message, font=("Arial", 9, "bold"))
                self.error_label.place(relx=0.8, rely=0.7, anchor=tk.CENTER)
            else:
                self.error_label.config(text=error_message)
            return False


def graph_drawing(generation_progress, technique=""):
    """
    This method draws the graph-each line represents the max anf min values for fitness at this generation and also the
    change of the average value during the generations progress
    :param generation_progress: a list of tuples containing the min max and average in each generation
    :param technique: the technique of running - "random", "Darwin", "Lamarck"
    """
    # unpack the minimum, maximum, and average values into separate lists
    maxs, mins, averages = zip(*generation_progress)

    # plot the graph
    bar_width = 0.7  # Width of each bar

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(13, 6))

    for i in range(0, len(generation_progress), steps_of_graph):
        # plot the column graph
        ax.bar(i + 1 - bar_width / 2, maxs[i], color='C0', alpha=0.5, width=bar_width)
        ax.bar(i + 1 + bar_width / 2, mins[i], color='C6', alpha=0.5, width=bar_width)

        # plot the curve connecting the columns
        if i + steps_of_graph - 1 < len(generation_progress) - 1:
            ax.plot([i + 1, i + 1 + steps_of_graph], [averages[i], averages[i + steps_of_graph]], color='C4',
                    linewidth=2)
            ax.scatter([i + 1, i + 1 + steps_of_graph], [averages[i], averages[i + steps_of_graph]], color='C4', s=15)

    # set the x-axis ticks and labels
    ax.set_xticks(np.arange(1, len(generation_progress) + 1, steps_of_graph))
    ax.tick_params(axis='x', labelrotation=90, labelsize=9)
    ax.set_xlabel('Generations', fontsize=12)

    # set the y-axis label and limit
    ax.set_ylabel('Fitness', fontsize=12)
    ax.set_ylim([0, max(maxs) + 25])

    # add a legend
    ax.plot([], [], color='C0', alpha=0.5, label='Max')
    ax.plot([], [], color='C6', alpha=0.5, label='Min')
    ax.plot([], [], color='C4', linewidth=2, label='Average')
    ax.legend(fontsize=10, loc='upper left')

    # set the titles
    title_string = 'Overview of the performance- %s\npopulation size %s, %s%% mutations,' \
                   ' %s%% permutations to keep, fitness activation: %s' % (technique, population_size,
                                                                           percentage_of_mutations, percentage_to_keep,
                                                                           activate_fitness_func_count)
    plt.title(title_string, fontsize=13, y=1.025, fontweight='bold')

    # show the plot
    plt.show()
    # fig.savefig('%s.png' % technique)


def calculation_frequencies(decrypted_text, keys):
    """
    This function calculate the frequencies of each key in the decrypted text
    :param decrypted_text: the decrypted text
    :param keys: the key value to search in the given text
    :return: a dictionary of the frequencies
    """
    frequencies_of_char = {}
    for c in keys:
        frequencies_of_char[c] = decrypted_text.count(c)
    total_letters = sum(frequencies_of_char.values())

    for c in frequencies_of_char:
        frequencies_of_char[c] /= total_letters

    return frequencies_of_char


def calculate_score(original_statistics, frequencies):
    """
    This function sums the difference between the expected frequency and the actual frequency
    :param original_statistics: the original statistics
    :param frequencies: the frequencies
    :return: the sum of the difference
    """
    score = 0
    for c in original_statistics.keys():
        score += abs(frequencies[c] - original_statistics[c]) ** 2
    return 1 - score
    # score = 0
    #
    # for c in original_statistics.keys():
    #     difference = (frequencies.get(c, 0) - original_statistics[c])
    #     if abs(difference) <= 0.001:
    #         score += 0.1
    #
    # return score


def count_common_words_binary(decrypted_text):

    # Convert the decrypted text to a list of lowercase words
    decrypted_words = decrypted_text.lower().split()

    # Sort the words list for binary search
    sorted_words_list = sorted(common_words)

    # Use binary search to check if each word in the decrypted text appears in the words list
    found_words = []
    for word in decrypted_words:
        # Use bisect_left from the bisect module to perform binary search
        index = bisect_left(sorted_words_list, word)
        if index < len(sorted_words_list) and sorted_words_list[index] == word:
            found_words.append(word)

    # closer to 0: indicates that the model's predictions are less accurate (less known words)
    return len(found_words) * 2
    # return math.pow(len(found_words), 2) / len(decrypted_words)
    # return (len(found_words) * 2) / len(decrypted_words)


def count_common_words(decrypted_text):

    # Convert the decrypted text to a set of lowercase words
    set_decrypted_words = set(decrypted_text.lower().split())
    set_common_words = set(common_words)

    # common words between the file and the decrypted text
    appears = set_common_words & set_decrypted_words

    return len(appears) * 2


def write_to_file(file_name, content, flag = None):
    """
    This method writes the output to the wanted files
    :param file_name: the file name
    :param content: the content to write the file
    """
    # with open(file_name, 'w') as f:
    #     if flag:
    #         f.write(content)
    #     else:
    #         for key, value in content.items():
    #             if key not in SPECIAL_CHARS and key != '':
    #                 f.write(f'{key} {value}\n')

    with open(file_name, 'w') as f:
        if file_name == "plain.txt":
                f.write(content)
        else:
            for key, value in content.items():
                if key not in SPECIAL_CHARS and key != '':
                    f.write(f'{key} {value}\n')


def fitness(decrypted_text):
    """
    Calculate the fitness score of a candidate key based on the frequency of letters
    in the decrypted text.
    :param decrypted_text: the decrypted text.
    :return: the fitness score (0 = best).
    """
    # Count for entering the function
    global activate_fitness_func_count
    activate_fitness_func_count += 1

    # number of appearances of each letter
    # frequencies_one_char, frequencies_two_char = check(decrypted_text)
    frequencies_one_char = calculation_frequencies(decrypted_text, letter_freq.keys())
    frequencies_two_char = calculation_frequencies(decrypted_text, letter2_freq.keys())

    # Expected frequencies based on provided files
    score = calculate_score(letter_freq, frequencies_one_char)
    score += calculate_score(letter2_freq, frequencies_two_char)

    # We would like the best solution to have the lowest score
    score += count_common_words(decrypted_text)
    return score


def initialize_population():
    """
    This function create the first generation
    :return: the first generation
    """
    population = []
    for i in range(population_size):
        used_char = set()
        unused_chars = set(set(ALL_CHARS))
        key = {}
        for c in ALL_CHARS:
            new_char = random.choice(list(unused_chars))  # initialize randomly
            key[c] = new_char
            used_char.add(new_char)
            unused_chars.remove(new_char)
        population.append(key)
    return population


def decrypt_text(key):
    """
    This function decrypts the encrypted text using the provided key.
    :param key: the key to decrypt by.
    :return: the decrypt text.
    """
    decrypted_text = ''
    for c in encrypted_text:
        if c in key:
            decrypted_text += key[c]
        else:
            decrypted_text += c
    return decrypted_text


def crossover(parent1, parent2):
    """
    # This function does the crossover between 2 parents
    :param parent1: the first solution
    :param parent2: the second solution
    :return: the new solution - the child
    """
    child = {}
    keys = sorted(parent1.keys())
    crossover_point = random.randint(0, len(keys) - 1)
    for i, k in enumerate(keys):
        if i < crossover_point:
            child[k] = parent1[k]
        else:
            child[k] = parent2[k]

    # handle duplicate values in child dictionary
    unused_chars = set(set(ALL_CHARS) - set(child.values()))
    used_values = set()
    for k, v in child.items():
        if v in used_values:
            new_v = unused_chars.pop()
            child[k] = new_v
            used_values.add(new_v)
        else:
            used_values.add(v)
    return child


def optimize_one_item(individual, technique):
    original_item = individual[0].copy()
    best_individual = original_item
    best_fitness = individual[1]
    for i in range(number_of_optimization_steps):
        c1, c2 = random.sample(list(individual[0].keys()), 2)
        individual[0][c1], individual[0][c2] = individual[0][c2], individual[0][c1]
        current_fitness = fitness(decrypt_text(individual[0]))
        if current_fitness >= best_fitness:
            best_individual = individual[0].copy()
            best_fitness = current_fitness
    if technique == "Darwin":
        return original_item, best_fitness
    else:  # Lamarck
        return best_individual, best_fitness


def LAMARCKIAN_new_population(fitness_score):
    """
    This function base on the lamarck method that say - good features are passed on to children, therefore the
    function has optimizations for the next generation
    :param fitness_score: list of the next generation
    :return: the next generation optimize
    """
    next_generation_after_optimization = []
    for individual in fitness_score:
        optimized_item = optimize_one_item(individual, "Lamarck")

        next_generation_after_optimization.append(optimized_item)

    return next_generation_after_optimization


def DARWINIAN_new_population(fitness_score):
    """
    This function base on the Darwin method that say - the strong survives, therefore the
    function has optimizations for the current generation - the strong have more Chance
    :param fitness_score: list of the current generation and there fitness scores
    :return: the current generation optimize
    """
    next_generation_after_optimization = []
    for individual in fitness_score:
        optimized_item = optimize_one_item(individual, "Darwin")

        next_generation_after_optimization.append(optimized_item)

    return next_generation_after_optimization


def biased_lottery(fitness_score, total_fitness):
    # relative fitness scores:
    relative_fitness_scores = []
    for score in fitness_score:
        relative_score = score[1] / total_fitness
        relative_fitness_scores.append(relative_score)
    biased_vector = []
    for i, score in enumerate(relative_fitness_scores):
        num_times = int(score * total_fitness)
        biased_vector += [i] * num_times
    # Shuffle of random selection:
    random.shuffle(biased_vector)
    return biased_vector


def create_new_population(fitness_score, technique="random"):
    """
    # create the next generation base on the fitness score
    :param fitness_score: the fitness scores of the entire population
    :param technique: the method by which we would like to create the next generation
    :return: the next generation
    """
    if technique == "Darwin":
        fitness_score = DARWINIAN_new_population(fitness_score)  # for darwin we need to choose the best parents

    elif technique == "Lamarck":
        fitness_score = LAMARCKIAN_new_population(fitness_score)

    # save as is in next generation:
    n_duplicates = population_size * percentage_to_keep // 100  # calculate 5% of the list size
    fitness_score.sort(key=lambda x: x[1], reverse=True)
    top_percentage = fitness_score[:n_duplicates]
    new_population = [x[0] for x in top_percentage]

    # crossover
    total_fitness = sum(score[1] for score in fitness_score)
    biased_vector = biased_lottery(fitness_score, total_fitness)
    for i in range(population_size - n_duplicates):

        parent_index = []
        while len(parent_index) < 2:
            index = random.choice(biased_vector)
            if index not in parent_index:
                parent_index.append(index)

        parent1 = fitness_score[parent_index[0]][0]
        parent2 = fitness_score[parent_index[1]][0]

        child = {}
        child = crossover(parent1, parent2)
        new_population.append(child)

    # mutation
    mutation_indices = random.sample(range(population_size), int(population_size * percentage_of_mutations // 100))
    for i in mutation_indices:
        # keys = [key for key in set(encrypted_text) if key not in SPECIAL_CHARS]
        key1, key2 = random.sample(list(ALL_CHARS), 2)  # Choose two keys at random
        # Swap the values of the two keys
        new_population[i][key1], new_population[i][key2] = new_population[i][key2], new_population[i][key1]

    return new_population


def genomic_progression(technique="random"):
    """
    This function start the genomic progression in order to find the solutions
    :return: the best_value = the solution
    """
    global percentage_of_mutations, generation_progress
    # Define the population size and the history of the generation
    generation_progress = []

    # population initialize
    population = initialize_population()

    max_value_so_far = 0 # keep the best value- it should be the value seen the last (before break)
    convergence_count = 0
    continue_hyper_mutations = False
    counter = 0
    convergence_check = 0
    iterations = 0
    while True:
        iterations += 1
        fitness_score = []

        # find to each solution its fitness score
        for solution in population:
            decrypted_text = decrypt_text(solution)
            score = fitness(decrypted_text)
            fitness_score.append((solution, score))

        # calculate and save the best, worst and average fitness scores
        maximum = max(fitness_score, key=lambda x: x[1])
        max_value = maximum[1]  # fitness score of best permutation
        best_value = maximum[0]  # best permutation so far
        worst_value = min(fitness_score, key=lambda x: x[1])[1]
        fitness_scores_list = [score[1] for score in fitness_score]
        average_fitness = sum(fitness_scores_list) / len(fitness_scores_list)
        generation_progress.append((max_value, worst_value, average_fitness))

        if max_value > max_value_so_far:
            max_value_so_far = max_value

        if iterations >= definition_of_plateau and abs(average_fitness - generation_progress[-2][2]) <= gap_of_plateau:
            stop = generation_progress[-plateau_of_maximum:]
            last_10_values = [value[0] for value in stop]
            all_same = all(x == max_value_so_far for x in last_10_values)
            if all_same:
                break
            if convergence_count == 0:
                convergence_check = iterations
                convergence_count += 1
            elif iterations - convergence_check == 1:
                convergence_count += 1
                convergence_check = iterations
            else:
                convergence_count = 0

        # early convergence fixation:
        if convergence_count == definition_of_plateau or continue_hyper_mutations:
            convergence_count = 0
            if continue_hyper_mutations is False:
                continue_hyper_mutations = True
                counter = num_of_generation_to_fix_convergence
                percentage_of_mutations += 30
            elif counter == 0 and continue_hyper_mutations:
                continue_hyper_mutations = False
                percentage_of_mutations -= 30
            else:
                counter -= 1

        # create new population
        population = create_new_population(fitness_score, technique)

    # draw graph
    global steps_of_graph
    steps_of_graph = iterations // 50
    graph_drawing(generation_progress, technique)
    return best_value


def open_files():
    """
    This function open the needed file as global variables
    """
    global encrypted_text, common_words, letter_freq, letter2_freq

    # open the encrypted text
    with open('enc.txt', 'r') as f:
        encrypted_text = f.read()
    encrypted_text = encrypted_text.lower()

    # open the common words in English
    with open('dict.txt', 'r') as f:
        lines = f.readlines()
        common_words = []
        for line in lines:
            common_words.append(line.strip())

    # open the one word frequencies file
    letter_freq = {}
    with open("Letter_Freq.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            freq, letter = line.strip().split("\t")
            letter_freq[letter.lower()] = float(freq)

    # open the two word frequencies file
    letter2_freq = {}
    with open("Letter2_Freq.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if line[0].isdigit():
                freq, letter = line.strip().split("\t")
                letter2_freq[letter.lower()] = float(freq)


if __name__ == "__main__":

    global encrypted_text, common_words, letter_freq, letter2_freq

    start = time.time()

    # open needed files:
    open_files()

# Three methods running of the program: -------------------------------------------------------------------------------
    # the best permutation to understand the text:
    # type = ["Lamarck", "random", "Darwin"]
    # for i in range(3):
    #     activate_fitness_func_count = 0
    #     best_value = genomic_progression(type[i])
    #     # write decrypted output to file
    #     write_to_file("plain_%s.txt" % (type[i]), decrypt_text(best_value), 1)
    #     write_to_file("perm_%s.txt" % (type[i]), best_value, 0)


# normal running of the program: --------------------------------------------------------------------------------------

    gui = VariableInputGUI()
    gui.mainloop()

    best_value = genomic_progression(method_of_running) # "random", "Darwin", "Lamarck"

    # write decrypted output to file
    write_to_file("plain.txt", decrypt_text(best_value))
    write_to_file("perm.txt", best_value)


# stat compression of the program: ------------------------------------------------------------------------------------
    global generation_progress

    # population_average_compare=[]
    # population_max_compare = []
    # population_fitness_compare = []
    # pop_size = [50,100,200,300]
    # for i in range(len(pop_size)):
    #     activate_fitness_func_count = 0
    #     population_size = pop_size[i]
    #     best_value = genomic_progression("Lamarck")
    #     population_max_compare.append([value[0] for value in generation_progress])
    #     population_average_compare.append([value[2] for value in generation_progress])
    #     population_fitness_compare.append(activate_fitness_func_count)
    # # compare_results(population_fitness_compare, population_max_compare, "random", pop_size, "population", "population")
    # compare_results(population_max_compare, population_average_compare, population_fitness_compare, "Lamarck", pop_size, "population", "population")
    # population_size = 100

    # keep_average_compare = []
    # keep_max_compare = []
    # keep_fitness_compare = []
    # keep_amount = [5,10,15,20]
    # for i in range(len(keep_amount)):
    #     activate_fitness_func_count = 0
    #     percentage_to_keep = keep_amount[i]
    #     best_value = genomic_progression("Darwin")
    #     keep_max_compare.append([value[0] for value in generation_progress])
    #     keep_average_compare.append([value[2] for value in generation_progress])
    #     keep_fitness_compare.append(activate_fitness_func_count)
    # compare_results(keep_max_compare, keep_average_compare, keep_fitness_compare, "Darwin", keep_amount, "percentage of items to keep", "keep")
    # percentage_to_keep = 5
    #
    # mutations_average_compare = []
    # mutations_max_compare = []
    # mutations_fitness_compare = []
    # mutations_amount = [10,20,40,60]
    # for i in range(len(mutations_amount)):
    #     activate_fitness_func_count = 0
    #     percentage_of_mutations = mutations_amount[i]
    #     best_value = genomic_progression("Darwin")
    #     mutations_max_compare.append([value[0] for value in generation_progress])
    #     mutations_average_compare.append([value[2] for value in generation_progress])
    #     mutations_fitness_compare.append(activate_fitness_func_count)
    # compare_results(mutations_max_compare, mutations_average_compare, mutations_fitness_compare, "Darwin", mutations_amount, "mutations amount", "mutations")
    # percentage_of_mutations = 25

    # convergence_average_compare = []
    # convergence_max_compare = []
    # convergence_fitness_compare = []
    # convergence_amount = [5, 10, 15, 20]
    # for i in range(len(convergence_amount)):
    #     activate_fitness_func_count = 0
    #     num_of_generation_to_fix_convergence = convergence_amount[i]
    #     best_value = genomic_progression("random")
    #     convergence_max_compare.append([value[0] for value in generation_progress])
    #     convergence_average_compare.append([value[2] for value in generation_progress])
    #     convergence_fitness_compare.append(activate_fitness_func_count)
    # compare_results(convergence_max_compare, convergence_average_compare, convergence_fitness_compare, "random",
    #                 convergence_amount, "convergence fix round", "convergence")
    # num_of_generation_to_fix_convergence = 10

    end = time.time()
    print(end - start)
