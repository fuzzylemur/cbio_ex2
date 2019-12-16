import argparse
import numpy as np
from scipy.misc import logsumexp
from numpy import logaddexp

LETTER_TO_INDEX = {'A':0, 'C':1, 'G':2, 'T':3, '^':4, '$':5}

def build_transition_matrix(num_states, motif_len, p, q):
    transition_matrix = np.zeros((num_states, num_states))
    transition_matrix[0, 1] = q
    transition_matrix[0, -2] = 1-q
    transition_matrix[1, 1] = 1-p
    transition_matrix[1, 2] = p
    transition_matrix[-2, -2] = 1-p
    transition_matrix[-2, -1] = p
    for i in range(motif_len):
        transition_matrix[2+i, 3+i] = 1
    #print("### transition matrix ###\n", np.log(transition_matrix), "\n")
    return np.log(transition_matrix)

def build_emission_matrix(initial_emissions, motif_len):
    zero_row = np.zeros((1, 4))
    zero_col = np.zeros((motif_len+4, 1))
    bg_emissions = np.array([0.25, 0.25, 0.25, 0.25])
    emission_matrix = np.vstack((zero_row, bg_emissions, initial_emissions, bg_emissions, zero_row))
    emission_matrix = np.hstack((emission_matrix, zero_col, zero_col))
    emission_matrix[0, -2] = 1
    emission_matrix[-1, -1] = 1
    #print("### emission matrix ###\n", np.log(emission_matrix), "\n")
    return np.log(emission_matrix)

def calculate_viterbi(num_states, seq, transition_matrix, emission_matrix):
    viterbi_matrix = np.full((num_states, len(seq)), fill_value=np.NINF, dtype=float)
    trace_matrix = np.zeros((num_states, len(seq)), dtype=int)
    viterbi_matrix[0, 0] = 0
    for j in range(1, len(seq)):
        for i in range(num_states):
            vec = viterbi_matrix[:,j-1] + transition_matrix[:,i]
            viterbi_matrix[i, j] = max(vec) + emission_matrix[i, LETTER_TO_INDEX[seq[j]]]
            trace_matrix[i, j] = np.argmax(vec)
    #print("### viterbi matrix ###\n", viterbi_matrix[:, :], "\n")
    return viterbi_matrix, trace_matrix

def calculate_forward(num_states, seq, transition_matrix, emission_matrix):
    forward_matrix = np.full((num_states, len(seq)), fill_value=np.NINF, dtype=float)
    forward_matrix[0, 0] = 0
    for j in range(1, len(seq)):
        for i in range(num_states):
            vec = forward_matrix[:, j-1] + transition_matrix[: ,i]
            forward_matrix[i, j] = logsumexp(vec) + emission_matrix[i, LETTER_TO_INDEX[seq[j]]]
            #vec = forward_matrix[:, j-1] + transition_matrix[: ,i]
            #forward_matrix[i, j] = np.sum(vec) + emission_matrix[i, LETTER_TO_INDEX[seq[j]]]
    result = logsumexp(forward_matrix[:, -2])
    return forward_matrix, result

def calculate_backward(num_states, seq, transition_matrix, emission_matrix):
    backward_matrix = np.full((num_states, len(seq)), fill_value=np.NINF, dtype=float)
    backward_matrix[:, -1] = 0
    for j in reversed(range(1, len(seq))):
        for i in range(num_states):
            vec = backward_matrix[:, j] + transition_matrix[i,:] + emission_matrix[:, LETTER_TO_INDEX[seq[j]]]
            backward_matrix[i, j-1] = logsumexp(vec)
    backward_matrix[0, 0] = 0
    result = logsumexp(backward_matrix[:, 1])
    return backward_matrix, result

def print_50(str1, str2):
    i,j = 0,0
    while i < len(str1)-1:
        j = min(i+50, len(str1))
        print(str1[i:j])
        print(str2[i:j], '\n')
        i = j

def main():
    np.set_printoptions(precision=2)
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', help='Algorithm (e.g. viterbi)', required=True)
    parser.add_argument('seq', help='A sequence over the alphabet [A,C,G,T] (e.g. ACTGGACTACGTCATGCA)')
    parser.add_argument('initial_emission', help='Path to emission table (e.g. initial_emission.tsv)')
    parser.add_argument('p', help='transition probability p (e.g. 0.01)', type=float)
    parser.add_argument('q', help='transition probability q (e.g. 0.5)', type=float)
    args = parser.parse_args()

    # handle input parameters
    initial_emissions = np.genfromtxt(args.initial_emission, delimiter='\t')[1:,]
    seq = '^' + args.seq + '$'
    seq_len = len(seq)
    motif_len = len(initial_emissions)
    motif_states = ["s"+str(i+1) for i in range(motif_len)]
    states = ["s", "b1"] + motif_states + ["b2", "e"]
    num_states = len(states)
    p = args.p
    q = args.q

    transition_matrix = build_transition_matrix(num_states, motif_len, p, q)
    emission_matrix = build_emission_matrix(initial_emissions, motif_len)

    if args.alg == 'viterbi':
        viterbi_matrix, trace_matrix = calculate_viterbi(num_states, seq, transition_matrix, emission_matrix)

        # traceback state sequence
        index = np.argmax(viterbi_matrix[:, -1])
        trace = str(int(trace_matrix[index, -1]))
        for j in range(seq_len-2):
            index = int(trace[-1])
            trace += str(int(trace_matrix[index, -j-2]))
        trace = trace[::-1]
        result = ''.join(['M' if int(trace[j]) > 1 and int(trace[j]) < (num_states-2) else 'B' for j in range(1, len(trace))])
        #print("trace: ", trace)
        #print("result: ", result)
        print_50(result, seq[1:-1])
        return result

    elif args.alg == 'forward':
        forward_matrix, result = calculate_forward(num_states, seq, transition_matrix, emission_matrix)
        #print("### forward matrix ###\n", forward_matrix, "\n")
        print(result)
        return result

    elif args.alg == 'backward':
        backward_matrix, result = calculate_backward(num_states, seq, transition_matrix, emission_matrix)
        #print("### backward matrix ###\n", backward_matrix, "\n")
        print(result)
        return result
    
    elif args.alg == 'posterior':
        forward_matrix, result1 = calculate_forward(num_states, seq, transition_matrix, emission_matrix)
        backward_matrix, result2 = calculate_backward(num_states, seq, transition_matrix, emission_matrix)
        posterior_matrix = forward_matrix * backward_matrix
        #print("### forward matrix ###\n", forward_matrix, "\n")
        #print("### backward matrix ###\n", backward_matrix, "\n")
        #print("### posterior matrix ###\n", posterior_matrix, "\n")

        # traceback state sequence
        trace = str(int(np.argmax(posterior_matrix[:, -2])))
        for j in range(1, seq_len-1):
            trace += str(int(np.argmax(posterior_matrix[:, -j-2])))
        trace = trace[::-1]
        result = ''.join(['M' if int(trace[j]) > 1 and int(trace[j]) < (num_states-2) else 'B' for j in range(1, len(trace))])
        #print("trace: ", trace)
        #print("result: ", result)
        print_50(result, seq[1:-1])
        return result

if __name__ == '__main__':
    main()
