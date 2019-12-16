from motif_find import *
import os

# def main_test():
#     np.set_printoptions(precision=2)
#     # parse arguments
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--alg', help='Algorithm (e.g. viterbi)', required=True)
#     # parser.add_argument('seq', help='A sequence over the alphabet [A,C,G,T] (e.g. ACTGGACTACGTCATGCA)')
#     parser.add_argument('initial_emission', help='Path to emission table (e.g. initial_emission.tsv)')
#     parser.add_argument('p', help='transition probability p (e.g. 0.01)', type=float)
#     parser.add_argument('q', help='transition probability q (e.g. 0.5)', type=float)
#     args = parser.parse_args()

#     # handle input parameters
#     initial_emissions = np.genfromtxt(args.initial_emission, delimiter='\t')[1:,]
#     # seq = '^' + args.seq + '$'
#     # seq_len = len(seq)
#     motif_len = len(initial_emissions)
#     motif_states = ["s"+str(i+1) for i in range(motif_len)]
#     states = ["s", "b1"] + motif_states + ["b2", "e"]
#     num_states = len(states)
#     p = args.p
#     q = args.q

#     transition_matrix = build_transition_matrix(num_states, motif_len, p, q)
#     emission_matrix = build_emission_matrix(initial_emissions, motif_len)

#     print(transition_matrix)
#     print(emission_matrix)

if __name__ == '__main__':
    algorithm = 'viterbi'
    seq = 'AAAAAAAAAAAAAAAAAAA'
    emission = 'initial_emision.tsv'
    p = '0.01'
    q = '1'
    command = 'python3 motif_find.py --alg {} {} {} {} {}'.format(algorithm, seq, emission, p, q)
    print(command)
    os.system(command)
    # main_test()
