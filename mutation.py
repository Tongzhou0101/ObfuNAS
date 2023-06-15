from setting import *
import copy
import numpy as np
import random

def mutate_spec(victim_spec, old_spec, mutation_rate=1.0):
    """Computes a valid mutated spec from the old_spec."""
    while True:
        new_matrix = copy.deepcopy(old_spec.original_matrix)
        new_ops = copy.deepcopy(old_spec.original_ops)
        NUM_VERTICES = new_matrix.shape[0]
        idle_VERTICES = MAX_VERTICES - NUM_VERTICES
        seq_list = np.arange(NUM_VERTICES).tolist()
        para_list = np.arange(NUM_VERTICES - 1).tolist()  # avoid output

        # In expectation, one op is resampled.
        # Kernel widening
        op_mutation_prob = mutation_rate / OP_SPOTS
        for ind in range(1, NUM_VERTICES - 1):
            if new_ops[ind] == CONV1X1:
                if random.random() < op_mutation_prob:
                    new_ops[ind] = CONV3X3

        # Add operations if the num of current ops is not up to 7 and the num of edges is not up to 9
        # Layer deepening
        if new_matrix.sum() < MAX_EDGES and idle_VERTICES > 0:
            # random select 2 ops and add another op between them

            [x, y] = sorted(random.sample(seq_list, 2))
            if new_matrix[x, y] == 1 and [x, y] != [0, seq_list[-1]]:
                op_add_prob = mutation_rate / 5  # 3 options: parallel adding, sequential adding, and no change
                # add one sequential op
                if random.random() < op_add_prob:
                    # select one op:
                    add_op = random.choice(sequential_op)
                    new_ops.insert(y, add_op)
                    idle_VERTICES -= 1
                    # remove original connection
                    new_matrix[x, y] = 0
                    # expand the matrix size
                    new_matrix = np.insert(np.insert(new_matrix, y, 0, axis=1), y, 0, axis=0)
                    # add new connections between the new op and original ops
                    # len(new_matrix)-2 is the index of inserted column
                    add_conn_idx1 = tuple([x, y])
                    new_matrix[add_conn_idx1] = 1

                    add_conn_idx2 = tuple([y, y + 1])

                    new_matrix[add_conn_idx2] = 1
        # Layer branch adding
        if new_matrix.sum() < MAX_EDGES and idle_VERTICES > 0:
            # random select 2 ops and add parallel op

            [x, y] = sorted(random.sample(para_list, 2))
            op_add_prob = mutation_rate / 5  # 2 options: parallel adding and no change
            # add one parallel op:
            # The difference b
            if random.random() < op_add_prob:
                # select one op:
                add_op = random.choice(parallel_op)
                new_ops.insert(y, add_op)
                idle_VERTICES -= 1
                # expand the matrix size
                new_matrix = np.insert(np.insert(new_matrix, y, 0, axis=1), y, 0, axis=0)

                add_conn_idx1 = tuple([x, y])
                new_matrix[add_conn_idx1] = 1

                add_conn_idx2 = tuple([y, y + 1])

                new_matrix[add_conn_idx2] = 1

        # Shortcut adding
        # In expectation, V edges flipped (note that most end up being pruned).
        if new_matrix.sum() < MAX_EDGES:
            NUM_VERTICES = new_matrix.shape[0]
            edge_mutation_prob = mutation_rate / 4
            for src in range(0, NUM_VERTICES - 2):
                for dst in range(src + 1, NUM_VERTICES - 1):
                    if random.random() < edge_mutation_prob and new_matrix[src, dst] == 0:
                        new_matrix[src, dst] = 1
                        # residual connection
            if random.random() < edge_mutation_prob and new_matrix[0, -1] == 0:
                new_matrix[0, -1] = 1

        new_spec = api.ModelSpec(new_matrix, new_ops)
        if nasbench.is_valid(new_spec):
            if len(new_spec.original_ops) != len(old_spec.original_ops):
                return new_spec
            elif np.any(
                    new_spec.original_matrix != old_spec.original_matrix) or new_spec.original_ops != old_spec.original_ops:
                return new_spec
            # Avoid getting stuck in an infinite loop
            elif new_matrix.sum() == MAX_EDGES or CONV1X1 not in new_spec.original_ops:
                old_spec = victim_spec


def random_combination(iterable, sample_size):
    """Random selection from itertools.combinations(iterable, r)."""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)
