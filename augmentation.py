from numpy.random import default_rng


rng = default_rng(42)


def random_ex_swap(text, m):
    ex_list = text.split('+')
    if 0.0 <= m < 0.333:
        pos1, pos2 = rng.integers(low=0, high=len(ex_list), size=2)
        ex_list[pos1], ex_list[pos2] = ex_list[pos2], ex_list[pos1]

    elif 0.333 <= m < 0.666:
        pos1, pos2 = rng.integers(low=0, high=len(ex_list), size=2)
        ex_list[pos1], ex_list[pos2] = ex_list[pos2], ex_list[pos1]

        pos1, pos2 = rng.integers(low=0, high=len(ex_list), size=2)
        ex_list[pos1], ex_list[pos2] = ex_list[pos2], ex_list[pos1]

    else:
        pos1, pos2 = rng.integers(low=0, high=len(ex_list), size=2)
        ex_list[pos1], ex_list[pos2] = ex_list[pos2], ex_list[pos1]

        pos1, pos2 = rng.integers(low=0, high=len(ex_list), size=2)
        ex_list[pos1], ex_list[pos2] = ex_list[pos2], ex_list[pos1]

        pos1, pos2 = rng.integers(low=0, high=len(ex_list), size=2)
        ex_list[pos1], ex_list[pos2] = ex_list[pos2], ex_list[pos1]

    new_prompt = '+'.join(ex_list)
    return new_prompt


def add_ex(text, training_ds, m):
    rint = rng.integers(low=0, high=len(training_ds))

    new_ex = training_ds[rint]

    # TODO: Generalize this to 5+ examples
    new_prompt = text + '+' + new_ex
    return new_prompt