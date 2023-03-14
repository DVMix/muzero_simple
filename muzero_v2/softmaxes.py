import numpy


def softmax_sample(distribution, temperature: float):
    # Stubs to make the typechecker happy.
    if temperature == 0:
        temperature = 1
    # distribution = numpy.array(distribution) ** (1 / temperature)
    distribution = numpy.array(distribution)[:, 0] ** (1 / temperature)
    p_sum = distribution.sum()
    sample_temp = distribution / p_sum
    return 0, numpy.argmax(numpy.random.multinomial(1, sample_temp, 1))


def visit_softmax_temperature(num_moves, training_steps):
    if num_moves < 30:
        return 1.0
    else:
        return 0.0  # Play according to the max.
