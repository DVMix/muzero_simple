import numpy


# Stubs to make the typechecker happy.
def softmax_sample(distribution, temperature: float):
    if temperature == 0:
        temperature = 1
    distribution = numpy.array(distribution) ** (1 / temperature)
    p_sum = distribution.sum()
    sample_temp = distribution / p_sum
    # return 0, numpy.argmax(numpy.random.multinomial(1, sample_temp, 1))
    return 0, numpy.argmax(numpy.random.multinomial(1, sample_temp[:, 0], 1))


def launch_job(f, *args):
    f(*args)
