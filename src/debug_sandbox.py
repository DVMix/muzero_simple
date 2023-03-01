import ray

ray.init(local_mode=True)
# i = ray.init(local_mode=True)
# ip, port = i.get('address').split(':')
# pydevd_pycharm.settrace(ip, port=22000, stdoutToServer=True, stderrToServer=True)


@ray.remote
def fact(n):
    if n == 1:
        return n
    else:
        n_ref = fact.remote(n - 1)
        return n * ray.get(n_ref)


@ray.remote
def compute():
    # breakpoint()
    result_ref = fact.remote(5)
    result = ray.get(result_ref)


ray.get(compute.remote())
