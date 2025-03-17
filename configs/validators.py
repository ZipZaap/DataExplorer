import functools
import inspect

RULES = {
    'min_lat': (
        lambda x: -90 < x < 90,
        'Value out of range! Latitude must be between -90 and 90 degrees'
    ),
    'min_samples': (
        lambda x: x >= 2,
        'Value out of range! At least 2 samples required to form a cluster'
    ),
    'min_years': (
        lambda x: x >= 2,
        'Value out of range! At least 2 unique mars years required to retain a cluster'
    ),
    'algorithm': (
        lambda x: x in ['hdbscan', 'dbscan'],
        'Invalid algorithm! Choose either hdbscan or dbscan'
    ),
    'scale': (
        lambda x: x in (0.25, 0.5, 1),
        'Invalid scale! Choose either 0.25, 0.5 or 1'
    )
}


def run_tests(params):
    """
    Runs validation tests on given parameters.

    Args:
        params (dict): A dictionary mapping parameter names to their values.

    Raises:
        ValueError: If a parameter fails its validation test.
    """
    for name, (test, msg) in RULES.items():
        if name in params and not test(params[name]):
            raise ValueError(msg)


def validate_and_log(X):
    """
    A decorator or a standalone validation function. If X is callable, 
    returns a wrapped function that validates its arguments using the 
    defined rules before executing. Otherwise, assumes X is a dict and 
    directly runs validation.

    Args:
        X (callable or dict): Function to be decorated or dictionary
                              of parameters to validate.

    Returns:
        The wrapped function if X is callable, otherwise None.

    Raises:
        ValueError: If any validation test fails.
        Exception: If any decorated method is applied twice
    """
    if callable(X):
        @functools.wraps(X)
        def wrapper(self, *args, **kwargs):
            sig = inspect.signature(X)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.arguments.pop('self')

            if bound_args.arguments:
                run_tests(bound_args.arguments)

            result = X(self, *args, **kwargs)

            bound_args.apply_defaults()
            if bound_args.arguments['commit'] == True:
                bound_args.arguments.pop('commit')
                self.local_conf[X.__name__] = bound_args.arguments
                
            return result
        return wrapper
    else:
        run_tests(X)