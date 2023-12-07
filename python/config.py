import sys
from ast import literal_eval
import os

# Get the directory of the original script
script_dir = os.path.dirname(os.path.realpath(__file__))

for arg in sys.argv[1:]:
    if '=' not in arg:
        # Assume it's the name of a config file
        assert not arg.startswith('--')
        # Adjust path to look in the parent directory of script_dir
        config_file = os.path.join(script_dir, '../' + arg)
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read(), globals())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
