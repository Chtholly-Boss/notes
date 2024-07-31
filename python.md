# Python Tour
## Basics
Most of the basics can be quickly referenced from [The Python Tutorial](https://docs.python.org/3/tutorial/).So here will only cover something new for myself.
```python
# Print raw String 
print(r"C\:home\...")
# match statement
def http_error(status):
    match status:
        case 404:
            return "Not found"
        case 401 | 403 | 404:
            return "Not allowed"
        case _:
            return "Something's wrong with the internet"
# List Comprehension
squares = list(map(lambda x: x**2, range(10)))
# or the following way:
squares = [x**2 for x in range(10)]
```
## Packages
Use pip to install/uninstall packages.

Refer to [Managing Packages with pip](https://docs.python.org/3/tutorial/venv.html)
```bash
# install a package
python -m pip install [package name]
# list all the packages
python -m pip list
```
## Resources
Here are useful links for `python` development
- [Python Tutorial](https://docs.python.org/3/tutorial/)
- [PyPI](https://pypi.org/)