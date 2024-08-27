# Scripts Using Python
## Options
You can deal with options using **argparse** lib.
Refer to [argparse-doc](https://docs.python.org/zh-cn/3.12/library/argparse.html#module-argparse) for more details
```python
import argparse

def main():
    # create the parser
    parser = argparse.ArgumentParser(description='Example of a script')

    # add arguments
    parser.add_argument('-n', '--name', type=str, help='your name', required=True, dest='user_name')
    parser.add_argument('-a', '--age', type=int, help='your age', required=True, dest='user_age')

    # parse
    args = parser.parse_args()

    print(f'Hello, {args.user_name}! you are {args.user_age} years old.')

if __name__ == '__main__':
    main()
```
