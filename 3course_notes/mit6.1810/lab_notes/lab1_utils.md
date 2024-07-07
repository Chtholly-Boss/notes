# Lab: Xv6 and Unix utilities
Lab site: [lab:util](https://pdos.csail.mit.edu/6.S081/2022/labs/util.html)
## Setup
See this [6.S081 Tools](https://pdos.csail.mit.edu/6.S081/2022/tools.html) to install the dependencies.
```bash
# After setup the environments, Do the followings
# Clone the git repo
git clone git://g.csail.mit.edu/xv6-labs-2022
# Checkout to the corresponding branch
git checkout util
# build and run xv6
make qemu
# xv6 has no ps command 
# but if you type Ctrl-p,t
# the kernel will print information about each process.
# To quit qemu type: Ctrl-a x
```
## Tasks
```bash
# to Grade all your program
make grade
# to Grade a specific program
./grade-lab-util sleep
# or:
make GRADEFLAGS=sleep grade
```
### sleep
### pingpong
### primes
### find
### xargs

## Conclusion
| sleep | pingpong | primes | find | xargs |
| --- | --- | --- | --- | --- |
| - | - | - | - | - |