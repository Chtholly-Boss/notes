# Makefile
refer to [How-To-Write-Makefile](https://seisman.github.io/how-to-write-makefile/overview.html) for detailed guidance
## Contents
* [core-contents](#core)
* [rules](#rule)
* [commands](#command)
* [variable](#variable)
* [condition](#condition)

## Core Concept {#core}
```bash
# comment using #, only support single-line comment

# when one of the prerequisites is newer than target, recipe will be executed
target ... : prerequisites ...; command
# a tab must prefix a command
<tab>recipe
    ...
    ...
```
## Rules {#rule}
```bash
# using \ to seperate line
foo.o: foo.c \
        def.h

# support ~, *, ? wildcards
# define VPATH for makefile
VPATH = src:../headers
# when makefile couldn't find file in environment varibles
# it will find the paths in VPATH
# also, you can use vpath
vpath %.h ../headers
# find *.h in ../headers, note the % prefix .h

.PHONY clean
clean:
    rm -rf *.c
# clean is a pseudo target
# pseudo target shouldn't be the same name as a file
# to force a name to be pseudo target, use .PHONY
# target can also be a prerequisite
cleanobj:
    rm -rf *.o
cleansrc: cleanobj
    rm -rf *.c
```

# Commands {#command}
```bash
# to cancel displaying a command, prefix it a @
@echo "compiling..."

# just print, not execute 
make -n
make --just-print
# This will be helpful to learn about the execution order

# when you want to apply a command to another, you should use ;
# This will print the makefile directory
cd /home
pwd
# This will print home
cd /home; pwd

# to omit an error in execution, prefix a -
clean:
    -rm -f *.o
# when an error occur, it will continue
```

## Variables {#variable}
```bash
# define and use variables
objects = program.o foo.o utils.o
program : $(objects)
    cc -o program $(objects)

$(objects) : defs.h
# using := to avoid recursively define
x := foo
y := $(x) bar
x := later
# y will use the predefined value of x
# i.e. y = foo bar

# define a variable when not defined
FOO ?= bar
# add value to a variable 
FOO = foo
FOO += bar
# FOO = foo bar
```

## Condition {#condition}
```bash
# ifeq(<arg1,arg2>)
# ifneq
# ifdef
# ifndef
<conditional-directive>
<text-if-true>
else
<text-if-false>
endif

# an example
libs_for_gcc = -lgnu
normal_libs =

ifeq ($(CC),gcc)
    libs=$(libs_for_gcc)
else
    libs=$(normal_libs)
endif

foo: $(objects)
    $(CC) -o foo $(objects) $(libs)
```

## Auto Variables
See `自动化变量` in [learn-makefile](https://seisman.github.io/how-to-write-makefile/implicit_rules.html)