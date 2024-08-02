# Operating System Interfaces
## Process and Memory
OS can
- manage and abstract the low=level hardware
- share the hardware among multiple programs
- control ways for programs to interact

system calls
- `fork()`
- `wait(int* status)`
- `exec(char* file, char* argv[])`

!!! Key: Why `fork` and `exec` are not combined in a single call?
## I/O and File Descripters
A **file descriptor** is a small _integer_ representing a kernel-managed object that a process may read from or write to
- `read(fd,buf,n)`: read at most n bytes from fd to buf
- `write(fd,buf,n)`: write n bytes from buf to fd
- `close(fd)`
- `dup(fd)`: return a new fd refers to the same object
  - `ls 2>&1`: give the command a fd 2 that is a duplicate of fd 1
```C
// Redirection
char *argv[2];
argv[0] = "cat";
argv[1] = 0;
if(fork() == 0) {
    close(0);
    open("input.txt", O_RDONLY);
    exec("cat", argv);
}

```
!!! Answer: By Seperating `fork` and `exec`,redirection is easy to implement.

## Pipes
A **pipe** is a small kernel _buffer_ exposed to processes as a pair of file descriptors, one for reading and one for writing.
```C
// An implementation of pipe
// refer to user/sh.c 101
int p[2];
char *argv[2];
argv[0] = "wc";
argv[1] = 0;
pipe(p);
if(fork() == 0) {
    close(0);
    dup(p[0]);
    close(p[0]);
    close(p[1]); // ! a pitfall resolved here
    exec("/bin/wc", argv);
} else {
    close(p[0]);
    write(p[1], "hello world\n", 12);
    close(p[1]);
}

```
!!! Note: Pipe Versus Temp files
- _pipes_ automatically clean themselves up
  - with the _file redirection_, a shell would have to be careful to remove /tmp/xyz when done. 
- _pipes_ can pass arbitrarily long streams of data
  - _file redirection_ requires enough free space on disk to store all the data.
- _pipes_ allow for parallel execution of pipeline stages
  - _the file approach_ requires the first program to finish before the second starts.

## File System
The same underlying file, called an **inode**, can have multiple names, called **links**. 
- Each link consists of an **entry** in a directory
  - the entry contains a **file name** and a **reference to an inode**. 
  - An inode holds metadata about a file
    - type (file or directory or device)
    - length
    - the location of the file’s content on disk
    - the number of links to a file
  - Each inode is identified by a unique **inode number**
- `fstat`
- `link(name_a,name_b)`: create a new name `b` refer to the same inode as a refers to.
- `unlink(name)`
- `chdir(path)`: change directory in the **child**
- `cd`: must be built-in to change the current working directory
```C
// `fstat(fd,&state)` retrieves info from the inode it refers to into state
// kernel/stat.h
struct stat {
  int dev; // File system’s disk device
  uint ino; // Inode number
  short type; // Type of file
  short nlink; // Number of links to file
  uint64 size; // Size of file in bytes
};

```

```C
// Belows are the same
chdir("/a");
chdir("b");
open("c", O_RDONLY);

open("/a/b/c", O_RDONLY);
```
- `mkdir(path)`: create a directory
- `open(fd,O_CREATE)`: create a new data file
- `mknod(device,major,minor)`: create a device file
## Real World
POSIX:Portable OS Interface