# Lecture 1
Most things can be found in xv6 Book Chapter 1.

Here are some supplyments

* What is the purpose of an O/S?
  * Abstract the hardware for convenience and portability
  * Multiplex the hardware among many applications
  * Isolate applications in order to contain bugs
  * Allow sharing among cooperating applications
  * Control sharing for security
  * Don't get in the way of high performance
  * Support a wide range of applications
* What services does an O/S kernel typically provide?
  * process (a running program)
  * memory allocation
  * file contents
  * file names, directories
  * access control (security)
  * many others: users, IPC, network, time, terminals
* It's worth asking "why" about design decisions:
  * Why these I/O and process abstractions? Why not something else?
  * Why provide a file system? Why not let programs use the disk their own way?
  * Why FDs? Why not pass a filename to write()?
  * Why are files streams of bytes, not disk blocks or formatted records?
  * Why not combine fork() and exec()?
  * The UNIX design works well, but we will see other designs!
* System Calls may be used in lab1
  * fork
  * exec
  * open
