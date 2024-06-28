# Virtualization
## Process
### Notes
- a process is simply a running program
  - memory
  - registers
  - I/O information
  - `mechanism`: providing the answer to a `how` question about a system
    - for example, how does an operating system perform a contextswitch?
  - `policy`: provides the answer to a `which` question
    - for example,which process should the operating system run right now
- what must be included in any interface of an operating system
  - Create
  - Destroy
  - Wait
  - Miscellaneous Control
  - Status
- Process Creation
  - Load Code and Static Data 
  - Allocate memory for run-time stack
  - Allocate memory for heap
  - Other works like I/O Setup
  - start the program running at the entry point
- Process State
  - Running
  - Ready
  - Blocked ![state-transition](./figure/ostep/process_trans.png)
- Data Structures
  - process list
  - register context
  - 
![terms](./figure/ostep/process_terms.png)
## Process API
### Notes
- `fork()` is used to create a new process
  - the process that is created is an (almost) exact copy of the calling process
  - the new process just comes into life as if it had called fork() itself. 
  - the value it returns to the caller of `fork()` is different.
    - the parent receives the PID of the newly-created child
    - the child receives a return code of zero
- `wait()`: parent waits for a child process to finish what it has been doing
- `exec()`: run a program that is different from the calling program
  - it does not create a new process; rather, it transforms the currently running program into a different running program.
  - a successful call to exec() never returns.
- `kill()`: send signals
- `signal()`: catch signals
- user tools
  - ps: see which processes are running
  - top: displays the processes of the system and how much CPU and other resources they are eating up
  - kill: send signals
- relevant reads
  - APUE:chapters on Process Control, Process Relationships, and Signals
  - “A fork() in the road” by Andrew Baumann, Jonathan Appavoo, Orran Krieger, Timothy Roscoe. HotOS ’19, Bertinoro, Italy
![api-terms](./figure/ostep/process_api_terms.png)
## Direct Execution
- Challenges
  - Performance
  - Control
- Limited Direct Execution
  - run the program directly on the CPU
  - Problem:
  - how can the OS make sure the program `doesn’t do anything that we don’t want it to do`, while still running it efficiently?
  - when we are running a process, how does the operating system s`top it
from running and switch to another process`, thus `implementing the time
sharing` we require to virtualize the CPU
- Restricted Operations
  - user mode and kernel mode
  - using system call
    - the parts of the C library that make system calls are hand-coded in assembly
    - To execute a system call, a program must execute a special `trap` instruction
    - When finished, the OS calls a special `return-from-trap` instruction
    - trap table: locations of `trap handlers`
    - a system-call number is usually assigned to each system call
- LDE protocol
  - In the first (at boot time), the kernel initializes the trap table, and the CPU remembers its location for subsequent use(via privileged instruction)
  - In the second (when running a process), the kernel sets up a few things (e.g., allocating a node on the process list, allocating memory) before using a return-from-trap instruction to start the execution of the process; this switches the CPU to user mode and begins running the process.
- Switch Between Processes
  - A Cooperative Approach: Wait For System Calls
    - yield
    - trap
    - infinite loop? --- `reboot the machine`
  - A Non-Cooperative Approach: The OS Takes Control
    - a timer interrupt
  - Saving and Restoring Context
    -  whether to continue running the currently-running process, or switch to a different one.  --- made by scheduler
    -  when trapped
       -  current regs -> kernel stack of A
       -  switch ? 
       -  if switch, move kernel regs to process structure of A(in memory)
       -  jump to kernel stack of B
       -  restore regs of B
-  interrupt concurrency
   -  disable interrupts during interrupt processing
   -  locking 
-  relevant reads
   -  “Why Aren’t Operating Systems Getting Faster as Fast as Hardware?” by J. Ousterhout. USENIX Summer Conference, June 1990

![mechanism_terms](./figure/ostep/cpu_mechanisms_terms.png)