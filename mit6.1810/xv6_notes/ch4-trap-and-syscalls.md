# ch4: Traps and system calls
Events which cause the CPU to set aside ordinary execution of instructions and force a transfer of control to special code that handles the event:
- System call
- Exception
- Interrupt

This book uses `trap` as a generic term for these situations

The usual sequence: 
* a trap forces a transfer of control into the kernel; the kernel saves registers and other state so that execution can be resumed
* the kernel executes appropriate handler code
* the kernel restores the saved state and returns from the trap,and the original code resumes where it left off

Xv6 trap handling proceeds in four stages: 
* hardware actions taken by the RISC-V CPU
* some assembly instructions that prepare the way for kernel C code
* a C function that decides what to do with the trap
* the system call or device-driver service routine

Kernel code (assembler or C) that processes a trap is often called a `handler`
## RISC-V Trap Machinary
- kernel writes to **control registers** to tell CPU how to handle traps and can read them to find out a trap has occured.
- refer to `kernel/riscv.h` to get more details about control registers
  - `stvec`: kernel writes the trap handler address here
  - `sepc`: when trap occurs, kernel saves the program counter here.
  - `scause`: RISC-V puts a number here to describe the reason of the trap
  - `sscratch`: help to avoid overwriting registers before saving them
  - `sstatus`
    - `SIE`: control whether device interrupts are enabled
    - `SPP`: whether a trap came from user mode or supervisor mode

To force a trap:
* if the trap is device interrupt, check `SIE` of `sstatus`, if is clear, do nothing
* clear `SIE` to disable device interrupt
* copy pc to `sepc`
* save the current mode in `SPP` of `sstatus`
* set `scause`
* set the mode to supervisor
* copy `stvec` to pc
* start executing at new pc

Note that Xv6 doesn't switch to kernel page-table or kernel stack.

## Traps from user space
Syscall/Illegal things/device interrupt to cause a trap. 
The high level path:
* uservec: `kernel/trampoline.S:21`
* usertrap: `kernel/trap.c:37`
* usertrapret: `kernel/trap.c:90` 
* userret: `kernel/trampoline.S:101`

follows are more detailed:
* uservec starts
  * save the 32 regs to the memory
    * `csrw` to save `a0` to `sscratch`
    * load trapframe page address into a0 and save all the user registers, including `a0` from `sscratch`
  * trapframe contains kernel stack address, cpu's hartid,`usertrap` function address,kernel pagetable address
    * retrieve these values
    * call `usertrap`
* usertrap: determine the cause of the trap,process it and return
  * change `stvec`
  * save `sepc`
  * if a syscall, call `syscall` to handle it
  * if a device interrupt, call `devintr` to handle it
  * if an exception, kill the process
  * adds 4 to the saved program counter
* usertrapret
  * set up control registers for future user traps
  * call userret
* userret
  * switch `satp` to the process's user page table
  * restore registers
  * sret
## Code: Calling System Calls
```C
// place arguments
la a0, init
la a1, argv
// place syscall number
li a7, SYS_exec
// call 
ecall
// return value will be placed in a0
```

## Code: System Call Arguments
refer to `kernel/syscall.c` for more detail.
`argint`, `argaddr` and `argfd` retrieve the n'th system call arguments from the trapframe as an int, an addr and a file descriptor.They all call `argraw` to retrieve the appropriate saved user register.

To use pointers safely, use function like `fetchstr` etc.

## Traps from kernel space
* kernelvec:
  * kernel points `stvec` to the code at `kernelvec`(`kernel/kernelvec.S:12`)
  * pushes all 32 registers to its kernel stack
  * call `kerneltrap`(`kernel/trap.c:135`)
* kerneltrap
  * call devintr to check if a device interrupt and handle it
  * if not, it must be an exception, call `panic`
  * if a timer interrupt, call `yield` to give other threads to run
* restore registers from stack and `sret`

> Xv6 sets a CPU’s `stvec` to `kernelvec` when that CPU enters the kernel from user space; you can see this in usertrap (kernel/trap.c:29). There’s **a window of time when the kernel has started executing but `stvec` is still set to `uservec`**, and it’s crucial that no device interrupt occur during that window.
## Page-Fault Exceptions
Xv6's way of responding to exceptions:
* if from user space: kill the process
* if from kernel: call `panic`

The combination of **page tables and page faults** opens up a wide range of interesting possibilities
- Copy-on-Write fork
- Lazy Allocation
- Demand Paging
- ...
## Real World
if kernel memory were mapped into every process’s user page table
* eliminate the need for special trampoline pages
* eliminate the need for a page table switch when trapping from user space into the kernel