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
Each RISC-V CPU has a set of **control registers** that the kernel writes to tell the CPU how to handle traps, and that the kernel can read to find out about a trap that has occurred.


1. If the trap is a device interrupt, and the `sstatus` SIE bit is clear, don’t do any of the following.
2. Disable interrupts by clearing the SIE bit in `sstatus`.
3. Copy the pc to `sepc`.
4. Save the current mode (user or supervisor) in the SPP bit in `sstatus`.
5. Set `scause` to reflect the trap’s cause.
6. Set the mode to supervisor.
7. Copy `stvec` to the pc.
8. Start executing at the new pc.

the RISC-V hardware does not switch page tables when it forces a trap.
## Traps from user space
The high-level path:
- `uservec`(`kernel/trampoline.S`)
- `usertrap`(`kernel/trap.c`)
- `usertrapret`(`kernel/trap.c`)
- `userret`(`kernel/trampoline.S`)

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
## Traps from kernel space
## Page-Fault Exceptions
## Real World