# Notes on Working
## Phase 1:Naive Pipeline
### Fetch
States:
RST,SEND,WAIT,DONE,PCGEN

### Decode
States:
IDLE,DECODE,READ,DONE

### Execute
States:
IDLE
ALUEXE
BRANCH
RD,RDWAIT
WR,WRWAIT,
DONE

### WriteBack
Just 1 cycle writing regs.

### Performance
Simulation Fib:28351ns
Test:
- ID:37399
- 60M
- Stream:0.380s
- Matrix:0.805s
- Cryptonight:2.027s

## Phase 2:Add BP
### Steps
- Delete Done in Fetch
  - Simulation Fib:25101ns
- BTFNT
  - Simulation Fib:22051ns
### Performance
Test:
- ID:37619
- Stream:0.288s
- Matrix:0.641s
- Cryptonight:1.617s

approximately 20% up.

## Phase 3:Accelerate Consuming
### Steps
- Delete Done in Exe
  - Simulation Fib:20771ns
- Coerse DECODE-READ in one State in Decode
  - Simulation Fib:20761ns
  - not up,need to Accelerate Fetch
- Use a 4 word Inst buffer
  - Simulation Fib:19811ns
  - 
Find the Bottleneck
- Fetch couldn't catch up with Decode
  - Decode will keep Idle for some cycles
- Execute couldn't catch up with Decode
  - Decode will keep req for some cycles

When these 2 bottleneck are dealt with,revise to multi-cycle memory access

## Phase 4:Accelerate Fetching Instructions
In the previous phase,a 4-word ibuffer was not so useful as expected.
It's because when fetching a line,it takes 4 cycles.Surely the req will keep 4 acks but,when the final ack comes,it tooks another 4 cycles,which will cause Decode to be Idle.
To deal with this,the next line should come in during the Decode was processing a req.This requires **Prefetch**.

- Suppose **A[0:3] and B[0:3]** are two adjacent lines.
- when access **A[0]**, miss, it takes 4 cycles to get the **A[0:3]**
- when `rvalid`,the bus becomes busy and get **B[0:3]** in the next 4 cycles
- It will look like the following table:

| Cycles: | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| States: | Get(A[0]) | Get(A[1]) | Get(A[2]) | Get(A[3]) | Get(B[0]) | Get(B[1]) | Get(B[2]) | Get(B[3]) |  |
| IF output: |  |  |  |  | A[0] | A[1] | A[2] | A[3] | B[0] |

- Thus constructs a pipeline and instruction flow will never stop.

## Phase 5:Accelerate Executing Instructions
Mainly caused by Load/Store.
In current version,Load/Store will make Decode's req hold for 2 cycles.
### Steps
1. Try to skip the address calculation cycle,merge it into `when IDLE has req`
   - Simulation Fib:18851ns.Good
   - but not solve the problem,when access the memory is more than 1 cycle,it will fail.
2. Using Techniques in Phase 4.Add a dataBuffer and try to pipeline
   - pipeline works well if read/write different memory

## Phase 6:Accelerate Decode


## Suspending
decode ack too fast cause 